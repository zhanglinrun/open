"""
深度特征分布优化模块 (4.2 核心)

================================================================================
  核心目标:
    对库内无偏样本的切片指纹进行 **解耦** 分布优化, 使得:
      1. 类内指纹更紧凑 (intra-class compactness)
      2. 类间指纹更分离 (inter-class separation)
      3. 对抗边界更规整 (adversarial boundary regularization)
      4. 开集拒判能力增强 (伪未知空间约束)

  与分类器模式的关键区别 (会议纪要 4.2):
    - 分类器模式: 优化的是分类器权重, 特征是副产物
    - 孪生模式:   直接优化切片指纹的分布, 指纹本身就是目标

  解耦设计 (会议纪要 2.2②):
    - 冻结 backbone (ResNet18, ~11M 参数): 视觉特征提取不变
    - 只优化 embed_head + classifier + proj_head (~116K 参数): 指纹映射优化
    - 这实现了 "解耦特征分布优化": backbone 提取 ≠ 指纹映射, 二者分别优化
    - 好处: 训练稳定、收敛快、不破坏 Phase A 的视觉表达

  优化策略 (SupCon + CosFace + Center + PseudoUnknown):
    Phase 1 - 对抗边界规整化:
      SupCon + CosFace + CenterLoss 联合优化指纹映射,
      让类内更紧凑、类间更分离、CosFace margin 构成对抗边界
    Phase 2 - 伪未知空间约束:
      在 Phase 1 基础上增加伪未知样本约束,
      训练指纹映射推远伪未知特征, 增强开集拒判能力

  安全机制:
    - 每轮监控 Fisher ratio (类间散度 / 类内散度)
    - Fisher 下降超过阈值则 early stop 并回滚到最佳 checkpoint
    - 训练结束后自动加载 Fisher 最高的权重

  可视化:
    在优化前后生成 t-SNE 图, 展示优化效果
================================================================================
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_library import FeatureLibrary
from siamese_losses import SupConLoss, CenterLoss


# =============================================================================
# 特征分布度量计算
# =============================================================================
class DistributionMetrics:
    """
    特征分布质量度量

    用于量化优化前后的分布变化:
      - intra_compactness: 类内紧凑度 (越小越好)
      - inter_separation: 类间分离度 (越大越好)
      - silhouette_score: 轮廓系数 (-1~1, 越大越好)
      - fisher_ratio: Fisher 判别比 (越大越好)
    """

    @staticmethod
    @torch.no_grad()
    def compute_all(feats: torch.Tensor,
                    labels: torch.Tensor,
                    prototypes: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        计算所有分布度量

        Args:
            feats: 切片指纹 [N, D]
            labels: 标签 [N]
            prototypes: 类原型 [K, D] (可选, 不提供则从 feats 计算)
        Returns:
            度量字典
        """
        metrics = {}

        unique_labels = labels.unique()
        K = unique_labels.numel()

        if K < 2:
            return {"intra_compactness": 0.0, "inter_separation": 0.0,
                    "silhouette_approx": 0.0, "fisher_ratio": 0.0}

        # 计算类原型 (如果未提供)
        if prototypes is None:
            prototypes = torch.zeros(K, feats.size(1), device=feats.device)
            for i, c in enumerate(unique_labels):
                mask = labels == c
                prototypes[i] = F.normalize(feats[mask].mean(dim=0), dim=0)

        # 1. 类内紧凑度: 平均类内余弦距离
        intra_dists = []
        for i, c in enumerate(unique_labels):
            mask = labels == c
            if mask.sum() < 2:
                continue
            c_feats = F.normalize(feats[mask], dim=1)
            c_proto = prototypes[i:i+1]
            sims = torch.matmul(c_feats, F.normalize(c_proto, dim=1).T).squeeze(1)
            intra_dists.append(float((1.0 - sims).mean().item()))

        metrics["intra_compactness"] = np.mean(intra_dists) if intra_dists else 0.0

        # 2. 类间分离度: 原型两两余弦距离的平均值
        p_norm = F.normalize(prototypes, dim=1)
        sim_matrix = torch.matmul(p_norm, p_norm.T)
        mask_off = ~torch.eye(K, dtype=torch.bool, device=feats.device)
        inter_sims = sim_matrix[mask_off]
        metrics["inter_separation"] = float((1.0 - inter_sims).mean().item())

        # 3. 近似轮廓系数
        silhouettes = []
        feats_norm = F.normalize(feats, dim=1)
        for idx in range(min(feats.size(0), 500)):  # 采样加速
            c = labels[idx]
            same_mask = labels == c
            diff_mask = ~same_mask

            if same_mask.sum() < 2 or not diff_mask.any():
                continue

            fp = feats_norm[idx:idx+1]

            # a(i): 平均类内距离
            sims_same = torch.matmul(fp, feats_norm[same_mask].T).squeeze(0)
            a_i = float((1.0 - sims_same).mean().item())

            # b(i): 最小平均类间距离
            min_b = float("inf")
            for j, c2 in enumerate(unique_labels):
                if c2 == c:
                    continue
                c2_mask = labels == c2
                if not c2_mask.any():
                    continue
                sims_c2 = torch.matmul(fp, feats_norm[c2_mask].T).squeeze(0)
                b_c2 = float((1.0 - sims_c2).mean().item())
                min_b = min(min_b, b_c2)

            if min_b < float("inf"):
                s_i = (min_b - a_i) / max(a_i, min_b, 1e-8)
                silhouettes.append(s_i)

        metrics["silhouette_approx"] = np.mean(silhouettes) if silhouettes else 0.0

        # 4. Fisher 判别比: Sb / Sw
        overall_mean = feats.mean(dim=0)
        Sb = 0.0
        Sw = 0.0
        for i, c in enumerate(unique_labels):
            mask = labels == c
            c_feats = feats[mask]
            n_c = c_feats.size(0)
            c_mean = c_feats.mean(dim=0)

            Sb += n_c * float((c_mean - overall_mean).pow(2).sum().item())
            Sw += float((c_feats - c_mean.unsqueeze(0)).pow(2).sum().item())

        metrics["fisher_ratio"] = Sb / max(Sw, 1e-8)

        return metrics


# =============================================================================
# 特征分布优化器 (解耦版: 冻结 backbone, 优化指纹映射)
# =============================================================================
class FeatureDistributionOptimizer:
    """
    切片指纹分布优化器 (解耦特征分布优化)

    核心思想 (会议纪要 2.2② + 4.2):
      Phase A 用 150 轮训练让整个模型 (backbone + embed_head + classifier)
      充分收敛. Phase D 不再重复训练整个模型, 而是:
        - 冻结 backbone: 视觉特征提取已经学好, 不再改动
        - 只优化 embed_head + classifier: 指纹映射空间的精细调整
      这就是 "解耦": 视觉特征提取 和 指纹空间映射 分别优化.

    训练参数量对比:
      Phase A: 全模型 ~11.3M 参数 (lr=1e-3, 150 轮)
      Phase D: embed_head+classifier ~66K 参数 (lr=5e-4, 20 轮)
      参数量减少 99.4%, 训练更稳定、更可控

    优化流程:
      Phase 1 (对抗边界规整化, ~15 轮):
        SupCon + CosFace + Center 联合优化, 规整化类边界
      Phase 2 (伪未知约束, ~5 轮):
        在 Phase 1 基础上增加伪未知样本推远约束

    Args:
        model: 孪生网络 (SiameseFingerprintWithClassifier)
        feature_library: 特征库
        device: 计算设备
        lr: 优化学习率 (只训练小模块, 可用正常学习率)
        supcon_temperature: SupCon Loss 温度参数
        cls_weight: CosFace 分类损失权重
        center_weight: Center Loss 权重
        pseudo_weight: 伪未知约束权重
        fisher_drop_limit: Fisher ratio 最大允许下降比例
    """

    def __init__(self,
                 model: nn.Module,
                 feature_library: FeatureLibrary,
                 device: torch.device,
                 lr: float = 5e-4,
                 supcon_temperature: float = 0.12,
                 cls_weight: float = 0.5,
                 center_weight: float = 0.01,
                 pseudo_weight: float = 0.1,
                 fisher_drop_limit: float = 0.10,
                 optimize_mode: bool = True):
        """
        Args:
            optimize_mode: 是否进入优化模式 (冻结 backbone, 创建优化器).
                           False 则仅用于 collect_features 等工具方法.
        """
        self.model = model
        self.library = feature_library
        self.device = device
        self.lr = lr
        self.fisher_drop_limit = fisher_drop_limit
        self.optimize_mode = optimize_mode

        self.num_classes = feature_library.num_classes
        self.embed_dim = feature_library.embed_dim

        # 与 Phase A 相同的损失函数
        self.supcon_loss = SupConLoss(temperature=supcon_temperature).to(device)
        self.center_loss = CenterLoss(self.num_classes, self.embed_dim, lr=0.5).to(device)

        self.cls_weight = cls_weight
        self.center_weight = center_weight
        self.pseudo_weight = pseudo_weight

        self.n_trainable = 0
        self.n_total = sum(p.numel() for p in model.parameters())
        self.optimizer = None
        self.scheduler = None

        if optimize_mode:
            # ---- 解耦: 冻结 backbone, 只优化 embed_head + classifier ----
            self._freeze_backbone()

            # 优化器: 只含 requires_grad=True 的参数 (embed_head + classifier + proj_head)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=20
            )

            # 记录可训练参数量
            self.n_trainable = sum(p.numel() for p in trainable_params)

    def _freeze_backbone(self):
        """冻结 backbone (ResNet18), 只留 embed_head + classifier + proj_head 可训练"""
        # 冻结 backbone
        for p in self.model.siamese.backbone.parameters():
            p.requires_grad = False

        # 确保 embed_head, classifier, proj_head 可训练
        for p in self.model.siamese.embed_head.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True
        if self.model.siamese.proj_head is not None:
            for p in self.model.siamese.proj_head.parameters():
                p.requires_grad = True

    def _unfreeze_backbone(self):
        """恢复 backbone 为可训练状态 (Phase D 结束后调用)"""
        for p in self.model.siamese.backbone.parameters():
            p.requires_grad = True

    # -----------------------------------------------------------------
    # 收集特征 (用于度量和可视化)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def collect_features(self,
                         loader: torch.utils.data.DataLoader) \
            -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        收集所有样本的切片指纹

        Returns:
            (feats [N, D], labels [N], orig_labels [N])
        """
        self.model.eval()
        all_feats, all_labels, all_origs = [], [], []

        for x, y, orig_labels, _ in loader:
            x = x.to(self.device)
            fingerprints = self.model.extract_fingerprint(x)
            all_feats.append(fingerprints.cpu())
            all_labels.append(y)
            all_origs.extend(orig_labels.tolist())

        feats = torch.cat(all_feats, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return feats, labels, all_origs

    # -----------------------------------------------------------------
    # 单轮训练
    # -----------------------------------------------------------------
    def _train_one_epoch(self,
                         train_loader: torch.utils.data.DataLoader,
                         use_pseudo: bool = False,
                         prototypes: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        单轮训练逻辑

        Args:
            train_loader: 训练数据加载器
            use_pseudo: 是否启用伪未知约束
            prototypes: 类原型 [K, D] (伪未知约束需要)
        Returns:
            本轮各损失的平均值
        """
        self.model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for x, y, _, _ in train_loader:
            mask = (y >= 0)
            if mask.sum() < 4:
                continue

            x = x[mask].to(self.device)
            y = y[mask].to(self.device)

            # 使用与 Phase A 相同的前向传播 (含 CosFace margin)
            out = self.model(x, labels=y)
            fingerprints = out["fingerprint"]

            # 1. SupCon Loss (核心: 类内紧凑, 类间分离)
            l_supcon, sc_stats = self.supcon_loss(fingerprints, y)

            # 2. CosFace 分类损失 (对抗边界规整化)
            l_cls = torch.tensor(0.0, device=self.device)
            if "logits" in out and self.cls_weight > 0:
                l_cls = F.cross_entropy(out["logits"], y)

            # 3. Center Loss (拉向类中心, 对抗随机性)
            l_center = self.center_loss(fingerprints, y)

            # 4. 伪未知约束 (可选: Phase 2 启用)
            l_pseudo = torch.tensor(0.0, device=self.device)
            if use_pseudo and prototypes is not None and fingerprints.size(0) > 2:
                perm = torch.randperm(fingerprints.size(0), device=self.device)
                diff_mask = (y != y[perm])
                if diff_mask.sum() > 0:
                    # 用 detach 防止伪未知梯度回传到正常样本
                    f1 = fingerprints[diff_mask].detach()
                    f2 = fingerprints[perm][diff_mask].detach()
                    lam = torch.rand(f1.size(0), 1, device=self.device) * 0.6 + 0.2
                    f_pseudo = F.normalize(lam * f1 + (1 - lam) * f2, dim=1)

                    # 伪未知应远离所有原型
                    sim_to_protos = torch.matmul(f_pseudo, prototypes.T)
                    max_sim = sim_to_protos.max(dim=1)[0]
                    l_pseudo = F.relu(max_sim - 0.5).mean()

            loss = (l_supcon
                    + self.cls_weight * l_cls
                    + self.center_weight * l_center
                    + self.pseudo_weight * l_pseudo)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()

            bs = x.size(0)
            epoch_losses["supcon"] += l_supcon.item() * bs
            epoch_losses["cls"] += l_cls.item() * bs
            epoch_losses["center"] += l_center.item() * bs
            epoch_losses["pseudo"] += l_pseudo.item() * bs
            epoch_losses["total"] += loss.item() * bs
            epoch_losses["sim_pos"] += sc_stats.get("mean_sim_pos", 0) * bs
            epoch_losses["n_samples"] += bs
            n_batches += 1

        self.scheduler.step()

        n = max(epoch_losses["n_samples"], 1)
        return {
            "loss_supcon": epoch_losses["supcon"] / n,
            "loss_cls": epoch_losses["cls"] / n,
            "loss_center": epoch_losses["center"] / n,
            "loss_pseudo": epoch_losses["pseudo"] / n,
            "loss_total": epoch_losses["total"] / n,
            "sim_pos": epoch_losses["sim_pos"] / n,
        }

    # -----------------------------------------------------------------
    # 完整优化流程 (对抗边界规整化 + 伪未知约束)
    # -----------------------------------------------------------------
    def optimize_full(self,
                      train_loader: torch.utils.data.DataLoader,
                      test_loader: torch.utils.data.DataLoader,
                      phase1_epochs: int = 15,
                      phase2_epochs: int = 5,
                      phase3_epochs: int = 0,
                      eval_interval: int = 5,
                      logger=None) -> Dict:
        """
        完整解耦优化流程

        Phase 1 (对抗边界规整化): SupCon + CosFace + Center
        Phase 2 (伪未知约束): SupCon + CosFace + Center + Pseudo

        每 eval_interval 轮评估 Fisher ratio, 保存最佳 checkpoint.
        Fisher 下降超过阈值则 early stop 并回滚.

        Args:
            train_loader: 训练 DataLoader
            test_loader: 测试 DataLoader
            phase1_epochs: Phase 1 轮数 (对抗边界规整化)
            phase2_epochs: Phase 2 轮数 (伪未知约束)
            phase3_epochs: 未使用 (保持接口兼容)
            eval_interval: 评估间隔
            logger: 日志记录器
        Returns:
            包含各阶段历史和前后度量的字典
        """
        _log = logger.info if logger else print

        result = {"phases": {}, "metrics_before": {}, "metrics_after": {}}
        total_epochs = phase1_epochs + phase2_epochs

        # ---- 记录解耦信息 ----
        _log(f"\n[解耦特征分布优化] 冻结 backbone, 只训练指纹映射层")
        _log(f"  可训练参数: {self.n_trainable:,} / {self.n_total:,} "
             f"({self.n_trainable/self.n_total*100:.1f}%)")
        _log(f"  Phase 1 (对抗边界规整化): {phase1_epochs} 轮")
        _log(f"  Phase 2 (伪未知约束): {phase2_epochs} 轮")
        _log(f"  lr={self.lr}, Fisher drop limit={self.fisher_drop_limit*100:.0f}%")

        # ---- 优化前度量 ----
        _log("\n[Before Optimization] 收集特征...")
        feats_before, labels_before, _ = self.collect_features(test_loader)
        known_mask = labels_before >= 0
        m_before = {}
        if known_mask.any():
            m_before = DistributionMetrics.compute_all(
                feats_before[known_mask], labels_before[known_mask]
            )
            result["metrics_before"] = m_before
            _log(f"  Fisher={m_before.get('fisher_ratio', 0):.1f}, "
                 f"intra={m_before.get('intra_compactness', 0):.6f}, "
                 f"inter={m_before.get('inter_separation', 0):.4f}, "
                 f"silhouette={m_before.get('silhouette_approx', 0):.4f}")

        fisher_baseline = m_before.get("fisher_ratio", 0.0)
        best_fisher = fisher_baseline
        best_state = copy.deepcopy(self.model.state_dict())

        # ---- 训练历史 ----
        history = {"loss_total": [], "loss_supcon": [], "loss_cls": [],
                   "loss_pseudo": [], "fisher": [], "phase_label": []}

        # ---- 获取初始原型 (Phase 2 用) ----
        prototypes = self.library.get_prototypes(device=self.device)
        early_stopped = False

        for epoch in range(1, total_epochs + 1):
            # 判断当前阶段
            in_phase2 = (epoch > phase1_epochs)
            phase_name = "P2(+pseudo)" if in_phase2 else "P1(boundary)"

            # 训练一轮
            stats = self._train_one_epoch(
                train_loader,
                use_pseudo=in_phase2,
                prototypes=prototypes if in_phase2 else None,
            )

            # 记录历史
            history["loss_total"].append(stats["loss_total"])
            history["loss_supcon"].append(stats["loss_supcon"])
            history["loss_cls"].append(stats["loss_cls"])
            history["loss_pseudo"].append(stats["loss_pseudo"])
            history["phase_label"].append(phase_name)

            # 定期评估 Fisher ratio
            do_eval = (epoch % eval_interval == 0) or (epoch == total_epochs)
            if do_eval:
                feats_ep, labels_ep, _ = self.collect_features(test_loader)
                known_mask_ep = labels_ep >= 0
                ep_metrics = DistributionMetrics.compute_all(
                    feats_ep[known_mask_ep], labels_ep[known_mask_ep]
                )
                cur_fisher = ep_metrics.get("fisher_ratio", 0.0)
                history["fisher"].append(cur_fisher)

                # 保存最佳
                if cur_fisher >= best_fisher:
                    best_fisher = cur_fisher
                    best_state = copy.deepcopy(self.model.state_dict())

                drop_pct = (fisher_baseline - cur_fisher) / max(fisher_baseline, 1e-8) * 100

                _log(f"  Epoch {epoch:02d}/{total_epochs} [{phase_name}]: "
                     f"loss={stats['loss_total']:.4f} "
                     f"(sc={stats['loss_supcon']:.4f} "
                     f"cls={stats['loss_cls']:.4f} "
                     f"pseudo={stats['loss_pseudo']:.4f}) "
                     f"| Fisher={cur_fisher:.1f} ({drop_pct:+.1f}%)")

                # Early stop: Fisher 下降过多
                if fisher_baseline > 0:
                    drop_ratio = (fisher_baseline - cur_fisher) / fisher_baseline
                    if drop_ratio > self.fisher_drop_limit:
                        _log(f"  [Early Stop] Fisher dropped {drop_ratio*100:.1f}% "
                             f"(> {self.fisher_drop_limit*100:.0f}% limit), "
                             f"rolling back to best (Fisher={best_fisher:.1f})")
                        self.model.load_state_dict(best_state)
                        early_stopped = True
                        break
            else:
                history["fisher"].append(None)  # 非评估轮
                _log(f"  Epoch {epoch:02d}/{total_epochs} [{phase_name}]: "
                     f"loss={stats['loss_total']:.4f} "
                     f"(sc={stats['loss_supcon']:.4f} "
                     f"cls={stats['loss_cls']:.4f} "
                     f"pseudo={stats['loss_pseudo']:.4f})")

            # Phase 2 开始时更新原型
            if epoch == phase1_epochs:
                self._update_prototypes_from_loader(train_loader)
                prototypes = self.library.get_prototypes(device=self.device)
                _log(f"\n  --- Phase 2: 启用伪未知约束 ---")

        result["phases"]["history"] = history

        # ---- 确保使用最佳权重 ----
        if not early_stopped:
            # 检查最终 Fisher 是否比 best 差
            feats_final, labels_final, _ = self.collect_features(test_loader)
            km_final = labels_final >= 0
            final_metrics = DistributionMetrics.compute_all(
                feats_final[km_final], labels_final[km_final]
            )
            final_fisher = final_metrics.get("fisher_ratio", 0.0)

            if final_fisher < best_fisher:
                _log(f"  [Rollback] Final Fisher={final_fisher:.1f} < "
                     f"Best Fisher={best_fisher:.1f}, loading best checkpoint")
                self.model.load_state_dict(best_state)

        # ---- 优化后度量 ----
        _log("\n[After Optimization] 收集特征...")
        feats_after, labels_after, _ = self.collect_features(test_loader)
        known_mask = labels_after >= 0
        m_after = {}
        if known_mask.any():
            m_after = DistributionMetrics.compute_all(
                feats_after[known_mask], labels_after[known_mask]
            )
            result["metrics_after"] = m_after
            _log(f"  Fisher={m_after.get('fisher_ratio', 0):.1f}, "
                 f"intra={m_after.get('intra_compactness', 0):.6f}, "
                 f"inter={m_after.get('inter_separation', 0):.4f}, "
                 f"silhouette={m_after.get('silhouette_approx', 0):.4f}")

        # ---- 打印改善 ----
        _log("\n" + "=" * 60)
        _log("[Improvement Summary]")
        for key in ["intra_compactness", "inter_separation",
                     "silhouette_approx", "fisher_ratio"]:
            before = m_before.get(key, 0)
            after = m_after.get(key, 0)
            delta = after - before
            direction = "↓" if key == "intra_compactness" else "↑"
            better = (delta < 0) if key == "intra_compactness" else (delta > 0)
            symbol = "✓" if better else "✗"
            _log(f"  {key}: {before:.4f} → {after:.4f} "
                 f"({direction} {abs(delta):.4f}) {symbol}")
        _log("=" * 60)

        # ---- 解冻 backbone (恢复全模型训练能力, 供 Phase E 使用) ----
        self._unfreeze_backbone()
        _log("  [解冻] backbone 已恢复可训练状态")

        return result

    # -----------------------------------------------------------------
    # 辅助方法
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _update_prototypes_from_loader(self, loader):
        """从 loader 更新特征库原型"""
        self.model.eval()
        feat_accum = defaultdict(list)

        for x, y, _, _ in loader:
            mask = (y >= 0)
            if not mask.any():
                continue

            x = x[mask].to(self.device)
            y = y[mask]

            fingerprints = self.model.extract_fingerprint(x).cpu()

            for i in range(fingerprints.size(0)):
                cid = int(y[i].item())
                feat_accum[cid].append(fingerprints[i])

        for cid, feats_list in feat_accum.items():
            if cid in self.library.entries:
                feats = torch.stack(feats_list)
                entry = self.library.entries[cid]
                entry.features = [f for f in feats]
                entry.count = len(feats_list)
                entry._stats_dirty = True
                entry.compute_stats()

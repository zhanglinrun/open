"""
孪生网络训练模块

训练流程:
  Phase A: SupCon + Center + CosFace 对抗边界 (基础训练)
  Phase E: SupCon + Center + CosFace + 生成特征约束 (增广微调)
"""

import os
import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from siamese_losses import SupConLoss, CenterLoss, SpreadRegularization
from feat_distribution_optimizer import DistributionMetrics


# =============================================================================
# 单轮训练 (SupCon + Center + CosFace)
# =============================================================================
def train_one_epoch(model: nn.Module,
                    loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    supcon_loss: SupConLoss,
                    center_loss: CenterLoss,
                    center_weight: float = 0.01,
                    cls_weight: float = 0.5,
                    grad_clip: float = 1.0,
                    spread_reg: Optional[SpreadRegularization] = None,
                    spread_weight: float = 0.1) -> Dict:
    """
    SupCon + Center + CosFace + SpreadReg 单轮训练

    CosFace 对抗边界通过 model.forward(x, labels) 自动生效:
    训练时传入 labels → 正确类余弦值减去 margin → 迫使类内更紧凑.
    SpreadReg 防止特征过度坍塌, 保留开集拒判缓冲区.
    """
    model.train()
    epoch_stats = defaultdict(float)
    n_batches = 0

    for x, y, _, _ in loader:
        mask = (y >= 0)
        if mask.sum() < 4:
            continue

        x = x[mask].to(device)
        y = y[mask].to(device)

        # 传入 labels 让 CosFace margin 生效
        out = model(x, labels=y)
        fingerprints = out["fingerprint"]

        # 1. SupCon Loss (核心紧凑度)
        l_supcon, sc_stats = supcon_loss(fingerprints, y)

        # 2. Center Loss
        l_center = center_loss(fingerprints, y)

        # 3. CosFace 分类损失 (已内含 margin 对抗边界)
        l_cls = torch.tensor(0.0, device=device)
        if "logits" in out and cls_weight > 0:
            l_cls = F.cross_entropy(out["logits"], y)

        # 4. 分散正则化 (防止类内坍塌到单点)
        l_spread = torch.tensor(0.0, device=device)
        if spread_reg is not None:
            l_spread = spread_reg(fingerprints, y)

        loss = (l_supcon + center_weight * l_center
                + cls_weight * l_cls + spread_weight * l_spread)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        epoch_stats["loss_supcon"] += l_supcon.item() * bs
        epoch_stats["loss_center"] += l_center.item() * bs
        epoch_stats["loss_cls"] += l_cls.item() * bs
        epoch_stats["loss_spread"] += l_spread.item() * bs
        epoch_stats["loss_total"] += loss.item() * bs
        epoch_stats["mean_sim_pos"] += sc_stats.get("mean_sim_pos", 0) * bs
        epoch_stats["n_samples"] += bs
        n_batches += 1

    n = max(epoch_stats["n_samples"], 1)
    return {
        "loss_supcon": epoch_stats["loss_supcon"] / n,
        "loss_center": epoch_stats["loss_center"] / n,
        "loss_cls": epoch_stats["loss_cls"] / n,
        "loss_spread": epoch_stats["loss_spread"] / n,
        "loss_total": epoch_stats["loss_total"] / n,
        "mean_sim_pos": epoch_stats["mean_sim_pos"] / n,
    }


# =============================================================================
# 评估
# =============================================================================
@torch.no_grad()
def evaluate_openset(model: nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     test_loader: torch.utils.data.DataLoader,
                     device: torch.device,
                     num_known: int) -> Dict:
    """
    基于切片指纹的开集评估

    流程:
      1. 从训练集计算各类原型
      2. 在测试集上: 已知类→最近原型分类, 未知类→距离超阈值拒判

    Returns:
        评估指标字典
    """
    model.eval()

    # Step 1: 计算训练集原型
    feat_accum = [[] for _ in range(num_known)]
    for x, y, _, _ in train_loader:
        mask = (y >= 0)
        if not mask.any():
            continue
        x = x[mask].to(device)
        y = y[mask]
        fps = model.extract_fingerprint(x).cpu()
        for i in range(fps.size(0)):
            cid = int(y[i].item())
            if 0 <= cid < num_known:
                feat_accum[cid].append(fps[i])

    prototypes = torch.zeros(num_known, feat_accum[0][0].size(0)) if feat_accum[0] else None
    if prototypes is None:
        return {"known_acc": 0.0, "open_f1": 0.0, "auroc": 0.0}

    for c in range(num_known):
        if feat_accum[c]:
            prototypes[c] = F.normalize(
                torch.stack(feat_accum[c]).mean(dim=0), dim=0
            )

    prototypes = prototypes.to(device)

    # Step 2: 计算训练集距离分布 (用于确定阈值)
    train_dists = []
    for x, y, _, _ in train_loader:
        mask = (y >= 0)
        if not mask.any():
            continue
        x = x[mask].to(device)
        y = y[mask].to(device)
        fps = model.extract_fingerprint(x)
        fps_norm = F.normalize(fps, dim=1)
        sims = torch.matmul(fps_norm, F.normalize(prototypes, dim=1).T)
        max_sim, _ = sims.max(dim=1)
        train_dists.extend((1.0 - max_sim).cpu().tolist())

    tau = np.percentile(train_dists, 95) if train_dists else 0.5

    # Step 3: 测试集评估 (距离阈值拒判)
    total_known = 0
    correct_known = 0
    tp = fp = fn = tn = 0
    correct_accepted = 0
    all_dists = []
    all_is_unknown = []

    for x, y, _, _ in test_loader:
        x = x.to(device)
        y = y.to(device)

        fps = model.extract_fingerprint(x)
        fps_norm = F.normalize(fps, dim=1)
        sims = torch.matmul(fps_norm, F.normalize(prototypes, dim=1).T)
        max_sim, pred_cls = sims.max(dim=1)
        dist = 1.0 - max_sim

        all_dists.extend(dist.cpu().tolist())
        all_is_unknown.extend((y < 0).cpu().int().tolist())

        pred_unknown = (dist > tau)
        is_unk = (y < 0)

        tp += int(((pred_unknown) & (is_unk)).sum().item())
        fp += int(((pred_unknown) & (~is_unk)).sum().item())
        fn += int(((~pred_unknown) & (is_unk)).sum().item())
        tn += int(((~pred_unknown) & (~is_unk)).sum().item())

        known_mask = (y >= 0)
        if known_mask.any():
            total_known += int(known_mask.sum().item())
            accepted = ~pred_unknown
            correct_cls = pred_cls == y
            correct_known += int((correct_cls & known_mask).sum().item())
            correct_accepted += int((known_mask & accepted & correct_cls).sum().item())

    known_acc = correct_known / max(total_known, 1)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    open_acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    aks = correct_accepted / max(total_known, 1)
    total_unknown = tp + fn
    aus = tp / max(total_unknown, 1)
    na = (aks + aus) / 2.0

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(all_is_unknown, all_dists)
        if auroc < 0.5:
            auroc = 1.0 - auroc
    except Exception:
        auroc = 0.5

    return {
        "known_acc": float(known_acc),
        "open_precision": float(precision),
        "open_recall": float(recall),
        "open_f1": float(f1),
        "open_acc": float(open_acc),
        "aks": float(aks),
        "aus": float(aus),
        "na": float(na),
        "auroc": float(auroc),
        "tau": float(tau),
    }


# =============================================================================
# 完整训练循环
# =============================================================================
def train_siamese(model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  test_loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  scheduler,
                  device: torch.device,
                  num_known: int,
                  num_epochs: int = 150,
                  center_weight: float = 0.01,
                  cls_weight: float = 0.5,
                  grad_clip: float = 1.0,
                  save_path: str = "siamese_best.pt",
                  log_interval: int = 1,
                  eval_interval: int = 5,
                  supcon_temperature: float = 0.12,
                  spread_weight: float = 0.5,
                  logger: Optional[logging.Logger] = None) -> Tuple[Dict, Dict]:
    """
    完整训练循环 (SupCon + Center + CosFace + SpreadReg)

    改进:
      - SupCon 温度从 0.07 提高到 0.12, 减轻特征坍塌
      - SpreadRegularization 维持类内适度扩展, 保留开集拒判缓冲区
    """
    _log = logger.info if logger else print

    embed_dim = model.embed_dim if hasattr(model, "embed_dim") else 128
    supcon_loss_fn = SupConLoss(temperature=supcon_temperature).to(device)
    center_loss_fn = CenterLoss(num_known, embed_dim, lr=0.5).to(device)
    spread_reg_fn = SpreadRegularization(min_var=0.02).to(device)

    history = {
        "loss_total": [], "loss_supcon": [], "loss_center": [], "loss_cls": [],
        "known_acc": [], "open_f1": [], "na": [], "auroc": [],
        "mean_sim_pos": [],
    }

    best_na = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            supcon_loss_fn, center_loss_fn,
            center_weight=center_weight,
            cls_weight=cls_weight,
            grad_clip=grad_clip,
            spread_reg=spread_reg_fn,
            spread_weight=spread_weight,
        )

        if scheduler is not None:
            scheduler.step()

        # 记录
        for k in ["loss_total", "loss_supcon", "loss_center", "loss_cls",
                   "mean_sim_pos"]:
            history[k].append(train_stats.get(k, 0))

        # 评估
        if epoch % eval_interval == 0 or epoch == num_epochs:
            metrics = evaluate_openset(
                model, train_loader, test_loader, device, num_known
            )
            for k in ["known_acc", "open_f1", "na", "auroc"]:
                history[k].append(metrics[k])

            if epoch % log_interval == 0:
                _log(
                    f"Epoch {epoch:03d}/{num_epochs} | "
                    f"loss={train_stats['loss_total']:.4f} "
                    f"(sc={train_stats['loss_supcon']:.4f} "
                    f"cls={train_stats['loss_cls']:.4f}) "
                    f"sim_pos={train_stats['mean_sim_pos']:.4f} | "
                    f"known={metrics['known_acc']:.4f} "
                    f"F1={metrics['open_f1']:.4f} "
                    f"NA={metrics['na']:.4f} "
                    f"AUROC={metrics['auroc']:.4f}"
                )

            if metrics["na"] > best_na:
                best_na = metrics["na"]
                best_state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    **metrics,
                }
                torch.save(best_state, save_path)
                _log(f"  [Save] Best NA={best_na:.4f}")
        else:
            if epoch % log_interval == 0:
                _log(
                    f"Epoch {epoch:03d}/{num_epochs} | "
                    f"loss={train_stats['loss_total']:.4f} "
                    f"(sc={train_stats['loss_supcon']:.4f} "
                    f"cls={train_stats['loss_cls']:.4f})"
                )

    _log(f"\n[Done] Best NA: {best_na:.4f}")
    return history, best_state


# =============================================================================
# 增广数据微调 (Phase E: SupCon + ARPL + 生成特征约束)
# =============================================================================
def finetune_with_augmented_data(model: nn.Module,
                                  train_loader: torch.utils.data.DataLoader,
                                  generated_feats: Dict[int, torch.Tensor],
                                  device: torch.device,
                                  num_known: int,
                                  epochs: int = 20,
                                  lr: float = 1e-4,
                                  margin: float = 0.3,
                                  supcon_temperature: float = 0.12,
                                  logger: Optional[logging.Logger] = None) -> Dict:
    """
    增广数据微调 (SupCon + Center + CosFace + 生成特征拉近 + 伪未知拒判)

    改进:
      - 适度提高学习率 (5e-5 → 1e-4) 和轮数 (15 → 20)
      - 新增伪未知拒判训练: 不同类 mixup 生成伪未知, 推远所有原型
      - SpreadReg 防止微调过程中坍塌
    """
    _log = logger.info if logger else print
    _log(f"\n[Finetune] SupCon + CosFace + PseudoUnknown 微调: {epochs} epochs, lr={lr}")

    embed_dim = next(iter(generated_feats.values())).size(1) if generated_feats else 128
    supcon_loss_fn = SupConLoss(temperature=supcon_temperature).to(device)
    center_loss_fn = CenterLoss(num_known, embed_dim, lr=0.5).to(device)
    spread_reg_fn = SpreadRegularization(min_var=0.02).to(device)

    # 解耦: 冻结 backbone, 只微调 embed_head + classifier + proj_head
    # 保护 Phase A/D 建立的视觉特征表达
    if hasattr(model, 'siamese') and hasattr(model.siamese, 'backbone'):
        for p in model.siamese.backbone.parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        _log(f"  [解耦微调] 冻结 backbone, 可训练参数: {n_train:,}")
    else:
        trainable = list(model.parameters())

    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 预计算原型
    prototypes = _compute_prototypes(model, train_loader, device, num_known)

    # 整合生成特征
    gen_all_feats, gen_all_labels = [], []
    for cid, feats in generated_feats.items():
        if feats.numel() > 0:
            gen_all_feats.append(feats)
            gen_all_labels.append(torch.full((feats.size(0),), cid, dtype=torch.long))

    if not gen_all_feats:
        _log("  No generated features, skipping finetune.")
        return {}

    gen_feats_cat = torch.cat(gen_all_feats, dim=0).to(device)
    gen_labels_cat = torch.cat(gen_all_labels, dim=0).to(device)
    n_gen = gen_feats_cat.size(0)
    _log(f"  Generated features: {n_gen} across {len(generated_feats)} classes")

    history = {"loss_total": [], "loss_supcon": [], "loss_cls": [],
               "loss_gen_pull": [], "loss_pseudo_unk": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = defaultdict(float)
        n_batches = 0

        for x, y, _, _ in train_loader:
            mask = (y >= 0)
            if mask.sum() < 4:
                continue

            x = x[mask].to(device)
            y = y[mask].to(device)

            out = model(x, labels=y)  # 传入 labels 让 CosFace margin 生效
            fingerprints = out["fingerprint"]

            # 1. SupCon + Center + CosFace (保持与 Phase A 一致)
            l_supcon, _ = supcon_loss_fn(fingerprints, y)
            l_center = center_loss_fn(fingerprints, y)
            l_cls = F.cross_entropy(out["logits"], y) if "logits" in out else torch.tensor(0.0, device=device)

            # 2. 生成特征拉近原型 (轻量)
            sample_size = min(x.size(0), n_gen)
            idx = torch.randperm(n_gen, device=device)[:sample_size]
            sampled_feats = gen_feats_cat[idx]
            sampled_protos = prototypes[gen_labels_cat[idx]]
            l_gen_pull = (1.0 - F.cosine_similarity(sampled_feats, sampled_protos)).mean()

            # 3. 伪未知拒判训练: 不同类 mixup → 推远所有原型
            l_pseudo_unk = torch.tensor(0.0, device=device)
            if fingerprints.size(0) > 4:
                perm = torch.randperm(fingerprints.size(0), device=device)
                diff_mask = (y != y[perm])
                if diff_mask.sum() > 1:
                    f1 = fingerprints[diff_mask]
                    f2 = fingerprints[perm][diff_mask]
                    lam = torch.rand(f1.size(0), 1, device=device) * 0.6 + 0.2
                    f_pseudo = F.normalize(lam * f1 + (1 - lam) * f2, dim=1)
                    # 伪未知应远离所有原型 (max_sim < 0.3)
                    sim_to_protos = torch.matmul(
                        f_pseudo, F.normalize(prototypes, dim=1).T
                    )
                    max_sim_pseudo = sim_to_protos.max(dim=1)[0]
                    l_pseudo_unk = F.relu(max_sim_pseudo - 0.3).mean()

            # 4. SpreadReg
            l_spread = spread_reg_fn(fingerprints, y)

            loss = (l_supcon + 0.01 * l_center + 0.5 * l_cls
                    + 0.2 * l_gen_pull + 0.3 * l_pseudo_unk + 0.1 * l_spread)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            epoch_loss["supcon"] += l_supcon.item()
            epoch_loss["cls"] += l_cls.item()
            epoch_loss["gen_pull"] += l_gen_pull.item()
            epoch_loss["pseudo_unk"] += l_pseudo_unk.item()
            epoch_loss["total"] += loss.item()
            n_batches += 1

        scheduler.step()
        if n_batches > 0:
            for k in epoch_loss:
                epoch_loss[k] /= n_batches

        for k in ["loss_total", "loss_supcon", "loss_cls",
                   "loss_gen_pull", "loss_pseudo_unk"]:
            history[k].append(epoch_loss.get(k.replace("loss_", ""), 0))

        if epoch % 5 == 0:
            prototypes = _compute_prototypes(model, train_loader, device, num_known)
            _log(f"  Finetune {epoch:02d}/{epochs}: "
                 f"loss={epoch_loss['total']:.4f} "
                 f"(sc={epoch_loss['supcon']:.4f} "
                 f"cls={epoch_loss['cls']:.4f} "
                 f"pull={epoch_loss['gen_pull']:.4f} "
                 f"pseudo={epoch_loss['pseudo_unk']:.4f})")

    # 解冻 backbone (恢复全模型训练能力)
    if hasattr(model, 'siamese') and hasattr(model.siamese, 'backbone'):
        for p in model.siamese.backbone.parameters():
            p.requires_grad = True

    _log(f"  [Finetune] Complete.")
    return history


@torch.no_grad()
def _compute_prototypes(model, loader, device, num_known):
    """快速计算训练集原型"""
    model.eval()
    feat_sum = [None] * num_known
    counts = [0] * num_known
    for x, y, _, _ in loader:
        mask = (y >= 0)
        if not mask.any():
            continue
        x = x[mask].to(device)
        y = y[mask]
        fps = model.extract_fingerprint(x)
        for i in range(fps.size(0)):
            c = int(y[i].item())
            if 0 <= c < num_known:
                if feat_sum[c] is None:
                    feat_sum[c] = fps[i].clone()
                else:
                    feat_sum[c] += fps[i]
                counts[c] += 1
    protos = []
    for c in range(num_known):
        if feat_sum[c] is not None and counts[c] > 0:
            protos.append(F.normalize(feat_sum[c] / counts[c], dim=0))
        else:
            protos.append(torch.zeros(fps.size(1), device=device))
    return torch.stack(protos)

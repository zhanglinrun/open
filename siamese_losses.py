"""
度量学习损失函数模块

包含:
  1. SupCon Loss (监督对比学习) - 核心度量损失, 产生极紧凑聚簇
  2. Center Loss (类内紧凑性)
  3. ARPL Loss (开放空间约束, 有界已知类区域)
  4. 对抗边界规整化损失

所有损失均针对孪生网络提取的 "切片指纹" 设计.
"""

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Supervised Contrastive Loss (SupCon)
# =============================================================================
class SupConLoss(nn.Module):
    """
    监督对比学习损失 (Khosla et al., 2020)

    核心优势: 每个 anchor 与 batch 内所有同类样本构成正对,
    利用率远超 Triplet Loss, 产生极紧凑的类簇.

    L = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a≠i} exp(z_i·z_a/τ) ]

    需要 BalancedBatchSampler 保证每类有多个样本在 batch 内.

    Args:
        temperature: 温度参数 (越小簇越紧)
                     0.07 → 极紧凑但易坍塌, 0.10~0.15 → 适度紧凑保留分布结构
    """

    def __init__(self, temperature: float = 0.12):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                feats: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            feats: 切片指纹 [B, D], L2 归一化
            labels: 标签 [B]
        Returns:
            (loss, stats)
        """
        feats_n = F.normalize(feats, dim=1)
        B = feats_n.size(0)
        device = feats_n.device

        # 相似度矩阵 / temperature
        sim = torch.matmul(feats_n, feats_n.T) / self.temperature  # [B, B]

        # 正对 mask: 同类且非自身
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [B, B]
        pos_mask.fill_diagonal_(0.0)

        # 排除自身的 mask
        logits_mask = 1.0 - torch.eye(B, device=device)

        # 数值稳定的 log-softmax
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()  # 减最大值防溢出

        exp_sim = torch.exp(sim) * logits_mask  # 排除自身
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # 对每个 anchor 的所有正对取平均
        n_pos = pos_mask.sum(dim=1)  # 每个 anchor 的正对数量
        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "n_pos_avg": 0.0, "mean_sim_pos": 0.0
            }

        mean_log_prob = (log_prob * pos_mask).sum(dim=1) / n_pos.clamp(min=1)
        loss = -mean_log_prob[valid].mean()

        # 统计
        with torch.no_grad():
            raw_sim = torch.matmul(feats_n, feats_n.T)
            pos_sim_avg = (raw_sim * pos_mask).sum() / pos_mask.sum().clamp(min=1)

        stats = {
            "n_pos_avg": float(n_pos[valid].float().mean().item()),
            "mean_sim_pos": float(pos_sim_avg.item()),
        }
        return loss, stats


# =============================================================================
# 2. Batch Hard Triplet Loss (在线 Hard Mining, 备用)
# =============================================================================
class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard 三元组损失: 在 batch 内在线挖掘最难的正/负样本

    对每个 anchor:
      - hardest positive: 同类中距离最远的
      - hardest negative: 异类中距离最近的

    Args:
        margin: margin 值
        distance: 距离类型
    """

    def __init__(self, margin: float = 0.3, distance: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def _pairwise_distances(self, feats: torch.Tensor) -> torch.Tensor:
        """计算 batch 内所有对的距离矩阵 [B, B]"""
        if self.distance == "cosine":
            sim = torch.matmul(feats, feats.T)  # [B, B]
            return 1.0 - sim
        else:
            # 欧式距离: ‖a-b‖² = ‖a‖² + ‖b‖² - 2<a,b>
            dot = torch.matmul(feats, feats.T)
            sq = torch.diag(dot)
            dist_sq = sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * dot
            return torch.sqrt(dist_sq.clamp(min=1e-12))

    def forward(self,
                feats: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            feats: 切片指纹 [B, D], L2 归一化
            labels: 标签 [B]
        Returns:
            (loss, stats) loss 标量, stats 包含挖掘统计
        """
        dist_mat = self._pairwise_distances(feats)  # [B, B]
        B = feats.size(0)

        # 创建标签匹配矩阵
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        label_neq = ~label_eq

        # Hardest positive: 同类中距离最大的
        # 将不同类距离设为 -inf
        dist_pos = dist_mat.clone()
        dist_pos[~label_eq] = -float("inf")
        dist_pos.fill_diagonal_(-float("inf"))  # 排除自身
        hardest_pos, _ = dist_pos.max(dim=1)  # [B]

        # Hardest negative: 异类中距离最小的
        # 将同类距离设为 +inf
        dist_neg = dist_mat.clone()
        dist_neg[~label_neq] = float("inf")
        hardest_neg, _ = dist_neg.min(dim=1)  # [B]

        # 过滤有效 triplet (至少有正负样本的)
        valid = (hardest_pos > -float("inf")) & (hardest_neg < float("inf"))
        if not valid.any():
            return torch.tensor(0.0, device=feats.device, requires_grad=True), {
                "n_valid": 0, "n_hard": 0, "mean_d_pos": 0.0, "mean_d_neg": 0.0
            }

        hp = hardest_pos[valid]
        hn = hardest_neg[valid]

        loss = F.relu(hp - hn + self.margin).mean()

        # 统计信息
        n_hard = int((hp - hn + self.margin > 0).sum().item())
        stats = {
            "n_valid": int(valid.sum().item()),
            "n_hard": n_hard,
            "mean_d_pos": float(hp.mean().item()),
            "mean_d_neg": float(hn.mean().item()),
        }

        return loss, stats


# =============================================================================
# 2. Center Loss (类内紧凑性)
# =============================================================================
class CenterLoss(nn.Module):
    """
    Center Loss: 最小化特征到对应类中心的距离

    L_center = 0.5 * Σ ‖f_i - c_{y_i}‖²

    中心 c 使用指数移动平均更新, 不直接通过梯度更新.

    Args:
        num_classes: 类别数
        feat_dim: 特征维度
        lr: 中心更新学习率
    """

    def __init__(self, num_classes: int, feat_dim: int, lr: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.center_lr = lr

        # 类中心 (不参与梯度)
        self.register_buffer("centers", torch.zeros(num_classes, feat_dim))
        self.centers_initialized = False

    def _update_centers(self, feats: torch.Tensor, labels: torch.Tensor):
        """使用当前 batch 更新类中心 (EMA)"""
        with torch.no_grad():
            for c in labels.unique():
                mask = labels == c
                c_feats = feats[mask]
                if c_feats.size(0) == 0:
                    continue

                new_center = c_feats.mean(dim=0)
                if not self.centers_initialized:
                    self.centers[c] = new_center
                else:
                    self.centers[c] = (1.0 - self.center_lr) * self.centers[c] \
                                      + self.center_lr * new_center

            if not self.centers_initialized:
                if labels.unique().numel() == self.num_classes:
                    self.centers_initialized = True

    def forward(self,
                feats: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: 切片指纹 [B, D]
            labels: 标签 [B]
        Returns:
            center loss 标量
        """
        # 更新中心
        self._update_centers(feats, labels)

        # 计算到对应中心的距离
        selected_centers = self.centers[labels]  # [B, D]
        loss = 0.5 * (feats - selected_centers).pow(2).sum(dim=1).mean()
        return loss


# =============================================================================
# 3. 特征分散正则化 (防止类内坍塌)
# =============================================================================
class SpreadRegularization(nn.Module):
    """
    特征分散正则化: 防止类内特征过度坍塌到单点

    当同类特征的方差低于最小阈值时施加惩罚,
    维持适度的类内扩展以保留分布边界信息.

    动机: SupCon + CosFace 组合会将同类特征驱动到 sim→1.0 (完全重合),
    导致决策边界极其锐利, 无法区分"正常边缘样本"和"未知类样本".
    本正则项保持适度的类内变异性, 为开集拒判提供缓冲区.

    Args:
        min_var: 最小方差阈值 (低于此值开始惩罚)
    """

    def __init__(self, min_var: float = 0.02):
        super().__init__()
        self.min_var = min_var

    def forward(self,
                feats: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: 切片指纹 [B, D], L2 归一化
            labels: 标签 [B]
        Returns:
            正则化损失标量
        """
        loss = torch.tensor(0.0, device=feats.device)
        count = 0
        for c in labels.unique():
            mask = labels == c
            if mask.sum() < 3:
                continue
            c_feats = feats[mask]
            # 每维方差的均值
            var = c_feats.var(dim=0).mean()
            # 当方差低于阈值时惩罚
            loss = loss + F.relu(self.min_var - var)
            count += 1
        return loss / max(count, 1)


# =============================================================================
# 4. 对抗边界规整化损失
# =============================================================================
class AdversarialBoundaryLoss(nn.Module):
    """
    对抗边界规整化损失 (解耦特征分布优化)

    设计思路:
      在特征空间中, 对类边界附近的样本施加额外约束,
      使得边界更清晰、更规整, 从而:
        - 减少已知类的误分类
        - 增强对未知类的拒判能力

    具体实现:
      1. 识别边界样本: 到最近异类原型距离 < 阈值的样本
      2. 对边界样本施加更强的类内聚合损失
      3. 对边界样本施加更强的类间分离损失

    Args:
        margin: 边界 margin
        boundary_radius: 边界区域半径 (余弦距离)
        amplify_factor: 边界样本损失放大系数
    """

    def __init__(self,
                 margin: float = 0.3,
                 boundary_radius: float = 0.5,
                 amplify_factor: float = 2.0):
        super().__init__()
        self.margin = margin
        self.boundary_radius = boundary_radius
        self.amplify_factor = amplify_factor

    def forward(self,
                feats: torch.Tensor,
                labels: torch.Tensor,
                prototypes: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            feats: 切片指纹 [B, D]
            labels: 标签 [B]
            prototypes: 类原型 [K, D]
        Returns:
            (loss, stats)
        """
        # 计算到所有原型的相似度
        sim = torch.matmul(feats, prototypes.T)  # [B, K]
        B, K = sim.shape

        # 自身类相似度
        pos_sim = sim[range(B), labels]  # [B]

        # 最近异类相似度
        neg_mask = torch.ones(B, K, dtype=torch.bool, device=feats.device)
        neg_mask[range(B), labels] = False
        neg_sim = sim.masked_fill(~neg_mask, -float("inf")).max(dim=1)[0]  # [B]

        # 识别边界样本: 到最近异类原型的余弦距离 < boundary_radius
        neg_cos_dist = 1.0 - neg_sim
        is_boundary = neg_cos_dist < self.boundary_radius  # [B]

        # 基础边界损失
        base_loss = F.relu(neg_sim - pos_sim + self.margin)  # [B]

        # 对边界样本放大损失
        weights = torch.ones(B, device=feats.device)
        if is_boundary.any():
            weights[is_boundary] = self.amplify_factor

        loss = (base_loss * weights).mean()

        stats = {
            "n_boundary": int(is_boundary.sum().item()),
            "mean_pos_sim": float(pos_sim.mean().item()),
            "mean_neg_sim": float(neg_sim.mean().item()),
        }

        return loss, stats




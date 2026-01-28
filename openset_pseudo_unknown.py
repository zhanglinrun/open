"""
伪未知类生成模块
包含凸组合生成、批量生成、对抗边界约束损失
"""

import random
from typing import Tuple

import torch
import torch.nn.functional as F


# -------------------------
# 凸组合生成伪未知特征
# -------------------------
def mixup_convex_combination(features: torch.Tensor,
                             labels: torch.Tensor,
                             num_classes: int,
                             min_mix: float = 0.2,
                             max_mix: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    已知类特征的凸组合,生成伪未知特征
    f_pseudo = λ * f_a + (1-λ) * f_b, 其中 a ≠ b

    Args:
        features: 已知类特征 [B, D]
        labels: 对应标签 [B]
        num_classes: 已知类数量
        min_mix: λ的最小值
        max_mix: λ的最大值

    Returns:
        pseudo_features: 伪未知特征 [B, D]
        pseudo_labels: 伪标签(设为-1,表示未知) [B]
    """
    batch_size = features.size(0)
    device = features.device

    # 1. 随机打乱样本,生成配对
    indices = torch.randperm(batch_size, device=device)
    mixed_labels = labels[indices]

    # 2. 随机生成混合系数λ
    lam = torch.rand(batch_size, device=device) * (max_mix - min_mix) + min_mix

    # 3. 凸组合: f_pseudo = λ * f_a + (1-λ) * f_b
    # 确保a和b属于不同类别
    mask_diff = (labels != mixed_labels)
    if mask_diff.sum() < batch_size // 2:
        # 如果不同类对太少,重新随机
        indices = torch.randperm(batch_size, device=device)
        mixed_labels = labels[indices]
        lam = torch.rand(batch_size, device=device) * (max_mix - min_mix) + min_mix

    # 执行凸组合
    pseudo_features = lam.unsqueeze(1) * features + (1 - lam).unsqueeze(1) * features[indices]

    # 4. 归一化
    pseudo_features = F.normalize(pseudo_features, dim=1)

    # 5. 伪标签设为-1(表示未知类)
    pseudo_labels = torch.full((batch_size,), -1, dtype=torch.long, device=device)

    return pseudo_features, pseudo_labels


def generate_pseudo_unknown_batch(features: torch.Tensor,
                                labels: torch.Tensor,
                                num_classes: int,
                                ratio: float = 0.3,
                                min_mix: float = 0.2,
                                max_mix: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量生成伪未知特征
    按给定比例从已知类特征中采样并组合

    Args:
        features: 已知类特征 [B, D]
        labels: 对应标签 [B]
        num_classes: 已知类数量
        ratio: 伪未知特征占原特征的比例
        min_mix: λ的最小值
        max_mix: λ的最大值

    Returns:
        all_features: 包含已知类和伪未知的特征 [B*(1+ratio), D]
        all_labels: 对应标签 [B*(1+ratio)]
    """
    batch_size = features.size(0)
    pseudo_size = int(batch_size * ratio)

    # 1. 从已知类特征中采样用于生成伪未知
    if pseudo_size > 0:
        pseudo_indices = torch.randperm(batch_size, device=features.device)[:pseudo_size]
        pseudo_input_feats = features[pseudo_indices]
        pseudo_input_labels = labels[pseudo_indices]

        # 2. 生成伪未知特征
        pseudo_feats, pseudo_labels = mixup_convex_combination(
            pseudo_input_feats,
            pseudo_input_labels,
            num_classes,
            min_mix,
            max_mix
        )
    else:
        pseudo_feats = torch.empty(0, features.size(1), device=features.device)
        pseudo_labels = torch.empty(0, dtype=torch.long, device=features.device)

    # 3. 合并已知类和伪未知特征
    all_features = torch.cat([features, pseudo_feats], dim=0)
    all_labels = torch.cat([labels, pseudo_labels], dim=0)

    return all_features, all_labels


# -------------------------
# 基于原型距离的伪未知分类损失
# -------------------------
def pseudo_unknown_classification_loss(pseudo_features: torch.Tensor,
                                     prototypes: torch.Tensor,
                                     margin: float = 0.5) -> torch.Tensor:
    """
    让伪未知特征远离所有已知类原型的损失
    目标: max(sim(f_pseudo, proto_k)) < margin

    Args:
        pseudo_features: 伪未知特征 [B, D] (已归一化)
        prototypes: 已知类原型 [K, D] (已归一化)
        margin: 最大允许相似度

    Returns:
        损失值(标量)
    """
    # 计算伪未知特征到所有原型的相似度
    sim_matrix = torch.matmul(pseudo_features, prototypes.T)  # [B, K]
    max_sim = sim_matrix.max(dim=1)[0]  # [B]

    # 损失: 如果max_sim > margin,则产生惩罚
    loss = torch.clamp(max_sim - margin, min=0.0).pow(2).mean()

    return loss


# -------------------------
# 生成边界原型(用于开放空间约束)
# -------------------------
def generate_boundary_prototypes(prototypes: torch.Tensor,
                                margin: float = 0.5) -> torch.Tensor:
    """
    在类间边界生成边界原型
    用于开放空间约束,让模型学会拒判边界样本

    Args:
        prototypes: 已知类原型 [K, D]
        margin: 距离原型的比例

    Returns:
        边界原型 [K*(K-1)/2, D]
    """
    K = prototypes.size(0)
    boundary_protos = []

    # 在每对原型之间生成边界点
    for i in range(K):
        for j in range(i + 1, K):
            # 线性插值: proto_boundary = (proto_i + proto_j) / 2
            # 稍微往中间移动
            boundary = (prototypes[i] + prototypes[j]) / 2.0
            boundary = F.normalize(boundary, dim=0)
            boundary_protos.append(boundary)

    if len(boundary_protos) > 0:
        return torch.stack(boundary_protos, dim=0)
    else:
        return torch.empty(0, prototypes.size(1), device=prototypes.device)


# -------------------------
# 开放空间训练样本生成
# -------------------------
class PseudoUnknownGenerator:
    """
    伪未知类生成器
    支持多种生成策略
    """

    def __init__(self,
                 strategy: str = "mixup",
                 ratio: float = 0.3,
                 min_mix: float = 0.2,
                 max_mix: float = 0.8):
        """
        Args:
            strategy: 生成策略 ("mixup", "boundary")
            ratio: 伪未知比例
            min_mix/max_mix: mixup参数范围
        """
        self.strategy = strategy
        self.ratio = ratio
        self.min_mix = min_mix
        self.max_mix = max_mix

    def __call__(self,
                 features: torch.Tensor,
                 labels: torch.Tensor,
                 num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成伪未知特征

        Args:
            features: 已知类特征 [B, D]
            labels: 对应标签 [B]
            num_classes: 已知类数量

        Returns:
            all_features: 包含已知类和伪未知的特征
            all_labels: 对应标签(未知类标记为-1)
        """
        if self.strategy == "mixup":
            return generate_pseudo_unknown_batch(
                features, labels, num_classes,
                self.ratio, self.min_mix, self.max_mix
            )
        elif self.strategy == "boundary":
            # 基于边界原型的生成(较复杂,暂不实现)
            return features, labels
        else:
            return features, labels


# -------------------------
# 测试代码
# -------------------------
if __name__ == "__main__":
    # 测试伪未知生成
    print("伪未知类生成模块测试")

    batch_size = 8
    num_classes = 7
    embed_dim = 256

    features = F.normalize(torch.randn(batch_size, embed_dim), dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 测试凸组合
    pseudo_feats, pseudo_labels = mixup_convex_combination(
        features, labels, num_classes
    )
    print(f"\n凸组合生成:")
    print(f"  原始特征: {features.shape}")
    print(f"  伪未知特征: {pseudo_feats.shape}")
    print(f"  伪未知标签: {pseudo_labels.unique()} (应该是[-1])")

    # 测试批量生成
    all_feats, all_labels = generate_pseudo_unknown_batch(
        features, labels, num_classes, ratio=0.3
    )
    print(f"\n批量生成:")
    print(f"  合并特征: {all_feats.shape}")
    print(f"  合并标签: {all_labels.shape}")
    print(f"  未知类数量: {(all_labels == -1).sum().item()}")

    # 测试边界原型
    protos = F.normalize(torch.randn(num_classes, embed_dim), dim=1)
    boundary_protos = generate_boundary_prototypes(protos)
    print(f"\n边界原型生成:")
    print(f"  原型数量: {protos.shape[0]}")
    print(f"  边界原型数量: {boundary_protos.shape[0]}")

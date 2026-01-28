"""
损失函数模块
包含带margin的交叉熵、对抗性margin损失、边界约束损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 基础损失函数
# -------------------------
def cross_entropy_with_margin(logits: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
    """
    标准交叉熵损失(假设logits已经过margin处理)

    Args:
        logits: 分类logits [B, C]
        labels: 标签 [B]

    Returns:
        损失值(标量)
    """
    return F.cross_entropy(logits, labels)


# -------------------------
# 对抗性Margin损失
# -------------------------
def adversarial_margin_loss(logit_neg: torch.Tensor,
                          logit_pos: torch.Tensor,
                          labels: torch.Tensor,
                          lam: float = 1.0) -> torch.Tensor:
    """
    对抗性margin损失(论文公式2)
    L_Adv = L_T + λ * L_D

    Args:
        logit_neg: 负margin分类器的logits (transferable) [B, C]
        logit_pos: 正margin分类器的logits (discriminative) [B, C]
        labels: 标签 [B]
        lam: L_T和L_D的权重比

    Returns:
        总损失(标量)
    """
    # L_T: Transferable loss (负margin,降低难度)
    loss_T = F.cross_entropy(logit_neg, labels)

    # L_D: Discriminative loss (正margin,增加难度)
    loss_D = F.cross_entropy(logit_pos, labels)

    # 对抗性损失
    loss_adv = loss_T + lam * loss_D

    return loss_adv


# -------------------------
# 边界约束损失
# -------------------------
def intra_class_aggregation_loss(features: torch.Tensor,
                                prototypes: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
    """
    类内聚合损失
    让同类样本向原型靠拢

    Args:
        features: 特征 [B, D] (已归一化)
        prototypes: 类原型 [K, D] (已归一化)
        labels: 标签 [B]

    Returns:
        损失值(标量)
    """
    # 计算每个样本到其所属类原型的余弦相似度
    selected_protos = prototypes[labels]  # [B, D]
    sim = torch.sum(features * selected_protos, dim=1)  # [B]

    # 损失: 1 - similarity (越小越好)
    loss = (1.0 - sim).mean()

    return loss


def inter_class_separation_loss(prototypes: torch.Tensor,
                               margin: float = 0.1) -> torch.Tensor:
    """
    类间分离损失
    让不同类原型远离

    Args:
        prototypes: 类原型 [K, D] (已归一化)
        margin: 最小余弦相似度阈值

    Returns:
        损失值(标量)
    """
    # 计算原型两两之间的余弦相似度
    sim_matrix = torch.matmul(prototypes, prototypes.T)  # [K, K]

    # 对角线置0(类自己和自己)
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    off_diag_sim = sim_matrix[mask]

    # 损失: 如果相似度 > margin,则产生惩罚
    # 目标是让off_diag_sim都 < margin
    loss = torch.clamp(off_diag_sim - margin, min=0.0).pow(2).mean()

    return loss


def open_space_constraint_loss(features: torch.Tensor,
                             prototypes: torch.Tensor,
                             labels: torch.Tensor,
                             margin: float = 0.5) -> torch.Tensor:
    """
    开放空间约束损失
    让已知类样本远离开放空间(未知区域)

    思路:让样本靠近自己的原型,远离其他原型

    Args:
        features: 特征 [B, D] (已归一化)
        prototypes: 类原型 [K, D] (已归一化)
        labels: 标签 [B]
        margin: 最小距离阈值

    Returns:
        损失值(标量)
    """
    # 计算到所有原型的相似度
    sim_matrix = torch.matmul(features, prototypes.T)  # [B, K]

    # 真实类别的相似度
    pos_sim = sim_matrix[range(features.size(0)), labels]  # [B]

    # 最大非真实类别的相似度
    mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    mask[range(features.size(0)), labels] = False
    neg_sim = sim_matrix[mask].view(features.size(0), -1).max(dim=1)[0]  # [B]

    # 损失: 让正相似度 > 负相似度 + margin
    # 即: pos_sim - neg_sim > margin
    loss = torch.clamp(margin - (pos_sim - neg_sim), min=0.0).pow(2).mean()

    return loss


def total_boundary_loss(features: torch.Tensor,
                       prototypes: torch.Tensor,
                       labels: torch.Tensor,
                       intra_weight: float = 1.0,
                       inter_weight: float = 1.0,
                       open_weight: float = 0.5) -> torch.Tensor:
    """
    总边界约束损失

    Args:
        features: 特征 [B, D] (已归一化)
        prototypes: 类原型 [K, D] (已归一化)
        labels: 标签 [B]
        intra_weight: 类内聚合损失权重
        inter_weight: 类间分离损失权重
        open_weight: 开放空间约束权重

    Returns:
        总损失(标量)
    """
    loss_intra = intra_class_aggregation_loss(features, prototypes, labels)
    loss_inter = inter_class_separation_loss(prototypes)
    loss_open = open_space_constraint_loss(features, prototypes, labels)

    total_loss = (intra_weight * loss_intra +
                 inter_weight * loss_inter +
                 open_weight * loss_open)

    return total_loss


# -------------------------
# EDL不确定性损失(可选扩展)
# -------------------------
def kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Dirichlet分布的KL散度(用于EDL校准)

    Args:
        alpha: Dirichlet参数 [B, K]
        num_classes: 类别数K

    Returns:
        KL散度 [B]
    """
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

    term1 = torch.lgamma(sum_alpha) - torch.lgamma(torch.tensor([num_classes], dtype=torch.float32, device=alpha.device))
    term2 = torch.sum(torch.lgamma(alpha) - torch.lgamma(ones), dim=1)
    term3 = torch.sum((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=1)

    kl = term1 + term2 + term3
    return kl


def edl_mse_loss(logits: torch.Tensor,
                 labels: torch.Tensor,
                 num_classes: int,
                 annealing_step: int = 10) -> torch.Tensor:
    """
    EDL MSE损失函数

    Args:
        logits: 模型输出(证据) [B, K], 必须非负
        labels: 标签 [B]
        num_classes: 类别数K
        annealing_step: KL散度退火步数

    Returns:
        损失值(标量)
    """
    # 确保logits非负(evidence)
    evidence = F.relu(logits)

    # Dirichlet参数
    alpha = evidence + 1

    # 转换为one-hot标签
    y = F.one_hot(labels, num_classes=num_classes).float()

    # 预测概率
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S

    # MSE损失
    mse = torch.sum((prob - y) ** 2, dim=1).mean()

    # KL散度校准项
    kl = kl_divergence(alpha, num_classes).mean()
    kl = torch.min(kl, torch.tensor(1.0, device=kl.device)) / annealing_step

    # 总损失
    loss = mse + kl

    return loss


def compute_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """
    计算不确定性u (用于EDL)

    Args:
        logits: 模型输出 [B, K]

    Returns:
        不确定性u [B], 值越大表示越不确定
    """
    evidence = F.relu(logits)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1)

    # u = K / S
    uncertainty = logits.size(1) / S

    return uncertainty


# -------------------------
# 测试代码
# -------------------------
if __name__ == "__main__":
    # 测试损失函数
    batch_size = 4
    num_classes = 7
    embed_dim = 256

    logit_neg = torch.randn(batch_size, num_classes)
    logit_pos = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    features = F.normalize(torch.randn(batch_size, embed_dim), dim=1)
    prototypes = F.normalize(torch.randn(num_classes, embed_dim), dim=1)

    # 测试对抗性损失
    loss_adv = adversarial_margin_loss(logit_neg, logit_pos, labels, lam=1.0)
    print(f"Adversarial Margin Loss: {loss_adv.item():.4f}")

    # 测试边界损失
    loss_boundary = total_boundary_loss(features, prototypes, labels)
    print(f"Total Boundary Loss: {loss_boundary.item():.4f}")

    # 测试EDL损失
    logits_edl = torch.randn(batch_size, num_classes)
    loss_edl = edl_mse_loss(logits_edl, labels, num_classes)
    print(f"EDL MSE Loss: {loss_edl.item():.4f}")

    uncertainty = compute_uncertainty(logits_edl)
    print(f"Uncertainty: {uncertainty}")

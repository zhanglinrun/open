"""
模型定义模块
包含辅助函数、AddMarginProduct分类器、ResNet18特征提取器、双头网络
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# -------------------------
# 辅助函数
# -------------------------
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    将标签转换为one-hot编码

    Args:
        labels: 标签张量 [B]
        num_classes: 类别数

    Returns:
        one-hot编码 [B, num_classes]
    """
    return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.bool)


# -------------------------
# AddMarginProduct分类器
# -------------------------
class AddMarginProduct(nn.Module):
    """
    大margin cosine距离分类器
    支持正负margin,用于开集识别

    Args:
        in_features: 输入特征维度
        out_features: 输出类别数
        scale_factor: 温度参数
        margin: margin值
            - 正margin (+mpos): 用于discriminative features,增加分类难度
            - 负margin (-mneg): 用于transferable features,降低分类难度
    """

    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.40):
        super(AddMarginProduct, self).__init__()
        self.scale_factor = float(scale_factor)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature: torch.Tensor, label: torch.Tensor = None):
        """
        前向传播

        Args:
            feature: 输入特征 [B, D], 需要L2归一化
            label: 标签 [B], 训练时提供

        Returns:
            logits: 分类logits [B, C], 已经过scale
        """
        # 计算cosine相似度
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))

        # 测试模式: 直接返回
        if label is None:
            return cosine * self.scale_factor

        # 训练模式: 应用margin
        # 只对真实类别减去margin
        phi = cosine - self.margin
        mask = one_hot(label, cosine.shape[1])
        out = torch.where(mask, phi, cosine) * self.scale_factor

        return out


# -------------------------
# ResNet2D特征提取器
# -------------------------
class ResNet2D_Embed(nn.Module):
    """
    基于ResNet18的2D图像特征提取器
    输出两路特征:
        - x_neg: Transferable features (配合负margin)
        - x_pos: Discriminative features (配合正margin)

    Args:
        embed_dim: 嵌入维度
        pretrained: 是否使用预训练权重
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super(ResNet2D_Embed, self).__init__()

        # 使用预训练的ResNet18
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # 替换最后一层为恒等映射
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 分支1: Transferable Features (负margin分类器)
        self.fc_neg = nn.Linear(in_dim, embed_dim)
        self.bn_neg = nn.BatchNorm1d(embed_dim)

        # 分支2: Discriminative Features (正margin分类器)
        # 对应论文中的变换 g(·)
        self.fc_pos = nn.Linear(embed_dim, embed_dim)
        self.bn_pos = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            x_neg: Transferable特征 [B, embed_dim]
            x_pos: Discriminative特征 [B, embed_dim]
        """
        h = self.backbone(x)  # [B, in_dim]

        # Neg分支: 提取通用特征
        x_neg = self.bn_neg(self.fc_neg(h))
        x_neg = F.relu(x_neg, inplace=True)

        # Pos分支: 提取判别特征
        x_pos = self.bn_pos(self.fc_pos(x_neg))
        x_pos = F.relu(x_pos, inplace=True)

        return x_neg, x_pos


# -------------------------
# 双头网络
# -------------------------
class NegPosNet2D(nn.Module):
    """
    基于正负margin的双头开集识别网络

    Args:
        num_known_classes: 已知类数量
        embed_dim: 嵌入维度
        mneg: 负margin值
        mpos: 正margin值
        scale_factor: 温度参数
        pretrained: 是否使用预训练
    """

    def __init__(self,
                 num_known_classes: int,
                 embed_dim: int = 256,
                 mneg: float = 0.20,
                 mpos: float = 0.40,
                 scale_factor: float = 30.0,
                 pretrained: bool = True):
        super(NegPosNet2D, self).__init__()

        # 特征提取器
        self.embed = ResNet2D_Embed(embed_dim=embed_dim, pretrained=pretrained)

        # 两个分类器: 一个用负margin,一个用正margin
        self.cls_neg = AddMarginProduct(
            embed_dim,
            num_known_classes,
            scale_factor=scale_factor,
            margin=-mneg  # 负margin
        )

        self.cls_pos = AddMarginProduct(
            embed_dim,
            num_known_classes,
            scale_factor=scale_factor,
            margin=+mpos  # 正margin
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]
            y: 标签 [B], 训练时提供

        Returns:
            logit_neg: 负margin分类器的logits
            logit_pos: 正margin分类器的logits
            feat_neg: Transferable特征
            feat_pos: Discriminative特征
        """
        feat_neg, feat_pos = self.embed(x)

        if y is None:
            # 测试模式
            logit_neg = self.cls_neg(feat_neg, None)
            logit_pos = self.cls_pos(feat_pos, None)
        else:
            # 训练模式
            logit_neg = self.cls_neg(feat_neg, y)
            logit_pos = self.cls_pos(feat_pos, y)

        return logit_neg, logit_pos, feat_neg, feat_pos


# -------------------------
# 测试代码
# -------------------------
if __name__ == "__main__":
    # 测试模型
    model = NegPosNet2D(num_known_classes=7, embed_dim=256)
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 7, (4,))

    # 训练模式
    logit_neg, logit_pos, f_neg, f_pos = model(x, y)
    print(f"训练模式:")
    print(f"  logit_neg shape: {logit_neg.shape}")
    print(f"  logit_pos shape: {logit_pos.shape}")
    print(f"  feat_neg shape: {f_neg.shape}")
    print(f"  feat_pos shape: {f_pos.shape}")

    # 测试模式
    logit_neg, logit_pos, f_neg, f_pos = model(x, None)
    print(f"\n测试模式:")
    print(f"  logit_neg shape: {logit_neg.shape}")
    print(f"  logit_pos shape: {logit_pos.shape}")

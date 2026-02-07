"""
孪生网络模型 - 切片指纹提取器

核心设计:
  1. 共享骨干网络 (ResNet18) 提取基础特征
  2. 嵌入头将特征映射到切片指纹空间
  3. 可选的投影头用于对比学习训练
  4. 支持 pair-wise 和 triplet 模式

切片指纹 (Slice Fingerprint):
  - 由孪生网络对目标切片提取的固定维度特征向量
  - 同类目标的指纹在特征空间中聚集
  - 不同类目标的指纹在特征空间中分离
  - 用于后续的特征库比对和有偏/无偏判定
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# =============================================================================
# 嵌入头 (Embedding Head)
# =============================================================================
class EmbeddingHead(nn.Module):
    """
    嵌入头: 将骨干网络输出映射到切片指纹空间

    Args:
        in_dim: 输入维度 (骨干网络输出)
        embed_dim: 切片指纹维度
        use_bn: 是否使用 BatchNorm
    """

    def __init__(self, in_dim: int, embed_dim: int = 128, use_bn: bool = True,
                 ):
        super().__init__()
        self.fc = nn.Linear(in_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.fc(x))
        h = F.relu(h, inplace=True)
        return F.normalize(h, p=2, dim=1)


# =============================================================================
# 投影头 (Projection Head) - 用于对比学习
# =============================================================================
class ProjectionHead(nn.Module):
    """
    投影头: 将切片指纹进一步映射到对比学习空间
    (仅训练时使用, 推理时丢弃)

    Args:
        in_dim: 输入维度 (切片指纹维度)
        hidden_dim: 隐藏层维度
        out_dim: 输出维度
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 切片指纹 [B, in_dim]
        Returns:
            投影特征 [B, out_dim], L2 归一化
        """
        return F.normalize(self.net(x), p=2, dim=1)


# =============================================================================
# 孪生网络主体
# =============================================================================
class SiameseFingerprint(nn.Module):
    """
    孪生网络切片指纹提取器

    流程:
      input image → backbone → embedding head → slice fingerprint
                                               ↘ projection head (训练)

    Args:
        backbone: 骨干网络名称 ("resnet18")
        embed_dim: 切片指纹维度
        pretrained: 是否使用预训练
        projection_head: 是否使用投影头
        proj_hidden_dim: 投影头隐藏层维度
        proj_out_dim: 投影头输出维度
    """

    def __init__(self,
                 backbone: str = "resnet18",
                 embed_dim: int = 128,
                 pretrained: bool = True,
                 projection_head: bool = True,
                 proj_hidden_dim: int = 256,
                 proj_out_dim: int = 64):
        super().__init__()

        # 1. 骨干网络
        if backbone == "resnet18":
            base = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            in_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == "resnet34":
            base = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            in_dim = base.fc.in_features
            base.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = base
        self.backbone_dim = in_dim

        # 2. 嵌入头 → 切片指纹
        self.embed_head = EmbeddingHead(in_dim, embed_dim)
        self.embed_dim = embed_dim

        # 3. 投影头 (可选, 仅训练)
        self.proj_head = None
        if projection_head:
            self.proj_head = ProjectionHead(embed_dim, proj_hidden_dim, proj_out_dim)

    def extract_fingerprint(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取切片指纹 (推理用)

        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            切片指纹 [B, embed_dim], L2 归一化
        """
        h = self.backbone(x)
        fingerprint = self.embed_head(h)
        return fingerprint

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            dict 包含:
              - "fingerprint": 切片指纹 [B, embed_dim]
              - "projection": 投影特征 [B, proj_out_dim] (仅训练)
              - "backbone_feat": 骨干网络原始特征 [B, backbone_dim]
        """
        h = self.backbone(x)
        fingerprint = self.embed_head(h)

        result = {
            "fingerprint": fingerprint,
            "backbone_feat": h,
        }

        if self.proj_head is not None:
            result["projection"] = self.proj_head(fingerprint)

        return result


# =============================================================================
# CosFace 对抗边界分类器 (AddMarginProduct / LMCL)
# =============================================================================
class AddMarginProduct(nn.Module):
    """
    Large Margin Cosine Loss (CosFace / LMCL) 分类器

    训练时: logits = s * (cos(theta) - m)  对正确类减去 margin
    推理时: logits = s * cos(theta)        无 margin, 正常分类

    margin 的存在迫使模型学到更紧凑的类边界, 构成对抗边界.

    Args:
        in_features: 输入特征维度 (切片指纹维度)
        out_features: 输出类别数 (已知类数量)
        scale_factor: 缩放因子 s
        margin: 余弦 margin m
    """

    def __init__(self, in_features: int, out_features: int,
                 scale_factor: float = 30.0, margin: float = 0.40):
        super().__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature: torch.Tensor,
                label: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feature: 切片指纹 [B, D], L2 归一化
            label: 标签 [B] (训练时提供, 推理时为 None)
        Returns:
            logits [B, K]
        """
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        if label is None:
            return cosine * self.scale_factor
        phi = cosine - self.margin
        one_hot_label = F.one_hot(label, cosine.size(1)).bool()
        output = torch.where(one_hot_label, phi, cosine)
        return output * self.scale_factor


# =============================================================================
# 带在线分类能力的孪生网络 (用于训练阶段的辅助分类)
# =============================================================================
class SiameseFingerprintWithClassifier(nn.Module):
    """
    带辅助分类器的孪生网络

    在度量学习的基础上增加分类头, 用于:
      1. 训练初期加速收敛
      2. 监督信号增强
      3. 有偏/无偏判定时辅助评估

    Args:
        num_known_classes: 已知类数量
        backbone: 骨干网络
        embed_dim: 切片指纹维度
        pretrained: 是否使用预训练
    """

    def __init__(self,
                 num_known_classes: int,
                 backbone: str = "resnet18",
                 embed_dim: int = 128,
                 pretrained: bool = True,
                 projection_head: bool = True,
                 proj_hidden_dim: int = 256,
                 proj_out_dim: int = 64):
        super().__init__()

        # 孪生网络主体
        self.siamese = SiameseFingerprint(
            backbone=backbone,
            embed_dim=embed_dim,
            pretrained=pretrained,
            projection_head=projection_head,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
        )

        # CosFace 对抗边界分类器
        self.classifier = AddMarginProduct(embed_dim, num_known_classes,
                                           scale_factor=30.0, margin=0.40)

        self.num_known_classes = num_known_classes
        self.embed_dim = embed_dim

    def extract_fingerprint(self, x: torch.Tensor) -> torch.Tensor:
        """提取切片指纹 (推理用)"""
        return self.siamese.extract_fingerprint(x)

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, C, H, W]
            labels: 标签 [B] (训练时提供启用 CosFace margin, 推理时为 None)
        Returns:
            dict 包含 fingerprint, projection, logits
        """
        out = self.siamese(x)
        fingerprint = out["fingerprint"]

        # CosFace: 训练时减 margin 构成对抗边界, 推理时正常余弦
        out["logits"] = self.classifier(fingerprint, labels)
        return out


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("孪生网络模型测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 基础孪生网络
    model = SiameseFingerprint(
        backbone="resnet18",
        embed_dim=128,
        pretrained=False,
        projection_head=True,
    ).to(device)

    x = torch.randn(4, 3, 128, 128, device=device)
    out = model(x)
    print(f"\n[基础孪生网络]")
    print(f"  fingerprint shape: {out['fingerprint'].shape}")
    print(f"  projection shape:  {out['projection'].shape}")
    print(f"  backbone_feat shape: {out['backbone_feat'].shape}")

    # 指纹范数检查 (应该为 1.0)
    norms = out["fingerprint"].norm(dim=1)
    print(f"  fingerprint norms: {norms.tolist()} (should be ~1.0)")

    # 2. 带分类器
    model_cls = SiameseFingerprintWithClassifier(
        num_known_classes=7,
        embed_dim=128,
        pretrained=False,
    ).to(device)

    out_cls = model_cls(x)
    print(f"\n[带分类器孪生网络]")
    print(f"  fingerprint shape: {out_cls['fingerprint'].shape}")
    print(f"  logits shape:      {out_cls['logits'].shape}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[参数量] {total_params:,}")

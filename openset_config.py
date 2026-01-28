"""
开集识别配置文件
包含数据集常量、超参数配置、训练参数默认值
"""

from typing import Dict

# -------------------------
# 数据集常量定义
# -------------------------
ID2NAME = {
    0: "Aircraft-Carrier",
    1: "Warship",
    2: "Bulk-Carrier",
    3: "Oil-Tanker",
    4: "Container-Ship",
    5: "Cargo-Ship",
    6: "Passenger-Cruise-Ship",
    7: "Tug",
    8: "Vehicles-Carrier",
    9: "Blurred",
}

NAME2ID = {v: k for k, v in ID2NAME.items()}

# 已知类和未知类划分
# 满足开题报告要求:库内≥7类,库外≥3类
# UNKNOWN_ORIG_IDS = {0, 6, 9}  # 3个未知类
# UNKNOWN_ORIG_IDS = {2, 6, 7}
UNKNOWN_ORIG_IDS = {6, 8, 9}
KNOWN_ORIG_IDS = [i for i in range(10) if i not in UNKNOWN_ORIG_IDS]  # 7个已知类: 1,2,3,4,5,7,8

# 图片文件扩展名
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# -------------------------
# 超参数配置
# -------------------------
class ModelConfig:
    """模型超参数"""
    embed_dim: int = 256
    mneg: float = 0.20  # 负margin (transferable classifier)
    mpos: float = 0.40  # 正margin (discriminative classifier)
    scale_factor: float = 30.0  # softmax温度参数
    pretrained: bool = True


class TrainingConfig:
    """训练超参数"""
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lam: float = 1.0  # L_T和L_D的权重比
    seed: int = 42
    num_workers: int = 4
    grad_clip: float = 0.0
    train_test_ratio: float = 0.5


class PseudoUnknownConfig:
    """伪未知类生成配置"""
    ratio: float = 0.3  # 伪未知类占训练集的比例
    min_mix: float = 0.2  # mixup λ的最小值
    max_mix: float = 0.8  # mixup λ的最大值
    use_boundary_loss: bool = True  # 是否使用边界约束损失
    intra_weight: float = 1.0  # 类内聚合损失权重
    inter_weight: float = 1.0  # 类间分离损失权重
    open_weight: float = 0.5  # 开放空间约束权重


class ThresholdConfig:
    """阈值配置"""
    percentile: float = 95.0  # 阈值分位数(用于训练集)
    search_steps: int = 201  # tau搜索步数
    use_class_tau: bool = True  # 是否使用按类阈值


class AugmentConfig:
    """数据增强配置"""
    image_size: int = 224
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05

    # 归一化参数(ImageNet)
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]


# 默认配置实例
model_cfg = ModelConfig()
training_cfg = TrainingConfig()
pseudo_unknown_cfg = PseudoUnknownConfig()
threshold_cfg = ThresholdConfig()
augment_cfg = AugmentConfig()

"""
孪生网络开集识别 - 配置文件
适用于 MSTAR 数据集的切片指纹提取与开集识别

核心概念:
  - 切片指纹 (Slice Fingerprint): 孪生网络共享骨干提取的深度特征向量,
    唯一标识一个目标切片的身份信息.
  - 特征库 (Feature Library): 存储每个已知类的原型指纹及统计信息,
    用于分类和有偏/无偏判定.
"""

from typing import Set, List

# =============================================================================
# MSTAR 数据集常量
# =============================================================================
ID2NAME = {
    0: "2S1",
    1: "BRDM_2",
    2: "BTR_60",
    3: "D7",
    4: "SN_132",
    5: "SN_9563",
    6: "SN_C71",
    7: "T62",
    8: "ZIL131",
    9: "ZSU_23_4",
}

NAME2ID = {v: k for k, v in ID2NAME.items()}

# 已知类 / 未知类划分 (默认: 7 known + 3 unknown)
# 未知类: SN_9563(自行火炮), ZIL131(卡车), ZSU_23_4(自行高炮)
# 已知类: 2S1, BRDM_2, BTR_60, D7, SN_132, SN_C71, T62
UNKNOWN_ORIG_IDS: Set[int] = {5, 8, 9}  # SN_9563, ZIL131, ZSU_23_4
KNOWN_ORIG_IDS: List[int] = [i for i in range(10) if i not in UNKNOWN_ORIG_IDS]

# 图片文件扩展名
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# 孪生网络模型配置
# =============================================================================
class SiameseModelConfig:
    """孪生网络模型超参数"""
    backbone: str = "resnet18"       # 骨干网络
    embed_dim: int = 128             # 切片指纹维度 (嵌入维度)
    pretrained: bool = True          # 是否使用 ImageNet 预训练
    projection_head: bool = True     # 是否使用投影头 (训练时用)
    proj_hidden_dim: int = 256       # 投影头隐藏层维度
    proj_out_dim: int = 64          # 投影头输出维度 (对比学习空间)


# =============================================================================
# 训练配置
# =============================================================================
class SiameseTrainingConfig:
    """训练超参数"""
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 4
    grad_clip: float = 1.0
    train_ratio: float = 0.8        # 训练集比例

    # 学习率调度
    scheduler: str = "cosine"        # "cosine" | "step"
    step_size: int = 30
    gamma: float = 0.1

    # 损失函数权重
    contrastive_weight: float = 1.0  # 对比损失权重
    center_weight: float = 0.01      # 中心损失权重
    triplet_weight: float = 1.0      # 三元组损失权重
    boundary_weight: float = 0.5     # 边界约束损失权重

    # 度量学习参数
    margin: float = 0.3              # Batch Hard Triplet Loss 的 margin


# =============================================================================
# 有偏 / 无偏判定配置 (4.1)
# =============================================================================
class BiasJudgeConfig:
    """
    库内有偏/无偏判定标准配置

    判定逻辑:
      对于库内某已知类 c 的一个新样本 x,提取其切片指纹 f_x,
      与该类原型 μ_c 比较:
        1. 余弦距离 d_cos = 1 - cos(f_x, μ_c)
        2. 马氏距离 d_maha = sqrt((f_x - μ_c)^T Σ_c^{-1} (f_x - μ_c))
        3. 特征范数偏差 |‖f_x‖ - mean_norm_c| / std_norm_c
      综合评分超过阈值则判定为有偏.
    """
    # 各判定准则权重
    cosine_weight: float = 0.4        # 余弦距离权重
    mahalanobis_weight: float = 0.4   # 马氏距离权重
    norm_weight: float = 0.2          # 范数偏差权重

    # 阈值参数
    percentile: float = 95.0          # 用训练集第 p 分位数定阈值
    cosine_tau_default: float = 0.3   # 余弦距离默认阈值
    maha_tau_default: float = 3.0     # 马氏距离默认阈值 (≈ χ² 分布)
    norm_tau_default: float = 2.0     # 范数偏差默认阈值 (几个标准差)

    # 综合评分阈值
    combined_threshold: float = 0.5   # 综合有偏评分阈值 (0~1)

    # 分布拟合
    use_per_class_threshold: bool = True  # 是否使用按类阈值


# =============================================================================
# 特征分布优化配置 (4.2)
# =============================================================================
class FeatOptimConfig:
    """
    深度特征分布优化配置

    针对无偏样本的切片指纹进行:
      - 类内紧凑性优化 (center loss)
      - 类间分离度优化 (triplet loss)
      - 对抗边界规整化 (adversarial boundary regularization)
    """
    # 优化目标
    intra_compact_weight: float = 1.0   # 类内紧凑权重
    inter_separate_weight: float = 1.0  # 类间分离权重
    boundary_regular_weight: float = 0.5 # 边界规整化权重

    # Center Loss 参数
    center_lr: float = 0.5             # 中心更新学习率
    center_momentum: float = 0.9       # 中心指数移动平均动量

    # 对抗边界约束参数
    adv_margin: float = 0.3            # 对抗边界 margin
    open_space_margin: float = 0.5     # 开放空间 margin


# =============================================================================
# 有偏样本图像生成配置 (4.3)
# =============================================================================
class BiasedImageGenConfig:
    """
    有偏样本图像生成配置

    生成策略:
      1. 几何变换增广 (模拟多方位角)
      2. 特征空间插值 (向原型方向插值)
      3. 对抗样本增强 (对抗扰动生成)
      4. Mixup 增广 (同类样本混合)
    """
    # 几何变换参数 (模拟 SAR 多方位角)
    rotation_range: float = 30.0       # 旋转角度范围 ±degrees
    rotation_steps: int = 6            # 旋转步数
    scale_range: tuple = (0.85, 1.15)  # 缩放范围
    translate_range: float = 0.1       # 平移范围 (比例)

    # SAR 特定增广
    speckle_noise_std: float = 0.1     # 散斑噪声标准差
    brightness_range: tuple = (0.8, 1.2)  # 亮度变化范围

    # 特征空间插值
    interpolation_steps: int = 5       # 插值步数 (从有偏到原型)
    interpolation_alpha: float = 0.3   # 插值强度

    # Mixup 参数
    mixup_alpha: float = 0.4           # Beta 分布参数
    mixup_num: int = 3                 # 每个有偏样本生成的混合样本数

    # 对抗样本生成
    adv_epsilon: float = 0.01          # 对抗扰动强度
    adv_steps: int = 5                 # PGD 步数
    adv_step_size: float = 0.003       # PGD 步长


# =============================================================================
# 数据增强配置
# =============================================================================
class AugmentConfig:
    """数据增强配置"""
    image_size: int = 128              # MSTAR 图像尺寸 (相比舰船更小)
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5    # SAR 图像上下翻转也有意义
    rotation_degrees: float = 15.0

    # 归一化参数 (ImageNet)
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]


# =============================================================================
# 可视化配置
# =============================================================================
class VizConfig:
    """可视化配置"""
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1500
    tsne_seed: int = 42
    max_per_class: int = 300           # t-SNE 每类最大采样数
    figure_dpi: int = 150
    figure_size: tuple = (12, 8)


# =============================================================================
# 默认配置实例
# =============================================================================
siamese_model_cfg = SiameseModelConfig()
siamese_training_cfg = SiameseTrainingConfig()
bias_judge_cfg = BiasJudgeConfig()
feat_optim_cfg = FeatOptimConfig()
biased_gen_cfg = BiasedImageGenConfig()
augment_cfg = AugmentConfig()
viz_cfg = VizConfig()

"""
有偏样本图像生成模块 (4.3 核心)

================================================================================
  设计背景:
    当检测到库内有偏样本后, 需要通过图像生成来扩充该类的训练数据,
    从而优化该类在特征空间中的分布, 提升识别精度.

  针对 SAR 图像的生成策略:
    策略1 - 多方位角几何变换:
      SAR 图像对方位角高度敏感, 通过旋转、缩放、平移等几何变换
      模拟不同方位角下的目标外观
    策略2 - SAR 散斑噪声增广:
      添加/调整散斑噪声, 模拟不同成像条件
    策略3 - 特征空间插值:
      在特征空间中, 从有偏样本向类原型方向插值, 生成更接近类中心的特征
    策略4 - 同类 Mixup:
      将有偏样本与同类无偏样本混合, 生成中间样本
    策略5 - 对抗样本生成 (PGD):
      在保持语义不变的前提下, 生成轻微扰动的图像

  生成后的特征分布优化:
    1. 生成的图像通过孪生网络提取新的切片指纹
    2. 新指纹加入特征库, 更新类原型
    3. 可视化前后对比, 展示有偏样本的分布改善
================================================================================
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from feature_library import FeatureLibrary


# =============================================================================
# 策略1: 多方位角几何变换 (SAR 专用)
# =============================================================================
class MultiAzimuthGenerator:
    """
    多方位角变换生成器

    通过一系列几何变换模拟 SAR 图像在不同方位角下的外观:
      - 旋转: 模拟方位角变化 (SAR 最敏感的因素)
      - 缩放: 模拟不同距离/分辨率
      - 翻转: 模拟对称视角

    Args:
        rotation_range: 旋转角度范围 ±degrees
        rotation_steps: 旋转步数 (每步角度 = 2*range/steps)
        scale_range: 缩放比例范围
        include_flip: 是否包含翻转
    """

    def __init__(self,
                 rotation_range: float = 30.0,
                 rotation_steps: int = 6,
                 scale_range: Tuple[float, float] = (0.85, 1.15),
                 include_flip: bool = True):
        self.rotation_range = rotation_range
        self.rotation_steps = rotation_steps
        self.scale_range = scale_range
        self.include_flip = include_flip

    def generate(self, image: Image.Image) -> List[Image.Image]:
        """
        为单张图像生成多方位角变换

        Args:
            image: PIL Image
        Returns:
            生成的图像列表
        """
        generated = []

        # 旋转变换
        angles = np.linspace(-self.rotation_range, self.rotation_range,
                            self.rotation_steps)
        for angle in angles:
            if abs(angle) < 1e-3:
                continue  # 跳过 0°
            rotated = TF.rotate(image, float(angle), fill=0)
            generated.append(rotated)

        # 缩放变换
        w, h = image.size
        for scale in [self.scale_range[0], self.scale_range[1]]:
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled = image.resize((new_w, new_h), Image.BILINEAR)
            # 居中裁剪/填充回原尺寸
            scaled = TF.center_crop(scaled, [h, w]) if scale > 1.0 else \
                     TF.pad(scaled, [(w - new_w) // 2, (h - new_h) // 2,
                                     (w - new_w + 1) // 2, (h - new_h + 1) // 2])
            generated.append(scaled)

        # 翻转变换
        if self.include_flip:
            generated.append(TF.hflip(image))
            generated.append(TF.vflip(image))

        return generated


# =============================================================================
# 策略2: SAR 散斑噪声增广
# =============================================================================
class SpeckleNoiseAugmentor:
    """
    SAR 散斑噪声增广

    SAR 图像固有散斑噪声, 通过调整噪声水平模拟不同成像条件

    散斑噪声模型: y = x * n, 其中 n ~ Gamma(L, 1/L)
    简化实现: y = x * (1 + σ * z), z ~ N(0, 1)

    Args:
        noise_levels: 噪声标准差列表
        brightness_range: 亮度调节范围
    """

    def __init__(self,
                 noise_levels: List[float] = None,
                 brightness_range: Tuple[float, float] = (0.8, 1.2)):
        self.noise_levels = noise_levels or [0.05, 0.1, 0.15, 0.2]
        self.brightness_range = brightness_range

    def generate(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        为单张图像生成不同噪声水平的变体

        Args:
            image_tensor: [C, H, W] 图像张量
        Returns:
            生成的图像张量列表
        """
        generated = []

        for sigma in self.noise_levels:
            # 乘性散斑噪声
            noise = 1.0 + sigma * torch.randn_like(image_tensor)
            noisy = image_tensor * noise
            noisy = noisy.clamp(0, 1) if image_tensor.max() <= 1.0 else noisy.clamp(0, 255)
            generated.append(noisy)

        # 亮度变换
        for factor in [self.brightness_range[0], self.brightness_range[1]]:
            bright = image_tensor * factor
            bright = bright.clamp(0, 1) if image_tensor.max() <= 1.0 else bright.clamp(0, 255)
            generated.append(bright)

        return generated


# =============================================================================
# 策略3: 特征空间插值
# =============================================================================
class FeatureInterpolator:
    """
    特征空间插值生成器

    将有偏样本的切片指纹向类原型方向插值,
    生成一系列从有偏到无偏的过渡特征

    f_interp(α) = normalize(α * f_biased + (1-α) * μ_class)
    α 从 1.0 (完全有偏) 变化到 0.0 (原型)

    这些插值特征可用于:
      1. 直接加入特征库, 充实类分布
      2. 可视化有偏→无偏的渐变过程

    Args:
        steps: 插值步数
        alpha_range: α 范围
    """

    def __init__(self,
                 steps: int = 5,
                 alpha_range: Tuple[float, float] = (0.3, 0.8)):
        self.steps = steps
        self.alpha_range = alpha_range

    def interpolate(self,
                    biased_feat: torch.Tensor,
                    prototype: torch.Tensor) -> torch.Tensor:
        """
        特征空间插值

        Args:
            biased_feat: 有偏样本指纹 [D]
            prototype: 类原型 [D]
        Returns:
            插值特征 [steps, D]
        """
        alphas = torch.linspace(self.alpha_range[0], self.alpha_range[1], self.steps)
        results = []
        for alpha in alphas:
            f_interp = alpha * biased_feat + (1.0 - alpha) * prototype
            f_interp = F.normalize(f_interp, dim=0)
            results.append(f_interp)
        return torch.stack(results)

    def interpolate_batch(self,
                          biased_feats: torch.Tensor,
                          prototype: torch.Tensor) -> torch.Tensor:
        """
        批量插值

        Args:
            biased_feats: [N, D]
            prototype: [D]
        Returns:
            [N * steps, D]
        """
        all_interp = []
        for i in range(biased_feats.size(0)):
            interp = self.interpolate(biased_feats[i], prototype)
            all_interp.append(interp)
        return torch.cat(all_interp, dim=0)


# =============================================================================
# 策略4: 同类 Mixup 增广
# =============================================================================
class IntraClassMixup:
    """
    同类 Mixup: 将有偏样本与同类无偏样本混合

    f_mix = λ * f_biased + (1-λ) * f_unbiased
    λ ~ Beta(α, α)

    Args:
        alpha: Beta 分布参数
        num_generated: 每个有偏样本生成的混合样本数
    """

    def __init__(self, alpha: float = 0.4, num_generated: int = 3):
        self.alpha = alpha
        self.num_generated = num_generated

    def generate(self,
                 biased_feat: torch.Tensor,
                 unbiased_feats: torch.Tensor) -> torch.Tensor:
        """
        为单个有偏样本生成混合特征

        Args:
            biased_feat: 有偏样本指纹 [D]
            unbiased_feats: 同类无偏样本指纹 [M, D]
        Returns:
            混合特征 [num_generated, D]
        """
        if unbiased_feats.size(0) == 0:
            return biased_feat.unsqueeze(0).repeat(self.num_generated, 1)

        results = []
        for _ in range(self.num_generated):
            # 随机选择一个无偏样本
            idx = torch.randint(0, unbiased_feats.size(0), (1,)).item()
            unbiased = unbiased_feats[idx]

            # Beta 分布采样 λ
            lam = float(np.random.beta(self.alpha, self.alpha))
            lam = max(0.2, min(0.8, lam))  # 限制范围

            f_mix = lam * biased_feat + (1.0 - lam) * unbiased
            f_mix = F.normalize(f_mix, dim=0)
            results.append(f_mix)

        return torch.stack(results)


# =============================================================================
# 策略5: 对抗样本生成 (PGD)
# =============================================================================
class AdversarialGenerator:
    """
    对抗样本生成器 (PGD)

    在保持目标类别的前提下, 生成对抗扰动图像:
    让生成的图像特征更接近类原型

    x_adv = x + δ
    其中 δ = argmin_{‖δ‖<ε} d(f(x+δ), μ_class)

    Args:
        epsilon: 扰动上限
        steps: PGD 步数
        step_size: 单步步长
    """

    def __init__(self,
                 epsilon: float = 0.01,
                 steps: int = 5,
                 step_size: float = 0.003):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size

    @torch.enable_grad()
    def generate(self,
                 model: nn.Module,
                 image: torch.Tensor,
                 prototype: torch.Tensor) -> torch.Tensor:
        """
        为单张图像生成对抗样本

        Args:
            model: 孪生网络
            image: 输入图像 [1, C, H, W]
            prototype: 目标类原型 [D], 需在同一设备
        Returns:
            对抗图像 [1, C, H, W]
        """
        was_training = model.training
        model.eval()

        x_adv = image.clone().detach().requires_grad_(True)

        for _ in range(self.steps):
            fingerprint = model.extract_fingerprint(x_adv)  # [1, D]
            proto = prototype.unsqueeze(0)

            # 目标: 最小化到原型的余弦距离
            loss = 1.0 - F.cosine_similarity(fingerprint, proto)
            loss = loss.mean()

            loss.backward()

            with torch.no_grad():
                # 沿梯度反方向 (最小化距离 → 梯度下降)
                x_adv = x_adv - self.step_size * x_adv.grad.sign()

                # 投影到 ε-ball
                perturbation = x_adv - image
                perturbation = perturbation.clamp(-self.epsilon, self.epsilon)
                x_adv = (image + perturbation).detach().requires_grad_(True)

        if was_training:
            model.train()

        return x_adv.detach()


# =============================================================================
# 综合图像生成器
# =============================================================================
class BiasedImageGenerator:
    """
    综合有偏样本图像生成器

    整合所有生成策略, 对识别出的有偏样本进行增广:
      1. 图像级: 几何变换 + 散斑噪声 + 对抗样本
      2. 特征级: 特征插值 + 同类 Mixup

    使用流程:
      1. 从 BiasJudge 获取有偏样本列表
      2. 调用 generate_for_class() 为每类生成增广
      3. 生成的图像/特征加入特征库
      4. 可视化前后对比

    Args:
        feature_library: 特征库
        各生成器参数...
    """

    def __init__(self,
                 feature_library: FeatureLibrary,
                 rotation_range: float = 30.0,
                 rotation_steps: int = 6,
                 noise_levels: List[float] = None,
                 interp_steps: int = 5,
                 mixup_alpha: float = 0.4,
                 mixup_num: int = 3,
                 adv_epsilon: float = 0.01,
                 adv_steps: int = 5):
        self.library = feature_library

        # 图像级生成器
        self.azimuth_gen = MultiAzimuthGenerator(
            rotation_range=rotation_range,
            rotation_steps=rotation_steps,
        )
        self.speckle_gen = SpeckleNoiseAugmentor(
            noise_levels=noise_levels or [0.05, 0.1, 0.15],
        )
        self.adv_gen = AdversarialGenerator(
            epsilon=adv_epsilon,
            steps=adv_steps,
        )

        # 特征级生成器
        self.feat_interp = FeatureInterpolator(steps=interp_steps)
        self.mixup_gen = IntraClassMixup(alpha=mixup_alpha, num_generated=mixup_num)

    # -----------------------------------------------------------------
    # 图像级生成
    # -----------------------------------------------------------------
    def generate_images(self,
                        image_paths: List[str],
                        transform: T.Compose = None) -> Tuple[List[torch.Tensor], int]:
        """
        对有偏样本图像进行多方位角生成

        Args:
            image_paths: 有偏样本图像路径列表
            transform: 变换 (若提供, 对生成图像应用变换)
        Returns:
            (generated_tensors, count) 生成的图像张量列表, 总数
        """
        all_generated = []

        for path in image_paths:
            img = Image.open(path).convert("RGB")

            # 策略1: 多方位角变换
            azimuth_imgs = self.azimuth_gen.generate(img)

            for gen_img in azimuth_imgs:
                if transform is not None:
                    tensor = transform(gen_img)
                else:
                    tensor = T.ToTensor()(gen_img)
                all_generated.append(tensor)

        return all_generated, len(all_generated)

    # -----------------------------------------------------------------
    # 特征级生成
    # -----------------------------------------------------------------
    def generate_features(self,
                          class_id: int,
                          biased_feats: List[torch.Tensor],
                          unbiased_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        为某类的有偏样本在特征空间中生成增广特征

        Args:
            class_id: 类别 ID
            biased_feats: 有偏样本指纹列表
            unbiased_feats: 无偏样本指纹列表
        Returns:
            生成的特征 [M, D]
        """
        if not biased_feats:
            return torch.empty(0, self.library.embed_dim)

        entry = self.library.entries[class_id]
        entry.compute_stats()
        prototype = entry.prototype

        all_generated = []

        biased_tensor = torch.stack(biased_feats)

        # 策略3: 特征空间插值
        interp_feats = self.feat_interp.interpolate_batch(biased_tensor, prototype)
        all_generated.append(interp_feats)

        # 策略4: 同类 Mixup
        if unbiased_feats:
            unbiased_tensor = torch.stack(unbiased_feats)
            for bf in biased_feats:
                mixup_feats = self.mixup_gen.generate(bf, unbiased_tensor)
                all_generated.append(mixup_feats)

        if all_generated:
            return torch.cat(all_generated, dim=0)
        else:
            return torch.empty(0, self.library.embed_dim)

    # -----------------------------------------------------------------
    # 完整生成流程
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate_and_update(self,
                            model: nn.Module,
                            class_id: int,
                            biased_feats: List[torch.Tensor],
                            unbiased_feats: List[torch.Tensor],
                            biased_paths: List[str],
                            device: torch.device,
                            transform: T.Compose = None,
                            use_image_gen: bool = True,
                            use_feat_gen: bool = True) -> Dict:
        """
        对某类的有偏样本进行完整生成和特征库更新

        Args:
            model: 孪生网络
            class_id: 类别 ID
            biased_feats: 有偏特征
            unbiased_feats: 无偏特征
            biased_paths: 有偏图像路径
            device: 计算设备
            transform: 图像变换
            use_image_gen: 是否使用图像级生成
            use_feat_gen: 是否使用特征级生成
        Returns:
            生成统计信息
        """
        model.eval()
        stats = {
            "class_id": class_id,
            "n_biased": len(biased_feats),
            "n_image_generated": 0,
            "n_feat_generated": 0,
            "n_total_new": 0,
        }

        new_features = []

        # --- 图像级生成 ---
        if use_image_gen and biased_paths:
            gen_tensors, n_gen = self.generate_images(biased_paths, transform)
            stats["n_image_generated"] = n_gen

            # 提取生成图像的指纹
            if gen_tensors:
                gen_batch = torch.stack(gen_tensors).to(device)
                # 分批处理避免 OOM
                batch_size = 64
                for start in range(0, gen_batch.size(0), batch_size):
                    end = min(start + batch_size, gen_batch.size(0))
                    fps = model.extract_fingerprint(gen_batch[start:end])
                    new_features.append(fps.cpu())

        # --- 特征级生成 ---
        if use_feat_gen:
            feat_gen = self.generate_features(class_id, biased_feats, unbiased_feats)
            stats["n_feat_generated"] = feat_gen.size(0)
            if feat_gen.numel() > 0:
                new_features.append(feat_gen)

        # --- 更新特征库 ---
        if new_features:
            all_new = torch.cat(new_features, dim=0)
            stats["n_total_new"] = all_new.size(0)
            self.library.update_class(class_id, all_new, mode="append")

        return stats

    def generate_all_classes(self,
                             model: nn.Module,
                             unbiased_feats: Dict[int, List[torch.Tensor]],
                             biased_feats: Dict[int, List[torch.Tensor]],
                             biased_paths: Dict[int, List[str]],
                             device: torch.device,
                             transform: T.Compose = None,
                             logger=None) -> Dict[int, Dict]:
        """
        对所有类进行有偏样本生成和更新

        Returns:
            {class_id: stats}
        """
        _log = logger.info if logger else print

        _log("\n" + "=" * 60)
        _log("[BiasedImageGenerator] 开始生成有偏样本增广")
        _log("=" * 60)

        all_stats = {}
        total_new = 0

        for cid in self.library.class_ids:
            bf = biased_feats.get(cid, [])
            uf = unbiased_feats.get(cid, [])
            bp = biased_paths.get(cid, [])

            if not bf:
                continue

            stats = self.generate_and_update(
                model, cid, bf, uf, bp, device, transform
            )
            all_stats[cid] = stats
            total_new += stats["n_total_new"]

            name = self.library.entries[cid].class_name
            _log(f"  Class {cid} ({name}): "
                 f"{stats['n_biased']} biased → "
                 f"{stats['n_image_generated']} image + "
                 f"{stats['n_feat_generated']} feat = "
                 f"{stats['n_total_new']} new")

        _log(f"\n  Total new features generated: {total_new}")
        _log("=" * 60)

        return all_stats

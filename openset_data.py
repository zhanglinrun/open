"""
数据加载模块
包含数据样本构建、数据集划分、PyTorch Dataset类
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from openset_config import IMG_EXT, NAME2ID


# -------------------------
# 数据样本定义
# -------------------------
@dataclass
class Sample:
    """单个样本数据"""
    path: str
    orig_label: int  # 0..9


# -------------------------
# 数据集构建
# -------------------------
def build_samples(data_dir: str) -> List[Sample]:
    """
    扫描数据集文件夹,构建样本列表

    Args:
        data_dir: 数据集根目录

    Returns:
        样本列表
    """
    samples = []
    for name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, name)
        if not os.path.isdir(class_dir):
            continue
        if name not in NAME2ID:
            continue

        orig_id = NAME2ID[name]
        for fn in os.listdir(class_dir):
            fp = os.path.join(class_dir, fn)
            if os.path.isdir(fp):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXT:
                samples.append(Sample(path=fp, orig_label=orig_id))
    return samples


def split_train_test(samples: List[Sample],
                   seed: int = 42,
                   ratio: float = 0.5) -> Tuple[List[Sample], List[Sample]]:
    """
    分层划分训练集和测试集

    Args:
        samples: 样本列表
        seed: 随机种子
        ratio: 训练集比例

    Returns:
        (train_samples, test_samples)
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = {}

    # 按类别分组
    for s in samples:
        by_class.setdefault(s.orig_label, []).append(s)

    train, test = [], []

    # 对每类按比例划分
    for c, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * ratio)
        train.extend(lst[:n_train])
        test.extend(lst[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


# -------------------------
# PyTorch Dataset
# -------------------------
class ShipImageFolderDataset(Dataset):
    """
    船舶图像文件夹数据集

    Args:
        samples: 样本列表
        transform: 图像变换
        known_id_map: 原始ID到新ID的映射(已知类)
        unknown_label: 未知类的标签(默认-1)
    """

    def __init__(self,
                 samples: List[Sample],
                 transform,
                 known_id_map: Dict[int, int],
                 unknown_label: int = -1):
        self.samples = samples
        self.transform = transform
        self.known_id_map = known_id_map
        self.unknown_label = unknown_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)

        # 如果是已知类,映射到0..6;如果是未知类,设为-1
        y = self.known_id_map.get(s.orig_label, self.unknown_label)

        return x, torch.tensor(y, dtype=torch.long), s.orig_label, s.path


# -------------------------
# 数据变换
# -------------------------
def get_train_transforms(image_size=224,
                       horizontal_flip_prob=0.5,
                       color_jitter_brightness=0.2,
                       color_jitter_contrast=0.2,
                       color_jitter_saturation=0.2,
                       color_jitter_hue=0.05,
                       normalize_mean=[0.485, 0.456, 0.406],
                       normalize_std=[0.229, 0.224, 0.225]):
    """
    获取训练数据增强

    Args:
        image_size: 图像大小
        horizontal_flip_prob: 水平翻转概率
        color_jitter_*: 颜色抖动参数
        normalize_mean/std: 归一化参数

    Returns:
        transforms.Compose对象
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])


def get_test_transforms(image_size=224,
                      normalize_mean=[0.485, 0.456, 0.406],
                      normalize_std=[0.229, 0.224, 0.225]):
    """
    获取测试数据变换(不进行数据增强)

    Args:
        image_size: 图像大小
        normalize_mean/std: 归一化参数

    Returns:
        transforms.Compose对象
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])


# -------------------------
# 数据加载器
# -------------------------
def get_dataloader(dataset,
                  batch_size,
                  shuffle=True,
                  num_workers=4,
                  pin_memory=True,
                  drop_last=False):
    """
    获取数据加载器

    Args:
        dataset: Dataset对象
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        drop_last: 是否丢弃最后不完整的batch

    Returns:
        DataLoader对象
    """

    def collate(batch):
        """自定义collate函数"""
        xs, ys, origs, paths = zip(*batch)
        return torch.stack(xs, 0), torch.stack(ys, 0), torch.tensor(origs, dtype=torch.long), list(paths)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=drop_last
    )

"""
孪生网络数据加载模块 - MSTAR 数据集

采样模式:
  - Single: 标准单图模式 (带标签), 配合 BalancedBatchSampler 做 Batch Hard Mining

MSTAR 数据集结构 (预期):
  MSTAR/
  ├── train/
  │   ├── 2S1/
  │   │   ├── img_001.png
  │   │   └── ...
  │   ├── BRDM_2/
  │   └── ...
  └── test/
      ├── 2S1/
      └── ...
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T

from siamese_config import IMG_EXT, NAME2ID, KNOWN_ORIG_IDS, UNKNOWN_ORIG_IDS, AugmentConfig


# =============================================================================
# 数据样本
# =============================================================================
@dataclass
class Sample:
    """单个样本"""
    path: str
    orig_label: int  # 原始标签 (0..9)


# =============================================================================
# 数据集构建
# =============================================================================
def build_samples(data_dir: str) -> List[Sample]:
    """
    扫描数据集文件夹, 构建样本列表

    Args:
        data_dir: 根目录, 内含各类别子文件夹
    Returns:
        样本列表
    """
    samples = []
    for name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, name)
        if not os.path.isdir(class_dir):
            continue
        if name not in NAME2ID:
            continue

        orig_id = NAME2ID[name]
        for fn in sorted(os.listdir(class_dir)):
            fp = os.path.join(class_dir, fn)
            if os.path.isdir(fp):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXT:
                samples.append(Sample(path=fp, orig_label=orig_id))
    return samples


def build_samples_from_split(data_dir: str, split: str) -> List[Sample]:
    """扫描 train/test 子目录"""
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Split folder not found: {split_dir}")
    return build_samples(split_dir)


def split_train_test(samples: List[Sample],
                     seed: int = 42,
                     ratio: float = 0.8) -> Tuple[List[Sample], List[Sample]]:
    """分层划分训练集和测试集"""
    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.orig_label].append(s)

    train, test = [], []
    for c, lst in by_class.items():
        rng.shuffle(lst)
        n_train = int(len(lst) * ratio)
        train.extend(lst[:n_train])
        test.extend(lst[n_train:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def group_by_class(samples: List[Sample]) -> Dict[int, List[Sample]]:
    """按类别分组"""
    groups = defaultdict(list)
    for s in samples:
        groups[s.orig_label].append(s)
    return dict(groups)


# =============================================================================
# 数据变换
# =============================================================================
def get_train_transforms(cfg: AugmentConfig = None) -> T.Compose:
    """训练数据增强 (适合 SAR 图像)"""
    if cfg is None:
        cfg = AugmentConfig()
    return T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.RandomHorizontalFlip(p=cfg.horizontal_flip_prob),
        T.RandomVerticalFlip(p=cfg.vertical_flip_prob),
        T.RandomRotation(degrees=cfg.rotation_degrees),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])


def get_test_transforms(cfg: AugmentConfig = None) -> T.Compose:
    """测试数据变换 (无增强)"""
    if cfg is None:
        cfg = AugmentConfig()
    return T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])


# =============================================================================
# 单图数据集 (标准模式)
# =============================================================================
class SingleImageDataset(Dataset):
    """
    标准单图数据集

    Args:
        samples: 样本列表
        transform: 图像变换
        known_id_map: 原始ID→新ID映射 (已知类)
        unknown_label: 未知类标签 (默认 -1)
    """

    def __init__(self,
                 samples: List[Sample],
                 transform: T.Compose,
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
        y = self.known_id_map.get(s.orig_label, self.unknown_label)
        return x, torch.tensor(y, dtype=torch.long), s.orig_label, s.path


# =============================================================================
# 在线 Batch 采样器 (用于 Batch Hard Mining)
# =============================================================================
class BalancedBatchSampler(Sampler):
    """
    平衡批采样器: 每个 batch 包含 P 个类, 每类 K 个样本

    这样可以在 batch 内进行 hard negative mining

    Args:
        labels: 所有样本的标签
        n_classes: 每 batch 采样的类别数 P
        n_samples: 每类采样的样本数 K
    """

    def __init__(self, labels: List[int], n_classes: int = 7, n_samples: int = 4):
        self.labels = labels
        self.n_classes = n_classes
        self.n_samples = n_samples

        # 按标签分组索引
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_to_indices[label].append(i)

        self.unique_labels = list(self.label_to_indices.keys())
        self.batch_size = n_classes * n_samples

    def __iter__(self):
        """每次迭代随机选 P 个类, 每类随机 K 个样本"""
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            classes = random.sample(
                self.unique_labels,
                min(self.n_classes, len(self.unique_labels))
            )
            indices = []
            for c in classes:
                c_indices = self.label_to_indices[c]
                if len(c_indices) >= self.n_samples:
                    indices.extend(random.sample(c_indices, self.n_samples))
                else:
                    indices.extend(random.choices(c_indices, k=self.n_samples))
            yield indices

    def __len__(self):
        return len(self.labels) // self.batch_size


# =============================================================================
# DataLoader 工厂函数
# =============================================================================
def create_single_loader(samples: List[Sample],
                         transform: T.Compose,
                         known_id_map: Dict[int, int],
                         batch_size: int = 128,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True) -> DataLoader:
    """创建标准单图 DataLoader"""
    ds = SingleImageDataset(samples, transform, known_id_map)

    def collate(batch):
        xs, ys, origs, paths = zip(*batch)
        return (torch.stack(xs), torch.stack(ys),
                torch.tensor(origs, dtype=torch.long), list(paths))

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate, drop_last=False,
    )


def create_batch_hard_loader(samples: List[Sample],
                             transform: T.Compose,
                             known_id_map: Dict[int, int],
                             n_classes: int = 7,
                             n_samples: int = 8,
                             num_workers: int = 4) -> DataLoader:
    """
    创建 Batch Hard Mining DataLoader
    每 batch = P classes × K samples
    """
    # 只保留已知类
    known_samples = [s for s in samples if s.orig_label in known_id_map]
    labels = [known_id_map[s.orig_label] for s in known_samples]

    ds = SingleImageDataset(known_samples, transform, known_id_map)
    sampler = BalancedBatchSampler(labels, n_classes=n_classes, n_samples=n_samples)

    def collate(batch):
        xs, ys, origs, paths = zip(*batch)
        return (torch.stack(xs), torch.stack(ys),
                torch.tensor(origs, dtype=torch.long), list(paths))

    return DataLoader(
        ds, batch_sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate,
    )

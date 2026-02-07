"""
可视化模块 - 切片指纹分布可视化

核心功能:
  1. t-SNE 特征分布可视化 (优化前后对比)
  2. 有偏/无偏样本分布标注
  3. 特征分布度量曲线
  4. 类间距离矩阵热力图
  5. 有偏样本生成前后对比
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE

from siamese_config import ID2NAME, UNKNOWN_ORIG_IDS, KNOWN_ORIG_IDS


# =============================================================================
# 颜色方案
# =============================================================================
KNOWN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
UNKNOWN_COLOR = "#000000"
BIASED_MARKER = "x"
UNBIASED_MARKER = "o"

# 未知子类名 / 颜色
_UNK_NAMES = {5: "SN_9563*", 8: "ZIL131*", 9: "ZSU_23_4*"}
_UNK_COLORS = {5: "#555555", 8: "#888888", 9: "#AAAAAA"}


# =============================================================================
# 工具: 2σ 置信椭圆
# =============================================================================
def _draw_confidence_ellipse(ax, points, color, alpha=0.15, n_std=2.0,
                              linestyle="-", linewidth=1.5):
    """
    在 ax 上绘制 2σ 置信椭圆

    Args:
        points: [N, 2] t-SNE 坐标
        color: 椭圆颜色
        n_std: 标准差倍数 (2σ 覆盖约 95% 数据)
    """
    if points.shape[0] < 3:
        return
    mean = points.mean(axis=0)
    cov = np.cov(points.T)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return

    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor=color, alpha=alpha, edgecolor=color,
                  linestyle=linestyle, linewidth=linewidth)
    ax.add_patch(ell)
    return width * height  # 面积 (用于度量紧凑度)


def _build_color_map(orig_ids):
    """为已知类构建颜色映射"""
    known_oids = sorted(o for o in set(orig_ids) if o not in UNKNOWN_ORIG_IDS)
    oid2color = {}
    for ci, oid in enumerate(known_oids):
        oid2color[oid] = KNOWN_COLORS[ci % len(KNOWN_COLORS)]
    return oid2color


def _stratified_sample(feats_np, origs_arr, max_n, rng):
    """按 orig_label 分层采样"""
    picked = []
    for oid in sorted(set(origs_arr)):
        idx = np.where(origs_arr == oid)[0]
        take = min(max_n, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    return np.array(picked)


# =============================================================================
# 10. 4.2 核心可视化: 无偏样本特征分布优化 (训练前散→优化后聚)
# =============================================================================
def plot_optimization_effect(feats_before: torch.Tensor,
                              labels_before: torch.Tensor,
                              origs_before: List[int],
                              feats_after: torch.Tensor,
                              labels_after: torch.Tensor,
                              origs_after: List[int],
                              save_path: str,
                              metrics_before: Dict = None,
                              metrics_after: Dict = None,
                              max_per_class: int = 300,
                              seed: int = 42):
    """
    4.2 核心可视化: 无偏样本深度特征分布优化

    展示 "训练前(散) → 优化后(聚)" 的效果:
      左图: 优化前 - 特征散乱, 类间混叠, 椭圆大
      右图: 优化后 - 特征紧凑, 类间分离, 椭圆小

    使用共享 t-SNE 坐标系 + 2σ 置信椭圆, 直观展示紧凑度变化.
    """
    rng = np.random.default_rng(seed)

    fb_np = feats_before.numpy() if isinstance(feats_before, torch.Tensor) else feats_before
    fa_np = feats_after.numpy() if isinstance(feats_after, torch.Tensor) else feats_after
    origs_b = np.array(origs_before)
    origs_a = np.array(origs_after)

    # 分层采样
    idx_b = _stratified_sample(fb_np, origs_b, max_per_class, rng)
    idx_a = _stratified_sample(fa_np, origs_a, max_per_class, rng)

    sub_b = fb_np[idx_b]
    sub_a = fa_np[idx_a]
    origs_sub_b = origs_b[idx_b]
    origs_sub_a = origs_a[idx_a]

    # 拼接跑一次 t-SNE (共享坐标系)
    combined = np.concatenate([sub_b, sub_a], axis=0)
    Nb = sub_b.shape[0]

    perp = min(40, max(5, combined.shape[0] // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                n_iter=2000, init="pca", early_exaggeration=12)
    Z = tsne.fit_transform(combined)
    Z_b = Z[:Nb]
    Z_a = Z[Nb:]

    oid2color = _build_color_map(list(set(origs_sub_b)) + list(set(origs_sub_a)))

    fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=150)

    for panel_idx, (Z_panel, origs_panel, ax) in enumerate([
        (Z_b, origs_sub_b, axes[0]),
        (Z_a, origs_sub_a, axes[1]),
    ]):
        drawn = set()
        # 对每个已知类画散点 + 置信椭圆
        ellipse_areas = {}
        for oid in sorted(set(origs_panel)):
            mask = origs_panel == oid
            is_unk = oid in UNKNOWN_ORIG_IDS

            if is_unk:
                name = _UNK_NAMES.get(oid, f"Unk_{oid}")
                color = _UNK_COLORS.get(oid, "#666666")
                marker = "^"
                size = 40
            else:
                name = ID2NAME.get(oid, str(oid))
                color = oid2color.get(oid, "#888888")
                marker = "o"
                size = 30

            ax.scatter(Z_panel[mask, 0], Z_panel[mask, 1],
                       s=size, alpha=0.7, c=color, marker=marker,
                       label=name if name not in drawn else "",
                       linewidths=0.3,
                       edgecolors="white" if is_unk else "none")
            drawn.add(name)

            # 2σ 置信椭圆 (仅已知类)
            if not is_unk and mask.sum() >= 3:
                area = _draw_confidence_ellipse(ax, Z_panel[mask], color,
                                                 alpha=0.12, n_std=2.0)
                if area is not None:
                    ellipse_areas[name] = area

        # 标题和度量信息
        metrics = metrics_before if panel_idx == 0 else metrics_after
        stage = "Before Optimization" if panel_idx == 0 else "After Optimization"
        subtitle = f"{stage}"
        if metrics:
            subtitle += (f"\nFisher={metrics.get('fisher_ratio', 0):.1f}  "
                         f"Intra={metrics.get('intra_compactness', 0):.4f}  "
                         f"Inter={metrics.get('inter_separation', 0):.4f}")

        # 计算平均椭圆面积
        if ellipse_areas:
            avg_area = np.mean(list(ellipse_areas.values()))
            subtitle += f"\nAvg Cluster Area={avg_area:.0f}"

        ax.set_title(subtitle, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=1.5, fontsize=8, loc="best", frameon=True, ncol=2)

    plt.suptitle("4.2 Unbiased Sample Feature Distribution Optimization\n"
                 "Shared t-SNE + 2σ Confidence Ellipses (dashed = class boundary)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 11. 4.3 核心可视化: 有偏样本图像生成与优化
# =============================================================================
def plot_biased_generation_effect(feats_before: torch.Tensor,
                                   labels_before: torch.Tensor,
                                   origs_before: List[int],
                                   is_biased_before: List[bool],
                                   feats_after: torch.Tensor,
                                   labels_after: torch.Tensor,
                                   origs_after: List[int],
                                   save_path: str,
                                   augmented_feats: Optional[Dict[int, torch.Tensor]] = None,
                                   metrics_before: Dict = None,
                                   metrics_after: Dict = None,
                                   max_per_class: int = 300,
                                   seed: int = 42):
    """
    4.3 核心可视化: 有偏样本图像生成与优化效果

    三面板:
      左: 生成前 - 有偏(✕)在簇外, 无偏(●)在簇内, 2σ 椭圆
      中: 生成的增广特征(★) + 原始特征
      右: 生成+微调后 - 有偏无偏都紧密聚在一起, 椭圆缩小

    关键: 有偏样本从簇边缘→簇中心的视觉变化
    """
    rng = np.random.default_rng(seed)

    fb_np = feats_before.numpy() if isinstance(feats_before, torch.Tensor) else feats_before
    fa_np = feats_after.numpy() if isinstance(feats_after, torch.Tensor) else feats_after
    lb_np = labels_before.numpy() if isinstance(labels_before, torch.Tensor) else labels_before
    la_np = labels_after.numpy() if isinstance(labels_after, torch.Tensor) else labels_after
    origs_b = np.array(origs_before)
    origs_a = np.array(origs_after)
    biased_b = np.array(is_biased_before)

    # 只取已知类 (有偏/无偏仅针对已知类)
    known_mask_b = lb_np >= 0
    known_mask_a = la_np >= 0

    fb_known = fb_np[known_mask_b]
    origs_known_b = origs_b[known_mask_b]  # 使用原始标签 (正确对应 ID2NAME)
    biased_known = biased_b[known_mask_b]

    fa_known = fa_np[known_mask_a]
    origs_known_a = origs_a[known_mask_a]

    # 按原始标签分层采样
    idx_b = _stratified_sample(fb_known, origs_known_b, max_per_class, rng)
    idx_a = _stratified_sample(fa_known, origs_known_a, max_per_class, rng)

    sub_b = fb_known[idx_b]
    sub_a = fa_known[idx_a]
    origs_sub_b = origs_known_b[idx_b]
    origs_sub_a = origs_known_a[idx_a]
    biased_sub_b = biased_known[idx_b]

    # 映射后标签也保留 (用于增广特征对齐)
    lbl_known_b = lb_np[known_mask_b]
    lbl_sub_b = lbl_known_b[idx_b]

    # 准备增广特征 (key 是映射后标签 0..K-1)
    aug_np = None
    aug_labels_np = None  # 映射后标签
    aug_origs_np = None   # 原始标签
    n_aug = 0
    if augmented_feats:
        aug_list, aug_lab_list, aug_orig_list = [], [], []
        for cid, gf in augmented_feats.items():
            gf_np = gf.numpy() if isinstance(gf, torch.Tensor) else gf
            if gf_np.shape[0] > 0:
                n_take = min(80, gf_np.shape[0])
                idx_g = rng.choice(gf_np.shape[0], size=n_take, replace=False)
                aug_list.append(gf_np[idx_g])
                aug_lab_list.append(np.full(n_take, cid))
                # 映射后标签→原始标签
                orig_id = KNOWN_ORIG_IDS[cid] if cid < len(KNOWN_ORIG_IDS) else cid
                aug_orig_list.append(np.full(n_take, orig_id))
        if aug_list:
            aug_np = np.concatenate(aug_list, axis=0)
            aug_labels_np = np.concatenate(aug_lab_list)
            aug_origs_np = np.concatenate(aug_orig_list)
            n_aug = aug_np.shape[0]

    # 拼接: before + augmented + after → 一次 t-SNE
    parts = [sub_b]
    if aug_np is not None:
        parts.append(aug_np)
    parts.append(sub_a)
    combined = np.concatenate(parts, axis=0)
    Nb = sub_b.shape[0]

    perp = min(40, max(5, combined.shape[0] // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                n_iter=2000, init="pca", early_exaggeration=12)
    Z = tsne.fit_transform(combined)
    Z_b = Z[:Nb]
    Z_aug = Z[Nb:Nb + n_aug] if n_aug > 0 else None
    Z_a = Z[Nb + n_aug:]

    # 颜色映射 (基于原始标签, 与 ID2NAME 一致)
    oid2color = _build_color_map(
        list(set(origs_sub_b.tolist())) + list(set(origs_sub_a.tolist())))

    fig, axes = plt.subplots(1, 3, figsize=(33, 10), dpi=150)

    # === 左图: 生成前, 有偏(✕) vs 无偏(●) ===
    ax = axes[0]
    for oid in sorted(set(origs_sub_b)):
        color = oid2color.get(oid, "#888888")
        name = ID2NAME.get(oid, f"C{oid}")

        # 无偏样本 (圆点, 簇核心)
        mask_ub = (origs_sub_b == oid) & (~biased_sub_b)
        if mask_ub.any():
            ax.scatter(Z_b[mask_ub, 0], Z_b[mask_ub, 1],
                       s=25, alpha=0.6, c=color, marker="o",
                       label=f"{name} (unbiased)", linewidths=0)

        # 有偏样本 (大叉, 簇边缘)
        mask_bi = (origs_sub_b == oid) & biased_sub_b
        if mask_bi.any():
            ax.scatter(Z_b[mask_bi, 0], Z_b[mask_bi, 1],
                       s=100, alpha=0.95, c=color, marker="X",
                       label=f"{name} (biased)", linewidths=1.5,
                       edgecolors="black", zorder=5)

        # 2σ 椭圆 (仅无偏样本的椭圆, 有偏应在椭圆外)
        if mask_ub.any() and mask_ub.sum() >= 3:
            _draw_confidence_ellipse(ax, Z_b[mask_ub], color,
                                      alpha=0.10, n_std=2.0, linewidth=1.5)

    n_biased = biased_sub_b.sum()
    title_l = (f"(a) Before Generation\n"
               f"✕ = {n_biased} biased (outliers)")
    if metrics_before:
        title_l += f"\nFisher={metrics_before.get('fisher_ratio', 0):.1f}"
    ax.set_title(title_l, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=1.0, fontsize=6, loc="best", frameon=True, ncol=2)

    # === 中图: 原始 + 增广(★) ===
    ax = axes[1]
    for oid in sorted(set(origs_sub_b)):
        color = oid2color.get(oid, "#888888")
        name = ID2NAME.get(oid, f"C{oid}")
        mask = origs_sub_b == oid
        ax.scatter(Z_b[mask, 0], Z_b[mask, 1],
                   s=15, alpha=0.4, c=color, marker="o", linewidths=0,
                   label=name)

    if Z_aug is not None and aug_origs_np is not None:
        for orig_id in sorted(np.unique(aug_origs_np)):
            color = oid2color.get(orig_id, "#888888")
            name = ID2NAME.get(orig_id, f"C{orig_id}")
            mask_g = aug_origs_np == orig_id
            ax.scatter(Z_aug[mask_g, 0], Z_aug[mask_g, 1],
                       s=120, alpha=0.9, c=color, marker="*",
                       edgecolors="black", linewidths=0.6,
                       label=f"{name} (generated)", zorder=5)

    title_m = f"(b) Generated Augmented Features\n★ = {n_aug} new samples"
    ax.set_title(title_m, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=1.0, fontsize=6, loc="best", frameon=True, ncol=2)

    # === 右图: 生成+微调后, 全部紧密聚合 ===
    ax = axes[2]
    for oid in sorted(set(origs_sub_a)):
        color = oid2color.get(oid, "#888888")
        name = ID2NAME.get(oid, f"C{oid}")
        mask = origs_sub_a == oid
        ax.scatter(Z_a[mask, 0], Z_a[mask, 1],
                   s=30, alpha=0.7, c=color, marker="o",
                   label=name, linewidths=0)

        # 2σ 椭圆
        if mask.sum() >= 3:
            _draw_confidence_ellipse(ax, Z_a[mask], color,
                                      alpha=0.10, n_std=2.0, linewidth=1.5)

    title_r = "(c) After Generation + Finetune\nAll samples tightly clustered"
    if metrics_after:
        title_r += f"\nFisher={metrics_after.get('fisher_ratio', 0):.1f}"
    ax.set_title(title_r, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=1.5, fontsize=7, loc="best", frameon=True, ncol=2)

    plt.suptitle("4.3 Biased Sample Image Generation & Feature Optimization\n"
                 "Shared t-SNE: (a) biased✕ outside → (b) augmentation★ → (c) all tightly clustered●",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 1. t-SNE 基础可视化
# =============================================================================
@torch.no_grad()
def plot_tsne_fingerprints(feats: torch.Tensor,
                           labels: torch.Tensor,
                           orig_labels: List[int],
                           save_path: str,
                           title: str = "t-SNE of Slice Fingerprints",
                           max_per_class: int = 300,
                           perplexity: float = 30.0,
                           seed: int = 42,
                           figsize: Tuple = (12, 8),
                           dpi: int = 150,
                           prototypes: Optional[torch.Tensor] = None,
                           id2name: Dict = None):
    """
    切片指纹 t-SNE 可视化

    Args:
        feats: 切片指纹 [N, D]
        labels: 新标签 [N] (-1 为未知)
        orig_labels: 原始标签列表
        save_path: 保存路径
        title: 标题
        prototypes: 原型 [K, D] (可选, 用于标注)
        id2name: ID→名称映射
    """
    if id2name is None:
        id2name = ID2NAME

    feats_np = feats.numpy() if isinstance(feats, torch.Tensor) else feats
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    # 构建显示标签
    display_labels = []
    for i, orig in enumerate(orig_labels):
        if orig in UNKNOWN_ORIG_IDS:
            display_labels.append("unknown")
        else:
            display_labels.append(id2name.get(orig, f"Class_{orig}"))
    display_labels = np.array(display_labels)

    # 分层采样
    rng = np.random.default_rng(seed)
    picked_idx = []
    for name in sorted(set(display_labels), key=lambda s: (s == "unknown", s)):
        idx = np.where(display_labels == name)[0]
        take = min(max_per_class, len(idx))
        picked_idx.extend(rng.choice(idx, size=take, replace=False).tolist())

    picked_idx = np.array(picked_idx)
    X = feats_np[picked_idx]
    disp = display_labels[picked_idx]

    # 如果有原型, 加入 t-SNE
    proto_labels = []
    if prototypes is not None:
        protos_np = prototypes.numpy() if isinstance(prototypes, torch.Tensor) else prototypes
        X = np.concatenate([X, protos_np], axis=0)
        for i in range(protos_np.shape[0]):
            proto_labels.append(f"proto_{i}")

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, X.shape[0] // 4)),
        early_exaggeration=18,
        learning_rate=200,
        n_iter=1500,
        init="pca",
        random_state=seed,
    )
    Z = tsne.fit_transform(X)

    # 分离样本和原型
    Z_samples = Z[:len(picked_idx)]
    Z_protos = Z[len(picked_idx):] if prototypes is not None else None

    # 绘图
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    uniq = sorted(set(disp), key=lambda s: (s == "unknown", s))
    cmap = plt.get_cmap("tab10")
    color_map = {}
    ci = 0
    for name in uniq:
        if name == "unknown":
            color_map[name] = UNKNOWN_COLOR
        else:
            color_map[name] = KNOWN_COLORS[ci % len(KNOWN_COLORS)]
            ci += 1

    for name in uniq:
        idx = np.where(disp == name)[0]
        pts = Z_samples[idx]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=30, alpha=0.7,
            c=color_map[name],
            label=name,
            linewidths=0,
            marker="o" if name != "unknown" else "^",
        )

    # 绘制原型
    if Z_protos is not None:
        ax.scatter(
            Z_protos[:, 0], Z_protos[:, 1],
            s=200, marker="*", c="red", edgecolors="black",
            linewidths=1.5, zorder=10, label="Prototypes",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=1.5, fontsize=9, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 2. 有偏/无偏标注的 t-SNE
# =============================================================================
@torch.no_grad()
def plot_tsne_bias_annotated(feats: torch.Tensor,
                             labels: torch.Tensor,
                             is_biased: List[bool],
                             save_path: str,
                             title: str = "Biased vs Unbiased Fingerprints",
                             id2name: Dict = None,
                             max_per_class: int = 300,
                             perplexity: float = 30.0,
                             seed: int = 42):
    """
    带有偏/无偏标注的 t-SNE 可视化

    无偏样本用圆点, 有偏样本用叉号
    """
    if id2name is None:
        id2name = ID2NAME

    feats_np = feats.numpy() if isinstance(feats, torch.Tensor) else feats
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    is_biased = np.array(is_biased)

    # 采样
    rng = np.random.default_rng(seed)
    N = feats_np.shape[0]
    if N > max_per_class * 10:
        idx = rng.choice(N, size=max_per_class * 10, replace=False)
        feats_np = feats_np[idx]
        labels_np = labels_np[idx]
        is_biased = is_biased[idx]

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, feats_np.shape[0] // 4)),
        random_state=seed,
        n_iter=1500,
        init="pca",
    )
    Z = tsne.fit_transform(feats_np)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    unique_labels = np.unique(labels_np[labels_np >= 0])
    cmap = plt.get_cmap("tab10")

    for i, c in enumerate(unique_labels):
        color = KNOWN_COLORS[i % len(KNOWN_COLORS)]
        name = id2name.get(c, f"Class_{c}")

        # 无偏
        mask_ub = (labels_np == c) & (~is_biased)
        if mask_ub.any():
            ax.scatter(Z[mask_ub, 0], Z[mask_ub, 1],
                      s=30, alpha=0.7, c=color,
                      marker="o", label=f"{name} (unbiased)",
                      linewidths=0)

        # 有偏
        mask_b = (labels_np == c) & is_biased
        if mask_b.any():
            ax.scatter(Z[mask_b, 0], Z[mask_b, 1],
                      s=60, alpha=0.9, c=color,
                      marker="x", label=f"{name} (biased)",
                      linewidths=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=1.2, fontsize=8, loc="best", frameon=True,
              ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 3. 优化前后对比可视化
# =============================================================================
def plot_tsne_before_after(feats_before: torch.Tensor,
                           labels_before: torch.Tensor,
                           orig_before: List[int],
                           feats_after: torch.Tensor,
                           labels_after: torch.Tensor,
                           orig_after: List[int],
                           save_path: str,
                           title_prefix: str = "Feature Distribution",
                           metrics_before: Dict = None,
                           metrics_after: Dict = None,
                           augmented_feats: Optional[Dict[int, torch.Tensor]] = None,
                           **kwargs):
    """
    优化前后 t-SNE 并排对比 (共享坐标系版本)

    关键: 将 before / after / augmented 特征拼接后跑一次 t-SNE,
    保证左右图使用相同坐标系, 可直接视觉对比.
    增广特征用星号在左图标注.
    """
    seed = kwargs.get("seed", 42)
    max_n = kwargs.get("max_per_class", 300)
    rng = np.random.default_rng(seed)

    fb_np = feats_before.numpy() if isinstance(feats_before, torch.Tensor) else feats_before
    fa_np = feats_after.numpy() if isinstance(feats_after, torch.Tensor) else feats_after
    origs_b = np.array(orig_before)
    origs_a = np.array(orig_after)

    # 按 orig_label 分层采样 (未知子类也各自采样)
    def _sample(feats_np, origs_arr):
        picked = []
        for oid in sorted(set(origs_arr)):
            idx = np.where(origs_arr == oid)[0]
            take = min(max_n, len(idx))
            picked.extend(rng.choice(idx, size=take, replace=False).tolist())
        return np.array(picked)

    idx_b = _sample(fb_np, origs_b)
    idx_a = _sample(fa_np, origs_a)

    sub_b = fb_np[idx_b]
    sub_a = fa_np[idx_a]
    origs_sub_b = origs_b[idx_b]
    origs_sub_a = origs_a[idx_a]

    # 准备增广特征 (与 Before 同一特征空间)
    aug_np = None
    aug_labels_np = None
    n_aug = 0
    if augmented_feats:
        aug_list, aug_lab_list = [], []
        for cid, gf in augmented_feats.items():
            gf_np = gf.numpy() if isinstance(gf, torch.Tensor) else gf
            if gf_np.shape[0] > 0:
                n_take = min(50, gf_np.shape[0])
                idx_g = rng.choice(gf_np.shape[0], size=n_take, replace=False)
                aug_list.append(gf_np[idx_g])
                aug_lab_list.append(np.full(n_take, cid))
        if aug_list:
            aug_np = np.concatenate(aug_list, axis=0)
            aug_labels_np = np.concatenate(aug_lab_list)
            n_aug = aug_np.shape[0]

    # 拼接 → 一次 t-SNE (before + augmented 同一空间, after 也拼入保持可比)
    # 增广特征由 before 模型生成, 和 sub_b 属同一特征空间
    parts = [sub_b]
    if aug_np is not None:
        parts.append(aug_np)
    parts.append(sub_a)
    combined = np.concatenate(parts, axis=0)
    Nb = sub_b.shape[0]

    perp = min(40, max(5, combined.shape[0] // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                n_iter=2000, init="pca", early_exaggeration=12)
    Z = tsne.fit_transform(combined)
    Z_b = Z[:Nb]
    Z_aug = Z[Nb:Nb + n_aug] if n_aug > 0 else None
    Z_a = Z[Nb + n_aug:]

    # 未知子类名
    UNK_NAMES = {5: "SN_9563*", 8: "ZIL131*", 9: "ZSU_23_4*"}
    UNK_COLORS = {5: "#444444", 8: "#777777", 9: "#AAAAAA"}

    fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=150)

    # 已知类 id → 颜色映射 (保证左右一致)
    known_oids = sorted(set(origs_sub_b) | set(origs_sub_a))
    known_oids = [o for o in known_oids if o not in UNKNOWN_ORIG_IDS]
    oid2color = {}
    for ci, oid in enumerate(known_oids):
        oid2color[oid] = KNOWN_COLORS[ci % len(KNOWN_COLORS)]

    for panel_idx, (Z_panel, origs_panel, ax, stage) in enumerate([
        (Z_b, origs_sub_b, axes[0], "Before Finetune"),
        (Z_a, origs_sub_a, axes[1], "After Finetune"),
    ]):
        drawn = set()
        for oid in sorted(set(origs_panel)):
            mask = origs_panel == oid
            is_unk = oid in UNKNOWN_ORIG_IDS
            if is_unk:
                name = UNK_NAMES.get(oid, f"Unk_{oid}")
                color = UNK_COLORS.get(oid, "#666666")
            else:
                name = ID2NAME.get(oid, str(oid))
                color = oid2color.get(oid, "#888888")

            marker = "^" if is_unk else "o"
            size = 40 if is_unk else 25
            ax.scatter(Z_panel[mask, 0], Z_panel[mask, 1],
                       s=size, alpha=0.75, c=color, marker=marker,
                       label=name if name not in drawn else "",
                       linewidths=0.3,
                       edgecolors="white" if is_unk else "none")
            drawn.add(name)

        # 在 Before 面板叠加增广特征 (增广特征由此模型生成, 属于此空间)
        if panel_idx == 0 and Z_aug is not None:
            for cid in sorted(np.unique(aug_labels_np)):
                mask_g = aug_labels_np == cid
                # 映射 ID → 原始 ID (增广特征的 key 是映射后的 0..K-1)
                orig_id = KNOWN_ORIG_IDS[cid] if cid < len(KNOWN_ORIG_IDS) else cid
                color = oid2color.get(orig_id, "#888888")
                name_g = ID2NAME.get(orig_id, f"C{cid}")
                ax.scatter(Z_aug[mask_g, 0], Z_aug[mask_g, 1],
                           s=120, alpha=0.9, c=color, marker="*",
                           edgecolors="black", linewidths=0.6,
                           label=f"{name_g} (augmented)",
                           zorder=5)

        subtitle = f"{title_prefix} ({stage})"
        if panel_idx == 0 and n_aug > 0:
            subtitle += f"\n★ = {n_aug} augmented samples"
        metrics = metrics_before if panel_idx == 0 else metrics_after
        if metrics:
            subtitle += (f"\nIntra={metrics.get('intra_compactness', 0):.4f}  "
                         f"Inter={metrics.get('inter_separation', 0):.4f}  "
                         f"Silhouette={metrics.get('silhouette_approx', 0):.3f}  "
                         f"Fisher={metrics.get('fisher_ratio', 0):.1f}")
        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=1.3, fontsize=7, loc="best", frameon=True, ncol=2)

    plt.suptitle("Shared t-SNE Coordinates (directly comparable)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 4. 类间距离矩阵热力图
# =============================================================================
def plot_class_distance_matrix(prototypes: torch.Tensor,
                               class_names: List[str],
                               save_path: str,
                               title: str = "Inter-class Distance Matrix"):
    """
    类间余弦距离矩阵热力图

    Args:
        prototypes: [K, D]
        class_names: 类名列表
    """
    protos = F.normalize(prototypes, dim=1)
    sim = torch.matmul(protos, protos.T).numpy()
    dist = 1.0 - sim

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(dist, cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # 数值标注
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{dist[i, j]:.2f}",
                   ha="center", va="center", fontsize=7,
                   color="white" if dist[i, j] > 0.5 else "black")

    plt.colorbar(im, label="Cosine Distance")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 5. 分布度量曲线
# =============================================================================
def plot_distribution_metrics(metrics_history: List[Dict],
                              save_path: str,
                              title: str = "Feature Distribution Metrics"):
    """
    绘制分布度量随训练的变化曲线

    Args:
        metrics_history: [{epoch, intra, inter, silhouette, fisher}, ...]
    """
    epochs = range(1, len(metrics_history) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Intra-class compactness (↓ better)
    ax = axes[0, 0]
    vals = [m.get("intra_compactness", 0) for m in metrics_history]
    ax.plot(epochs, vals, "b-o", markersize=3)
    ax.set_title("Intra-class Compactness (↓ better)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # Inter-class separation (↑ better)
    ax = axes[0, 1]
    vals = [m.get("inter_separation", 0) for m in metrics_history]
    ax.plot(epochs, vals, "r-o", markersize=3)
    ax.set_title("Inter-class Separation (↑ better)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # Silhouette (↑ better)
    ax = axes[1, 0]
    vals = [m.get("silhouette_approx", 0) for m in metrics_history]
    ax.plot(epochs, vals, "g-o", markersize=3)
    ax.set_title("Silhouette Score (↑ better)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # Fisher Ratio (↑ better)
    ax = axes[1, 1]
    vals = [m.get("fisher_ratio", 0) for m in metrics_history]
    ax.plot(epochs, vals, "m-o", markersize=3)
    ax.set_title("Fisher Discriminant Ratio (↑ better)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 6. 有偏样本生成效果对比
# =============================================================================
def plot_generation_effect(feats_before: torch.Tensor,
                           labels_before: torch.Tensor,
                           feats_generated: torch.Tensor,
                           feats_after: torch.Tensor,
                           labels_after: torch.Tensor,
                           class_id: int,
                           class_name: str,
                           save_path: str,
                           seed: int = 42):
    """
    有偏样本图像生成效果可视化

    三子图:
      左: 生成前 (标注有偏/无偏)
      中: 生成的新特征
      右: 生成后合并
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)

    # 合并所有特征做统一 t-SNE
    all_feats = torch.cat([feats_before, feats_generated, feats_after], dim=0)
    N_before = feats_before.size(0)
    N_gen = feats_generated.size(0)

    all_np = all_feats.numpy()

    perp = min(30, max(5, all_np.shape[0] // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, n_iter=1500, init="pca")
    Z = tsne.fit_transform(all_np)

    Z_before = Z[:N_before]
    Z_gen = Z[N_before:N_before + N_gen]
    Z_after = Z[N_before + N_gen:]

    # 左图: 生成前
    ax = axes[0]
    ax.scatter(Z_before[:, 0], Z_before[:, 1],
              s=30, alpha=0.7, c="#1f77b4", label="Original")
    ax.set_title(f"Before Generation\n(Class: {class_name})", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    # 中图: 生成的特征
    ax = axes[1]
    ax.scatter(Z_gen[:, 0], Z_gen[:, 1],
              s=30, alpha=0.7, c="#ff7f0e", marker="^", label="Generated")
    ax.scatter(Z_before[:, 0], Z_before[:, 1],
              s=15, alpha=0.3, c="#1f77b4", label="Original (ref)")
    ax.set_title(f"Generated Features\n({N_gen} new samples)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    # 右图: 合并后
    ax = axes[2]
    ax.scatter(Z_after[:, 0], Z_after[:, 1],
              s=30, alpha=0.7, c="#2ca02c", label="After Merge")
    ax.set_title(f"After Generation\n(Total: {feats_after.size(0)})", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    plt.suptitle(f"Biased Sample Generation Effect - {class_name}",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 7. 训练曲线
# =============================================================================
def plot_training_curves(history: Dict[str, List],
                         save_path: str,
                         title: str = "Training Curves"):
    """绘制训练损失和度量曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    epochs = range(1, len(next(iter(history.values()))) + 1)

    # 损失曲线
    ax = axes[0]
    for key in history:
        if "loss" in key.lower():
            ax.plot(epochs, history[key], label=key)
    ax.set_title("Training Loss", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 其他指标
    ax = axes[1]
    for key in history:
        if "loss" not in key.lower() and len(history[key]) == len(list(epochs)):
            ax.plot(epochs, history[key], label=key)
    ax.set_title("Metrics", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# 8. 有偏生成三阶段对比 (4.3 核心可视化)
#    关键: 左右面板共用一次 t-SNE (保证坐标系一致, 可直接对比)
# =============================================================================
def plot_generation_three_stage(feats_before: torch.Tensor,
                                labels_before: torch.Tensor,
                                origs_before: List[int],
                                generated_feats: Dict[int, torch.Tensor],
                                feats_after: torch.Tensor,
                                labels_after: torch.Tensor,
                                origs_after: List[int],
                                save_path: str,
                                metrics_before: Dict = None,
                                metrics_after: Dict = None,
                                max_per_class: int = 300,
                                seed: int = 42):
    """
    有偏样本生成效果三阶段可视化 (共享 t-SNE 坐标系)

      左: 微调前测试集 (共享坐标)
      中: 训练集已知类 + 生成的新特征 (独立 t-SNE)
      右: 微调后测试集 (共享坐标, 可直接与左对比)
    """
    fig, axes = plt.subplots(1, 3, figsize=(33, 10), dpi=150)
    rng = np.random.default_rng(seed)

    # ================================================================
    # 左 + 右: 共享 t-SNE (前后特征拼接, 跑一次)
    # ================================================================
    origs_before_arr = np.array(origs_before)
    origs_after_arr = np.array(origs_after)

    # 按原始类别采样 (未知类的每个子类也单独采样, 保证数量充足)
    def _stratified_sample(feats_t, labels_t, origs_arr, max_n, rng_inst):
        """按 orig_label 分层采样, 未知子类也各自采样"""
        feats_np = feats_t.numpy() if isinstance(feats_t, torch.Tensor) else feats_t
        picked = []
        for orig_id in sorted(set(origs_arr)):
            idx = np.where(origs_arr == orig_id)[0]
            take = min(max_n, len(idx))
            picked.extend(rng_inst.choice(idx, size=take, replace=False).tolist())
        return np.array(picked)

    idx_before = _stratified_sample(feats_before, labels_before, origs_before_arr, max_per_class, rng)
    idx_after = _stratified_sample(feats_after, labels_after, origs_after_arr, max_per_class, rng)

    fb_np = feats_before.numpy() if isinstance(feats_before, torch.Tensor) else feats_before
    fa_np = feats_after.numpy() if isinstance(feats_after, torch.Tensor) else feats_after

    sub_before = fb_np[idx_before]
    sub_after = fa_np[idx_after]
    origs_sub_before = origs_before_arr[idx_before]
    origs_sub_after = origs_after_arr[idx_after]

    # 拼接跑一次 t-SNE
    combined = np.concatenate([sub_before, sub_after], axis=0)
    N_before = sub_before.shape[0]
    N_after = sub_after.shape[0]

    perp = min(40, max(5, combined.shape[0] // 5))
    print(f"[t-SNE] Shared embedding: {combined.shape[0]} points, perplexity={perp}")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                n_iter=2000, init="pca", early_exaggeration=12)
    Z_combined = tsne.fit_transform(combined)

    Z_before = Z_combined[:N_before]
    Z_after = Z_combined[N_before:]

    # 构建显示标签的映射 (未知类用具体子类名而非笼统 "unknown")
    UNKNOWN_NAMES = {5: "SN_9563*", 8: "ZIL131*", 9: "ZSU_23_4*"}

    def _get_display_name(orig_id):
        if orig_id in UNKNOWN_ORIG_IDS:
            return UNKNOWN_NAMES.get(orig_id, f"Unk_{orig_id}")
        return ID2NAME.get(orig_id, str(orig_id))

    # 颜色: 已知类用 tab10, 未知子类用灰色系
    UNKNOWN_SUB_COLORS = {5: "#555555", 8: "#888888", 9: "#AAAAAA"}

    def _get_color(orig_id, ci_counter):
        if orig_id in UNKNOWN_ORIG_IDS:
            return UNKNOWN_SUB_COLORS.get(orig_id, "#666666")
        return KNOWN_COLORS[ci_counter[0] % len(KNOWN_COLORS)]

    # --- 画左图 (Before) ---
    ax = axes[0]
    drawn_labels = set()
    ci_counter = [0]
    for orig_id in sorted(set(origs_sub_before)):
        mask = origs_sub_before == orig_id
        name = _get_display_name(orig_id)
        is_unk = orig_id in UNKNOWN_ORIG_IDS
        color = _get_color(orig_id, ci_counter)
        if not is_unk:
            ci_counter[0] += 1
        marker = "^" if is_unk else "o"
        size = 40 if is_unk else 25
        ax.scatter(Z_before[mask, 0], Z_before[mask, 1],
                   s=size, alpha=0.75, c=color, marker=marker,
                   label=name if name not in drawn_labels else "",
                   linewidths=0.3, edgecolors="white" if is_unk else "none")
        drawn_labels.add(name)

    title_l = "Before Finetune"
    if metrics_before:
        title_l += (f"\nIntra={metrics_before.get('intra_compactness',0):.4f}  "
                    f"Inter={metrics_before.get('inter_separation',0):.4f}  "
                    f"Silhouette={metrics_before.get('silhouette_approx',0):.3f}  "
                    f"Fisher={metrics_before.get('fisher_ratio',0):.1f}")
    ax.set_title(title_l, fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.3, fontsize=7, loc="best", frameon=True, ncol=2)

    # --- 画右图 (After) ---
    ax = axes[2]
    drawn_labels = set()
    ci_counter = [0]
    for orig_id in sorted(set(origs_sub_after)):
        mask = origs_sub_after == orig_id
        name = _get_display_name(orig_id)
        is_unk = orig_id in UNKNOWN_ORIG_IDS
        color = _get_color(orig_id, ci_counter)
        if not is_unk:
            ci_counter[0] += 1
        marker = "^" if is_unk else "o"
        size = 40 if is_unk else 25
        ax.scatter(Z_after[mask, 0], Z_after[mask, 1],
                   s=size, alpha=0.75, c=color, marker=marker,
                   label=name if name not in drawn_labels else "",
                   linewidths=0.3, edgecolors="white" if is_unk else "none")
        drawn_labels.add(name)

    title_r = "After Finetune"
    if metrics_after:
        title_r += (f"\nIntra={metrics_after.get('intra_compactness',0):.4f}  "
                    f"Inter={metrics_after.get('inter_separation',0):.4f}  "
                    f"Silhouette={metrics_after.get('silhouette_approx',0):.3f}  "
                    f"Fisher={metrics_after.get('fisher_ratio',0):.1f}")
    ax.set_title(title_r, fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.3, fontsize=7, loc="best", frameon=True, ncol=2)

    # ================================================================
    # 中图: 已知类训练特征 + 生成特征 (独立 t-SNE)
    # ================================================================
    ax = axes[1]
    _draw_tsne_with_generated(
        ax, feats_before, labels_before, origs_before,
        generated_feats,
        title="Training Features + Generated (starred)",
        max_per_class=max_per_class, seed=seed,
    )

    plt.suptitle("Biased Sample Generation & Finetune Effect (4.3)\n"
                 "Left & Right share the same t-SNE coordinates for direct comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def _draw_tsne_with_generated(ax, feats_orig, labels_orig, origs_orig,
                               generated_feats_dict,
                               title="", max_per_class=200, seed=42):
    """中图: 已知类训练特征 + 生成特征 (星号) 叠加"""
    feats_np = feats_orig.numpy() if isinstance(feats_orig, torch.Tensor) else feats_orig
    labels_np = labels_orig.numpy() if isinstance(labels_orig, torch.Tensor) else labels_orig

    rng = np.random.default_rng(seed)

    # 只取已知类
    known_mask = labels_np >= 0
    feats_known = feats_np[known_mask]
    labels_known = labels_np[known_mask]

    # 采样
    picked = []
    for c in np.unique(labels_known):
        idx = np.where(labels_known == c)[0]
        take = min(max_per_class, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    picked = np.array(picked)
    orig_sub = feats_known[picked]
    orig_labels_sub = labels_known[picked]

    # 准备生成特征
    gen_feats_list, gen_labels_list = [], []
    for cid, gf in generated_feats_dict.items():
        gf_np = gf.numpy() if isinstance(gf, torch.Tensor) else gf
        if gf_np.shape[0] > 0:
            n_take = min(60, gf_np.shape[0])
            idx_g = rng.choice(gf_np.shape[0], size=n_take, replace=False)
            gen_feats_list.append(gf_np[idx_g])
            gen_labels_list.append(np.full(n_take, cid))

    if gen_feats_list:
        gen_np = np.concatenate(gen_feats_list, axis=0)
        gen_labels_np = np.concatenate(gen_labels_list)
    else:
        gen_np = np.empty((0, orig_sub.shape[1]))
        gen_labels_np = np.empty(0, dtype=int)

    # 合并跑 t-SNE
    all_feats = np.concatenate([orig_sub, gen_np], axis=0) if gen_np.shape[0] > 0 else orig_sub
    n_orig = orig_sub.shape[0]
    n_gen = gen_np.shape[0]

    perp = min(35, max(5, all_feats.shape[0] // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, n_iter=1500, init="pca")
    Z = tsne.fit_transform(all_feats)
    Z_orig = Z[:n_orig]
    Z_gen = Z[n_orig:]

    # 画原始
    unique_c = sorted(np.unique(orig_labels_sub))
    ci = 0
    for c in unique_c:
        color = KNOWN_COLORS[ci % len(KNOWN_COLORS)]
        name = ID2NAME.get(c, f"C{c}")
        mask = orig_labels_sub == c
        ax.scatter(Z_orig[mask, 0], Z_orig[mask, 1],
                   s=18, alpha=0.5, c=color, marker="o", linewidths=0,
                   label=name)
        ci += 1

    # 画生成 (星号)
    if n_gen > 0:
        for c in sorted(np.unique(gen_labels_np)):
            # 复用该类的颜色
            color = "gray"
            for j, uc in enumerate(unique_c):
                if uc == c:
                    color = KNOWN_COLORS[j % len(KNOWN_COLORS)]
                    break
            mask = gen_labels_np == c
            name = ID2NAME.get(c, f"C{c}")
            ax.scatter(Z_gen[mask, 0], Z_gen[mask, 1],
                       s=90, alpha=0.9, c=color, marker="*",
                       edgecolors="black", linewidths=0.5,
                       label=f"{name} (gen)")

    ax.set_title(f"{title}\n({n_gen} generated samples)", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.0, fontsize=6, loc="best", frameon=True, ncol=2)


# =============================================================================
# 9. 全流水线三阶段可视化: 训练前 → 训练后(+原型) → 增广后(+增广★)
# =============================================================================
def plot_pipeline_three_stage(feats_pretrain, labels_pretrain, origs_pretrain, metrics_pretrain,
                               feats_trained, labels_trained, origs_trained, metrics_trained,
                               feats_final, labels_final, origs_final, metrics_final,
                               save_path, max_per_class=300, seed=42,
                               prototypes=None,
                               augmented_feats=None):
    """
    全流水线三阶段可视化

      左:  训练前 (随机初始化)
      中:  训练后 (SupCon + CosFace) + 类原型 ★
      右:  增广+微调后 + 增广样本 ★ (点更多)
      未知类统一黑色三角
    """
    rng = np.random.default_rng(seed)

    def _tsne_with_extra(feats, origs, extra_feats=None):
        """t-SNE, 可附加 extra_feats 一起投影"""
        feats_np = feats.numpy() if isinstance(feats, torch.Tensor) else feats
        origs_arr = np.array(origs)
        picked = []
        for oid in sorted(set(origs_arr)):
            idx = np.where(origs_arr == oid)[0]
            take = min(max_per_class, len(idx))
            picked.extend(rng.choice(idx, size=take, replace=False).tolist())
        picked = np.array(picked)
        sub = feats_np[picked]
        origs_sub = origs_arr[picked]

        # 拼接额外特征
        n_main = sub.shape[0]
        if extra_feats is not None and extra_feats.shape[0] > 0:
            combined = np.concatenate([sub, extra_feats], axis=0)
        else:
            combined = sub

        perp = min(40, max(5, combined.shape[0] // 5))
        tsne_inst = TSNE(n_components=2, perplexity=perp, random_state=seed,
                         n_iter=2000, init="pca", early_exaggeration=12)
        Z = tsne_inst.fit_transform(combined)
        Z_main = Z[:n_main]
        Z_extra = Z[n_main:] if extra_feats is not None and extra_feats.shape[0] > 0 else None
        return Z_main, origs_sub, Z_extra

    # 构建颜色映射 (全局一致)
    all_known_oids = sorted(set(o for o in list(set(origs_pretrain)) + list(set(origs_trained))
                                + list(set(origs_final)) if o not in UNKNOWN_ORIG_IDS))
    oid2color = {}
    for ci, oid in enumerate(all_known_oids):
        oid2color[oid] = KNOWN_COLORS[ci % len(KNOWN_COLORS)]

    # --- 准备原型 (中图) ---
    protos_np = None
    if prototypes is not None:
        protos_np = prototypes.numpy() if isinstance(prototypes, torch.Tensor) else prototypes

    # --- 准备增广特征 (中图, 与训练后特征同属一个特征空间) ---
    aug_np = None
    aug_labels_np = None
    n_aug = 0
    if augmented_feats:
        aug_list, aug_lab_list = [], []
        for cid, gf in augmented_feats.items():
            gf_np = gf.numpy() if isinstance(gf, torch.Tensor) else gf
            if gf_np.shape[0] > 0:
                n_take = min(60, gf_np.shape[0])
                idx_g = rng.choice(gf_np.shape[0], size=n_take, replace=False)
                aug_list.append(gf_np[idx_g])
                aug_lab_list.append(np.full(n_take, cid))
        if aug_list:
            aug_np = np.concatenate(aug_list, axis=0)
            aug_labels_np = np.concatenate(aug_lab_list)
            n_aug = aug_np.shape[0]

    # --- 左图: 训练前 (独立 t-SNE) ---
    Z_pre, origs_pre, _ = _tsne_with_extra(feats_pretrain, origs_pretrain)

    # --- 中图: 训练后 + 原型 + 增广★ (增广特征是此模型生成的, 属于此空间) ---
    extra_mid = []
    if protos_np is not None:
        extra_mid.append(protos_np)
    n_protos_in_mid = protos_np.shape[0] if protos_np is not None else 0
    if aug_np is not None:
        extra_mid.append(aug_np)
    extra_mid_np = np.concatenate(extra_mid, axis=0) if extra_mid else None

    Z_train, origs_train, Z_extra_mid = _tsne_with_extra(
        feats_trained, origs_trained, extra_feats=extra_mid_np
    )
    # 拆分原型和增广
    Z_protos = None
    Z_aug_mid = None
    if Z_extra_mid is not None:
        Z_protos = Z_extra_mid[:n_protos_in_mid] if n_protos_in_mid > 0 else None
        Z_aug_mid = Z_extra_mid[n_protos_in_mid:] if n_aug > 0 else None

    # --- 右图: 增广+微调后 (独立 t-SNE, 干净展示微调结果) ---
    Z_final, origs_final_sub, _ = _tsne_with_extra(
        feats_final, origs_final, extra_feats=None
    )

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), dpi=150)

    # === 通用绘制函数 ===
    def _draw_scatter(ax, Z, origs_sub):
        for oid in sorted(set(origs_sub)):
            if oid in UNKNOWN_ORIG_IDS:
                continue
            mask = origs_sub == oid
            name = ID2NAME.get(oid, str(oid))
            color = oid2color.get(oid, "#888888")
            ax.scatter(Z[mask, 0], Z[mask, 1], s=25, alpha=0.75, c=color,
                       marker="o", label=name, linewidths=0)
        unk_mask = np.isin(origs_sub, list(UNKNOWN_ORIG_IDS))
        if unk_mask.any():
            ax.scatter(Z[unk_mask, 0], Z[unk_mask, 1], s=30, alpha=0.6,
                       c="black", marker="^", label="Unknown", linewidths=0.3,
                       edgecolors="gray")

    # --- (a) 左图: 训练前 ---
    ax = axes[0]
    _draw_scatter(ax, Z_pre, origs_pre)
    fisher_pre = metrics_pretrain.get("fisher_ratio", 0) if metrics_pretrain else 0
    ax.set_title(f"(a) Before Training\nFisher={fisher_pre:.1f}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.5, fontsize=7, loc="best", frameon=True, ncol=2)

    # --- (b) 中图: 训练后 + 原型★ + 增广★ ---
    ax = axes[1]
    _draw_scatter(ax, Z_train, origs_train)
    if Z_protos is not None:
        ax.scatter(Z_protos[:, 0], Z_protos[:, 1],
                   s=250, marker="*", c="red", edgecolors="black",
                   linewidths=1.5, zorder=10, label="Prototypes")
    if Z_aug_mid is not None and aug_labels_np is not None:
        for cid in sorted(np.unique(aug_labels_np)):
            mask_g = aug_labels_np == cid
            orig_id = KNOWN_ORIG_IDS[cid] if cid < len(KNOWN_ORIG_IDS) else cid
            color = oid2color.get(orig_id, "#888888")
            name_g = ID2NAME.get(orig_id, f"C{cid}")
            ax.scatter(Z_aug_mid[mask_g, 0], Z_aug_mid[mask_g, 1],
                       s=100, alpha=0.85, c=color, marker="*",
                       edgecolors="black", linewidths=0.5,
                       label=f"{name_g} (aug)", zorder=5)
    fisher_train = metrics_trained.get("fisher_ratio", 0) if metrics_trained else 0
    title_b = f"(b) After Training + Augmented Data\nFisher={fisher_train:.1f}"
    if n_aug > 0:
        title_b += f"  ★={n_aug} augmented"
    ax.set_title(title_b, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.2, fontsize=6, loc="best", frameon=True, ncol=2)

    # --- (c) 右图: 增广+微调后 (干净展示优化结果) ---
    ax = axes[2]
    _draw_scatter(ax, Z_final, origs_final_sub)
    fisher_final = metrics_final.get("fisher_ratio", 0) if metrics_final else 0
    ax.set_title(f"(c) After Augmentation+Finetune\nFisher={fisher_final:.1f}",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=1.5, fontsize=7, loc="best", frameon=True, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")

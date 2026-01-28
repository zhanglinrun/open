"""
原型计算模块
包含修正原型学习(论文公式4-6)、简单平均原型、按类阈值计算
"""

import numpy as np
import torch
import torch.nn.functional as F


# -------------------------
# 修正原型学习(论文公式4-6)
# -------------------------
@torch.no_grad()
def compute_rectified_prototypes(model: torch.nn.Module,
                               loader: torch.utils.data.DataLoader,
                               device: torch.device,
                               num_known: int,
                               use_pos_branch: bool = True) -> torch.Tensor:
    """
    实现论文公式4-6的修正原型计算
    基于样本典型性(typicality)加权,减少异常样本影响

    Args:
        model: 模型
        loader: 数据加载器(仅已知类样本)
        device: 设备
        num_known: 已知类数量
        use_pos_branch: 是否使用pos分支特征,否则用neg分支

    Returns:
        修正后的原型 [K, D] (已L2归一化)
    """
    model.eval()

    # 1. 收集所有已知类样本的特征
    all_feats = [[] for _ in range(num_known)]
    embed_dim = None

    for x, y, _, _ in loader:
        mask = (y >= 0)  # 只看已知类
        if not mask.any():
            continue

        x = x[mask].to(device)
        y = y[mask].to(device)

        # 提取特征
        _, _, f_neg, f_pos = model(x, None)

        # 选择使用哪个分支
        feats = f_pos if use_pos_branch else f_neg
        feats = F.normalize(feats, dim=1)  # 归一化 [B', D]

        if embed_dim is None:
            embed_dim = feats.size(1)

        # 按类别收集特征
        for i, label in enumerate(y):
            all_feats[label.item()].append(feats[i].cpu())  # 暂存到CPU省显存

    # 2. 计算修正原型
    rectified_protos = torch.zeros(num_known, embed_dim, device=device)

    for c in range(num_known):
        if len(all_feats[c]) == 0:
            continue  # 该类没有样本,保持零向量

        if len(all_feats[c]) == 1:
            # 只有1个样本,直接作为原型
            rectified_protos[c] = all_feats[c][0].to(device)
            continue

        # 拿到该类所有样本特征 [N, D]
        feats_c = torch.stack(all_feats[c]).to(device)
        N = feats_c.size(0)

        # --- 论文公式4: 计算两两距离矩阵 ---
        # 技巧: ||x-y||^2 = 2 - 2(x·y) (对于归一化向量)
        sim_matrix = torch.matmul(feats_c, feats_c.T)  # [N, N]
        # clamp防止数值误差导致负数
        dist_matrix = torch.sqrt((2.0 - 2.0 * sim_matrix).clamp(min=0.0))

        # SA_i: 第i个样本到其他样本的平均距离 (不合群程度)
        SA = dist_matrix.mean(dim=1)  # [N]

        # --- 论文公式5: 计算权重 ---
        # 距离SA越小,exp(-SA)越大,权重越大
        # 使用softmax自动归一化权重 (Softmax(-SA) 等价于公式5)
        w = F.softmax(-SA, dim=0)  # [N]

        # --- 论文公式6: 加权求和 ---
        # [N, 1] * [N, D] -> sum -> [D]
        mu_c = (w.unsqueeze(1) * feats_c).sum(dim=0)

        # L2归一化
        rectified_protos[c] = F.normalize(mu_c, dim=0)

    return rectified_protos


# -------------------------
# 简单平均原型
# -------------------------
@torch.no_grad()
def compute_simple_prototypes(model: torch.nn.Module,
                           loader: torch.utils.data.DataLoader,
                           device: torch.device,
                           num_known: int,
                           use_pos_branch: bool = True) -> torch.Tensor:
    """
    简单平均原型计算(作为对比基线)

    Args:
        model: 模型
        loader: 数据加载器(仅已知类样本)
        device: 设备
        num_known: 已知类数量
        use_pos_branch: 是否使用pos分支特征

    Returns:
        原型 [K, D] (已L2归一化)
    """
    model.eval()

    feats_sum = None
    counts = torch.zeros(num_known, dtype=torch.long, device=device)

    for x, y, _, _ in loader:
        mask = (y >= 0)
        if not mask.any():
            continue

        x = x[mask].to(device)
        y = y[mask].to(device)

        _, _, f_neg, f_pos = model(x, None)
        feats = f_pos if use_pos_branch else f_neg
        feats = F.normalize(feats, dim=1)

        if feats_sum is None:
            embed_dim = feats.size(1)
            feats_sum = torch.zeros(num_known, embed_dim, device=device)

        # 累加
        for c in range(num_known):
            m = (y == c)
            if m.any():
                feats_sum[c] += feats[m].sum(dim=0)
                counts[c] += int(m.sum().item())

    # 计算平均
    protos = torch.zeros(num_known, feats_sum.shape[1], device=device)
    for c in range(num_known):
        if counts[c] > 0:
            protos[c] = feats_sum[c] / counts[c]
            protos[c] = F.normalize(protos[c], dim=0)

    return protos


# -------------------------
# 按类阈值计算
# -------------------------
@torch.no_grad()
def compute_class_thresholds_p95(model: torch.nn.Module,
                               loader: torch.utils.data.DataLoader,
                               device: torch.device,
                               protos: torch.Tensor,
                               num_known: int,
                               percentile: float = 95.0,
                               use_pos_branch: bool = True) -> np.ndarray:
    """
    对每个已知类c计算阈值tau_c
    使用训练集的第95分位数,防止测试集数据泄露

    Args:
        model: 模型
        loader: 数据加载器(仅已知类样本)
        device: 设备
        protos: 类原型 [K, D]
        num_known: 已知类数量
        percentile: 分位数
        use_pos_branch: 是否使用pos分支特征

    Returns:
        每类的阈值 tau_c [K]
    """
    model.eval()
    dist_lists = [[] for _ in range(num_known)]

    for x, y, _, _ in loader:
        mask = (y >= 0)
        if not mask.any():
            continue

        x = x[mask].to(device)
        y = y[mask].to(device)

        _, _, f_neg, f_pos = model(x, None)
        feats = f_pos if use_pos_branch else f_neg
        feats = F.normalize(feats, dim=1)  # [B, D]

        # 取对应类原型的相似度: sim_i = <f_i, proto_{y_i}>
        sim = torch.sum(feats * protos[y], dim=1)  # [B]
        dist = (1.0 - sim).detach().cpu().numpy()

        y_np = y.detach().cpu().numpy()
        for i in range(len(y_np)):
            dist_lists[y_np[i]].append(dist[i])

    # 计算每类的阈值
    thresholds = []
    print("\n[Threshold Stats] Per-class thresholds:")
    for k in range(num_known):
        if len(dist_lists[k]) > 0:
            dists = np.array(dist_lists[k], dtype=np.float32)
            tau = np.percentile(dists, percentile)
            thresholds.append(tau)
            print(f"  Class {k}: tau={tau:.4f} (min={dists.min():.4f}, max={dists.max():.4f}, mean={dists.mean():.4f}, n={len(dists)})")
        else:
            thresholds.append(0.5)  # 默认值
            print(f"  Class {k}: tau=0.5000 (No samples)")
            
    return np.array(thresholds)


@torch.no_grad()
def compute_global_threshold_p95(model: torch.nn.Module,
                               loader: torch.utils.data.DataLoader,
                               device: torch.device,
                               protos: torch.Tensor,
                               percentile: float = 95.0,
                               use_pos_branch: bool = True) -> float:
    """
    计算全局阈值tau
    使用训练集的第95分位数

    Args:
        model: 模型
        loader: 数据加载器(仅已知类样本)
        device: 设备
        protos: 类原型 [K, D]
        percentile: 分位数
        use_pos_branch: 是否使用pos分支特征

    Returns:
        全局阈值 tau (float)
    """
    model.eval()
    all_dists = []

    for x, y, _, _ in loader:
        mask = (y >= 0)
        if not mask.any():
            continue

        x = x[mask].to(device)
        y = y[mask].to(device)

        _, _, f_neg, f_pos = model(x, None)
        feats = f_pos if use_pos_branch else f_neg
        feats = F.normalize(feats, dim=1)

        # 找到最近的原型
        sims = torch.matmul(feats, protos.T)  # [B, K]
        max_sim, _ = torch.max(sims, dim=1)  # [B]
        dist = (1.0 - max_sim).detach().cpu().numpy()

        all_dists.extend(dist)

    # 计算全局阈值
    dists = np.array(all_dists, dtype=np.float32)
    tau = np.percentile(dists, percentile)
    
    print(f"\n[Threshold Stats] Global threshold (p{percentile}):")
    print(f"  tau={tau:.4f} (min={dists.min():.4f}, max={dists.max():.4f}, mean={dists.mean():.4f}, n={len(dists)})")

    return tau


# -------------------------
# 测试代码
# -------------------------
if __name__ == "__main__":
    # 测试原型计算
    print("原型计算模块测试")

    # 这里需要真实的模型和数据加载器才能测试
    # 只是打印函数签名
    print("  - compute_rectified_prototypes(): 实现论文公式4-6")
    print("  - compute_simple_prototypes(): 简单平均")
    print("  - compute_class_thresholds_p95(): 按类阈值")
    print("  - compute_global_threshold_p95(): 全局阈值")

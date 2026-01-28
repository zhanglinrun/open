
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Tuple

# -------------------------
# 距离收集
# -------------------------
@torch.no_grad()
def collect_distances(model: torch.nn.Module,
                     loader: torch.utils.data.DataLoader,
                     device: torch.device,
                     protos: torch.Tensor,
                     use_pos_branch: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    收集所有样本到最近原型的距离，以及是否分类正确（针对已知类）
    """
    model.eval()
    dists = []
    is_unknown = []
    is_correct = [] # 1 if correctly classified (for known), 0 if wrong. For unknown, 0 (irrelevant)

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        # 提取特征
        _, _, f_neg, f_pos = model(x, None)
        feats = f_pos if use_pos_branch else f_neg
        feats = F.normalize(feats, dim=1)  # [B, D]

        # 计算到所有原型的余弦相似度
        sims = torch.matmul(feats, protos.T)  # [B, K]
        max_sim, pred_cls = torch.max(sims, dim=1)  # [B]

        # 距离 = 1 - 相似度
        dist = 1.0 - max_sim

        dists.extend(dist.detach().cpu().tolist())
        is_unknown_batch = (y < 0).detach().cpu().int()
        is_unknown.extend(is_unknown_batch.tolist())
        
        # Check classification correctness for known samples
        # For unknown samples (y < 0), correctness is not defined in closed set sense, set to 0
        correct_mask = (pred_cls == y) & (y >= 0)
        is_correct.extend(correct_mask.detach().cpu().int().tolist())

    return (np.array(dists, dtype=np.float32), 
            np.array(is_unknown, dtype=np.int32),
            np.array(is_correct, dtype=np.int32))


# -------------------------
# 阈值搜索
# -------------------------
def find_best_tau_by_f1(dists: np.ndarray,
                        is_unknown: np.ndarray,
                        is_correct: np.ndarray = None,
                        steps: int = 201) -> Dict:
    """
    在测试集上搜索最优tau (基于F1分数)
    """
    best = {
        "tau": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "aks": 0.0, "aus": 0.0, "na": 0.0
    }

    lo = float(np.min(dists))
    hi = float(np.max(dists))

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return best
        
    total_known = (is_unknown == 0).sum()
    total_unknown = (is_unknown == 1).sum()

    # 搜索最优tau
    for tau in np.linspace(lo, hi, steps):
        pred_unknown = (dists > tau).astype(np.int32)  # 1表示被判定为未知(拒绝)

        tp = int(((pred_unknown == 1) & (is_unknown == 1)).sum())
        fp = int(((pred_unknown == 1) & (is_unknown == 0)).sum())
        fn = int(((pred_unknown == 0) & (is_unknown == 1)).sum())
        tn = int(((pred_unknown == 0) & (is_unknown == 0)).sum())

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        if f1 > best["f1"]:
            # AKS: 已知类样本中，(未被拒绝) 且 (分类正确) 的比例
            # Pred=0 (Accepted) AND is_correct=1
            if is_correct is not None:
                correct_accepted = int(((pred_unknown == 0) & (is_correct == 1)).sum())
                aks = correct_accepted / (total_known + 1e-12)
            else:
                # Fallback if is_correct not provided (assume all accepted are correct - inaccurate)
                aks = tn / (total_known + 1e-12)

            # AUS: 未知类样本中，被正确拒绝的比例
            aus = tp / (total_unknown + 1e-12)
            
            # NA: Normalized Accuracy
            na = (aks + aus) / 2.0

            best.update({
                "tau": float(tau),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "aks": float(aks),
                "aus": float(aus),
                "na": float(na),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            })

    return best


# -------------------------
# 开集评估
# -------------------------
@torch.no_grad()
def evaluate_with_proto(model: torch.nn.Module,
                      loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      protos: torch.Tensor,
                      tau: float,
                      use_pos_branch: bool = True) -> Dict:
    """
    基于原型距离的开集评估
    """
    model.eval()
    total_known = 0
    correct_known = 0
    tp = fp = fn = tn = 0
    
    all_dists = []
    all_is_unknown = []
    
    # Track correct classification for AKS
    correct_accepted_known = 0

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        # 提取特征
        _, _, f_neg, f_pos = model(x, None)
        feats = f_pos if use_pos_branch else f_neg
        feats = F.normalize(feats, dim=1)

        # 计算到所有原型的相似度
        sims = torch.matmul(feats, protos.T)  # [B, K]
        max_sim, pred_cls = torch.max(sims, dim=1)  # [B]

        # 距离
        min_dist = 1.0 - max_sim
        
        # 收集距离信息用于AUROC
        all_dists.extend(min_dist.detach().cpu().tolist())
        all_is_unknown.extend((y < 0).detach().cpu().int().tolist())

        # 预测未知
        pred_unknown = (min_dist > tau) # Boolean: True if rejected (unknown)

        # 开集混淆矩阵
        is_unk = (y < 0)
        tp += int(((pred_unknown == True) & (is_unk == True)).sum().item())
        fp += int(((pred_unknown == True) & (is_unk == False)).sum().item())
        fn += int(((pred_unknown == False) & (is_unk == True)).sum().item())
        tn += int(((pred_unknown == False) & (is_unk == False)).sum().item())

        # 已知类统计
        known_mask = (y >= 0)
        if known_mask.any():
            total_known += int(known_mask.sum().item())
            # AKS calculation: Must be Known AND Not Rejected AND Correctly Classified
            # Not Rejected: pred_unknown == False
            # Correctly Classified: pred_cls == y
            
            accepted_mask = (pred_unknown == False)
            correct_cls_mask = (pred_cls == y)
            
            correct_accepted_known += int((known_mask & accepted_mask & correct_cls_mask).sum().item())
            
            # Pure Closed Set Accuracy (ignoring rejection)
            correct_known += int((correct_cls_mask & known_mask).sum().item())

    # 计算指标
    known_acc = correct_known / max(total_known, 1)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    open_acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    # AKS: 已知类样本被正确分类且未被拒判 / 总已知类样本
    if total_known > 0:
        aks = correct_accepted_known / total_known
    else:
        aks = 0.0

    # AUS: 未知类样本被正确拒判 (Recall of Unknown)
    # Total Unknown = TP + FN
    total_unknown = tp + fn
    if total_unknown > 0:
        aus = tp / total_unknown
    else:
        aus = 0.0
    
    na = (aks + aus) / 2.0
    
    # AUROC
    auroc = compute_auroc(np.array(all_dists), np.array(all_is_unknown))

    return {
        "known_acc": float(known_acc),
        "open_precision": float(precision),
        "open_recall": float(recall),
        "open_f1": float(f1),
        "open_acc": float(open_acc),
        "aks": float(aks),
        "aus": float(aus),
        "na": float(na),
        "auroc": float(auroc),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


@torch.no_grad()
def evaluate_with_proto_class_tau(model: torch.nn.Module,
                                 loader: torch.utils.data.DataLoader,
                                 device: torch.device,
                                 protos: torch.Tensor,
                                 tau_c: np.ndarray,
                                 use_pos_branch: bool = True) -> Dict:
    """
    使用按类阈值的开集评估
    """
    model.eval()
    total_known = 0
    correct_known = 0
    tp = fp = fn = tn = 0
    
    all_dists = []
    all_is_unknown = []
    
    correct_accepted_known = 0

    tau_t = torch.from_numpy(tau_c).to(device)  # [K]

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)

        # 提取特征
        _, _, f_neg, f_pos = model(x, None)
        feats = f_pos if use_pos_branch else f_neg
        feats = F.normalize(feats, dim=1)

        # 计算到所有原型的相似度
        sims = torch.matmul(feats, protos.T)  # [B, K]
        max_sim, pred_cls = torch.max(sims, dim=1)  # [B]

        # 距离
        dist = 1.0 - max_sim
        
        # 收集距离
        all_dists.extend(dist.detach().cpu().tolist())
        all_is_unknown.extend((y < 0).detach().cpu().int().tolist())

        # 每个样本用自己预测类别对应的阈值
        thr = tau_t[pred_cls]  # [B]
        pred_unknown = (dist > thr)

        # 开集混淆矩阵
        is_unk = (y < 0)
        tp += int(((pred_unknown == True) & (is_unk == True)).sum().item())
        fp += int(((pred_unknown == True) & (is_unk == False)).sum().item())
        fn += int(((pred_unknown == False) & (is_unk == True)).sum().item())
        tn += int(((pred_unknown == False) & (is_unk == False)).sum().item())

        # 已知类统计
        known_mask = (y >= 0)
        if known_mask.any():
            total_known += int(known_mask.sum().item())
            
            accepted_mask = (pred_unknown == False)
            correct_cls_mask = (pred_cls == y)
            
            correct_accepted_known += int((known_mask & accepted_mask & correct_cls_mask).sum().item())
            correct_known += int((correct_cls_mask & known_mask).sum().item())

    # 计算指标
    known_acc = correct_known / max(total_known, 1)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    open_acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    # AKS
    if total_known > 0:
        aks = correct_accepted_known / total_known
    else:
        aks = 0.0

    # AUS
    total_unknown = tp + fn
    if total_unknown > 0:
        aus = tp / total_unknown
    else:
        aus = 0.0
    
    na = (aks + aus) / 2.0
    
    # AUROC
    auroc = compute_auroc(np.array(all_dists), np.array(all_is_unknown))

    return {
        "known_acc": float(known_acc),
        "open_precision": float(precision),
        "open_recall": float(recall),
        "open_f1": float(f1),
        "open_acc": float(open_acc),
        "aks": float(aks),
        "aus": float(aus),
        "na": float(na),
        "auroc": float(auroc),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


# -------------------------
# AUROC和OSCR计算
# -------------------------
def compute_auroc(dists: np.ndarray,
                 is_unknown: np.ndarray) -> float:
    """
    计算AUROC (Area Under ROC Curve)
    """
    try:
        # 距离越大,越可能是未知类
        auroc = roc_auc_score(is_unknown, dists)
        return float(auroc)
    except ValueError:
        return 0.5


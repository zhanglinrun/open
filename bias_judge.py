"""
库内有偏/无偏判定标准模块 (4.1 核心) - 基于余弦距离

判定原理:
  计算每个样本与其所属类原型的余弦距离 d = 1 - cos(f_x, μ_c),
  距离越大说明样本偏离类中心越远, 即为"有偏样本".
  阈值由训练集上所有样本距离的 percentile 分位数确定.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from feature_library import FeatureLibrary


@dataclass
class BiasResult:
    is_biased: bool
    distance: float          # 余弦距离 = 1 - cos_sim
    distance_threshold: float
    predicted_class: int
    confidence: float        # 余弦相似度 = 1 - distance
    bias_score: float = 0.0  # distance / threshold (>1 则有偏)


@dataclass
class ClassBiasStats:
    class_id: int
    class_name: str
    total: int = 0
    biased: int = 0
    unbiased: int = 0
    bias_ratio: float = 0.0
    mean_distance: float = 0.0
    distance_threshold: float = 0.0


class BiasJudge:
    def __init__(self, feature_library: FeatureLibrary,
                 percentile: float = 95.0, **kwargs):
        self.library = feature_library
        self.percentile = percentile
        self.distance_threshold: float = 0.0
        self._fitted = False

    def fit(self, verbose: bool = True):
        """根据训练集特征拟合距离阈值"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"[BiasJudge] Fitting distance threshold (p={self.percentile})")
            print(f"{'='*60}")

        prototypes = self.library.get_prototypes()
        proto_norm = F.normalize(prototypes, dim=1)
        all_distances = []

        for idx, cid in enumerate(self.library.class_ids):
            entry = self.library.entries[cid]
            entry.compute_stats()
            if entry.count == 0:
                continue
            feats = torch.stack(entry.features)
            feats_norm = F.normalize(feats, dim=1)
            # 每个样本到自身类原型的余弦相似度
            cos_sim = torch.matmul(feats_norm, proto_norm[idx:idx+1].T).squeeze(1)
            dist = 1.0 - cos_sim  # 余弦距离
            all_distances.extend(dist.numpy().tolist())
            if verbose:
                print(f"  Class {cid} ({entry.class_name}): "
                      f"mean_dist={dist.mean():.4f}, std={dist.std():.4f}, "
                      f"max={dist.max():.4f}, n={entry.count}")

        self.distance_threshold = float(np.percentile(all_distances, self.percentile))
        self._fitted = True
        if verbose:
            print(f"  Distance threshold (p{self.percentile}): {self.distance_threshold:.4f}")
            print(f"[BiasJudge] Fitting complete.\n")

    def judge(self, fingerprint: torch.Tensor,
              predicted_class: Optional[int] = None) -> BiasResult:
        """判定单个样本是否有偏"""
        if not self._fitted:
            raise RuntimeError("BiasJudge not fitted. Call fit() first.")
        if fingerprint.dim() == 2:
            fingerprint = fingerprint.squeeze(0)

        fp_norm = F.normalize(fingerprint.unsqueeze(0), dim=1)
        prototypes = self.library.get_prototypes()
        proto_norm = F.normalize(prototypes, dim=1)
        # 与所有原型的余弦相似度
        cos_sims = torch.matmul(fp_norm, proto_norm.T).squeeze(0)

        if predicted_class is None:
            pred_idx = cos_sims.argmax().item()
            predicted_class = self.library.class_ids[pred_idx]

        # 找到预测类在 class_ids 中的索引
        try:
            class_idx = self.library.class_ids.index(predicted_class)
        except ValueError:
            class_idx = cos_sims.argmax().item()

        sim = float(cos_sims[class_idx].item())
        distance = 1.0 - sim
        is_biased = distance > self.distance_threshold

        return BiasResult(
            is_biased=is_biased, distance=distance,
            distance_threshold=self.distance_threshold,
            predicted_class=predicted_class, confidence=sim,
            bias_score=distance / max(self.distance_threshold, 1e-6),
        )

    @torch.no_grad()
    def judge_batch(self, model, loader, device):
        """批量判定"""
        model.eval()
        results = []
        class_stats = {}
        for cid in self.library.class_ids:
            e = self.library.entries[cid]
            class_stats[cid] = ClassBiasStats(class_id=cid, class_name=e.class_name)

        for x, y, orig_labels, paths in loader:
            mask = (y >= 0)
            if not mask.any():
                continue
            fps = model.extract_fingerprint(x[mask].to(device)).detach().cpu()
            y_k = y[mask]
            for i in range(fps.size(0)):
                cid = int(y_k[i].item())
                r = self.judge(fps[i], predicted_class=cid)
                results.append(r)
                if cid in class_stats:
                    class_stats[cid].total += 1
                    if r.is_biased:
                        class_stats[cid].biased += 1
                    else:
                        class_stats[cid].unbiased += 1

        for cid, s in class_stats.items():
            if s.total > 0:
                s.bias_ratio = s.biased / s.total
                dists = [r.distance for r in results if r.predicted_class == cid]
                if dists:
                    s.mean_distance = float(np.mean(dists))
            s.distance_threshold = self.distance_threshold
        return results, class_stats

    def report(self, class_stats):
        """生成可读报告"""
        lines = [
            f"\n{'='*70}",
            f"  库内有偏/无偏判定报告 (基于余弦距离)",
            f"{'='*70}",
            f"  percentile={self.percentile}",
            f"  distance_threshold={self.distance_threshold:.4f}",
            f"{'='*70}",
            f"  {'类别':<12} {'总数':>6} {'无偏':>6} {'有偏':>6} {'偏比':>8} "
            f"{'均距':>8} {'阈值':>8}",
            f"  {'-'*60}",
        ]
        total_all = biased_all = 0
        for cid in sorted(class_stats.keys()):
            s = class_stats[cid]
            total_all += s.total
            biased_all += s.biased
            lines.append(
                f"  {s.class_name:<12} {s.total:>6} {s.unbiased:>6} {s.biased:>6} "
                f"{s.bias_ratio:>7.1%} {s.mean_distance:>8.4f} "
                f"{s.distance_threshold:>8.4f}")
        ratio = biased_all / max(total_all, 1)
        lines.append(f"  {'-'*60}")
        lines.append(f"  {'总计':<12} {total_all:>6} {total_all-biased_all:>6} "
                     f"{biased_all:>6} {ratio:>7.1%}")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    @torch.no_grad()
    def split_biased_unbiased(self, model, loader, device):
        """分离有偏和无偏样本"""
        model.eval()
        ub = {cid: [] for cid in self.library.class_ids}
        bf = {cid: [] for cid in self.library.class_ids}
        bp = {cid: [] for cid in self.library.class_ids}
        cs = {}
        for cid in self.library.class_ids:
            e = self.library.entries[cid]
            cs[cid] = ClassBiasStats(class_id=cid, class_name=e.class_name)

        for x, y, orig_labels, paths in loader:
            mask = (y >= 0)
            if not mask.any():
                continue
            fps = model.extract_fingerprint(x[mask].to(device)).detach().cpu()
            y_k = y[mask]
            paths_k = [paths[i] for i in range(len(paths)) if mask[i]]
            for i in range(fps.size(0)):
                cid = int(y_k[i].item())
                if cid not in self.library.entries:
                    continue
                r = self.judge(fps[i], predicted_class=cid)
                cs[cid].total += 1
                if r.is_biased:
                    bf[cid].append(fps[i])
                    bp[cid].append(paths_k[i])
                    cs[cid].biased += 1
                else:
                    ub[cid].append(fps[i])
                    cs[cid].unbiased += 1

        for cid, s in cs.items():
            if s.total > 0:
                s.bias_ratio = s.biased / s.total
        return ub, bf, bp, cs

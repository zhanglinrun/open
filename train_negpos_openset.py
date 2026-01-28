# train_negpos_openset_v2.py
# 运行：
#   python train_negpos_openset_v2.py --data_dir data_cut_10_v2 --epochs 50 --batch_size 64
#
# 说明：
# - 已知类(训练/分类)：1,2,3,4,5,7,8 -> 映射成 0..6
# - 未知类(只用于测试开集)：0,6,9 -> label = -1
#
# 关键改动：
# - 开集 score = (1 - sim1) + beta*(1 - gap)  (gap = sim1 - sim2)
# - mixup open loss: KL(p || uniform)
# - 输出 AUROC / Best F1 / Best NA，并用 Best NA 保存模型

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


# -------------------------
# 0) 标签定义（按你给的 0~9）
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

UNKNOWN_ORIG_IDS = {0, 6, 9}
KNOWN_ORIG_IDS = [i for i in range(10) if i not in UNKNOWN_ORIG_IDS]  # [1,2,3,4,5,7,8]

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# -------------------------
# 1) AddMarginProduct（正/负 margin）
# -------------------------
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.bool)

class AddMarginProduct(nn.Module):
    """
    true logit = cosine - margin
    - 正 margin: margin=+mpos -> cos - mpos
    - 负 margin: margin=-mneg -> cos + mneg
    """
    def __init__(self, in_features, out_features, scale_factor=30.0, margin=0.40):
        super().__init__()
        self.scale_factor = float(scale_factor)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature: torch.Tensor, label: Optional[torch.Tensor] = None):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))  # [B,C]
        if label is None:
            return cosine * self.scale_factor
        phi = cosine - self.margin
        mask = one_hot(label, cosine.shape[1])
        out = torch.where(mask, phi, cosine) * self.scale_factor
        return out


# -------------------------
# 2) 2D ResNet backbone + 双 margin head
# -------------------------
class ResNet2DEmbed(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # f_neg = h -> embed
        self.fc_neg = nn.Linear(in_dim, embed_dim)
        self.bn_neg = nn.BatchNorm1d(embed_dim)

        # f_pos = g(f_neg)
        self.fc_pos = nn.Linear(embed_dim, embed_dim)
        self.bn_pos = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        h = self.backbone(x)  # [B,in_dim]
        f_neg = F.relu(self.bn_neg(self.fc_neg(h)), inplace=True)     # [B,D]
        f_pos = F.relu(self.bn_pos(self.fc_pos(f_neg)), inplace=True) # [B,D]
        return f_neg, f_pos


class NegPosOpenSetNet(nn.Module):
    def __init__(self, num_known: int, embed_dim: int = 256,
                 mneg: float = 0.20, mpos: float = 0.40, s: float = 30.0,
                 pretrained: bool = True):
        super().__init__()
        self.embed = ResNet2DEmbed(embed_dim=embed_dim, pretrained=pretrained)
        self.cls_neg = AddMarginProduct(embed_dim, num_known, scale_factor=s, margin=-mneg)
        self.cls_pos = AddMarginProduct(embed_dim, num_known, scale_factor=s, margin=+mpos)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        f_neg, f_pos = self.embed(x)
        if y is None:
            # 推理：不注入 margin
            logit_neg = self.cls_neg(f_neg, None)
            logit_pos = self.cls_pos(f_pos, None)
        else:
            logit_neg = self.cls_neg(f_neg, y)
            logit_pos = self.cls_pos(f_pos, y)
        return logit_neg, logit_pos, f_neg, f_pos


# -------------------------
# 3) Dataset：文件夹=类别
# -------------------------
@dataclass
class Sample:
    path: str
    orig_label: int

def build_samples(data_dir: str) -> List[Sample]:
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

def split_train_test(samples: List[Sample], seed: int = 0, ratio: float = 0.5) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = {}
    for s in samples:
        by_class.setdefault(s.orig_label, []).append(s)

    train, test = [], []
    for c, lst in by_class.items():
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(n * ratio)
        train.extend(lst[:n_train])
        test.extend(lst[n_train:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


class ShipImageDataset(Dataset):
    def __init__(self, samples: List[Sample], transform, known_id_map: Dict[int, int], unknown_label: int = -1):
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
        y = self.known_id_map.get(s.orig_label, self.unknown_label)  # known->0..6, unknown->-1
        return x, torch.tensor(y, dtype=torch.long), s.orig_label


# -------------------------
# 4) 原型与开集打分（sim1/sim2 gap 融合）
# -------------------------
@torch.no_grad()
def compute_prototypes(model: nn.Module, loader: DataLoader, device: torch.device, num_known: int) -> torch.Tensor:
    """
    用训练集(known) pos embedding 计算每类 prototype
    """
    model.eval()
    feat_sum = None
    count = torch.zeros(num_known, dtype=torch.long, device=device)

    for x, y, _ in loader:
        mask = (y >= 0)
        if not mask.any():
            continue
        x = x[mask].to(device)
        y = y[mask].to(device)

        _, _, _, f_pos = model(x, None)
        f_pos = F.normalize(f_pos, dim=1)

        if feat_sum is None:
            feat_sum = torch.zeros(num_known, f_pos.shape[1], device=device)

        for c in range(num_known):
            m = (y == c)
            if m.any():
                feat_sum[c] += f_pos[m].sum(dim=0)
                count[c] += int(m.sum().item())

    protos = torch.zeros(num_known, feat_sum.shape[1], device=device)
    for c in range(num_known):
        if count[c] > 0:
            protos[c] = feat_sum[c] / count[c]
    protos = F.normalize(protos, dim=1)
    return protos


@torch.no_grad()
def collect_scores_and_preds(model: nn.Module, loader: DataLoader, device: torch.device,
                            protos: torch.Tensor, beta_gap: float = 0.5) -> Dict[str, np.ndarray]:
    """
    对 test 集收集：
    - score: unknown 分数（越大越像 unknown）
    - unk: 真实 unknown 标记（1 unknown / 0 known）
    - y_true: known 标签(0..K-1) 或 -1
    - y_pred_closed: 最近原型分类结果（对 known 评估用）
    """
    model.eval()
    scores = []
    unk = []
    y_true = []
    y_pred_closed = []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        _, _, _, f_pos = model(x, None)
        f_pos = F.normalize(f_pos, dim=1)

        sims = torch.matmul(f_pos, protos.T)  # [B,K]
        # top1/top2
        top2_sim, top2_idx = torch.topk(sims, k=2, dim=1)
        sim1 = top2_sim[:, 0]
        sim2 = top2_sim[:, 1]
        pred = top2_idx[:, 0]

        gap = sim1 - sim2

        # unknown score: dist + beta*(1-gap)
        # dist = 1 - sim1
        score = (1.0 - sim1) + float(beta_gap) * (1.0 - gap)

        scores.extend(score.detach().cpu().tolist())
        unk.extend((y < 0).detach().cpu().int().tolist())
        y_true.extend(y.detach().cpu().tolist())
        y_pred_closed.extend(pred.detach().cpu().tolist())

    return {
        "score": np.array(scores, dtype=np.float32),
        "unk": np.array(unk, dtype=np.int32),
        "y_true": np.array(y_true, dtype=np.int32),
        "y_pred_closed": np.array(y_pred_closed, dtype=np.int32),
    }


def auroc(scores: np.ndarray, unk: np.ndarray) -> float:
    """
    unk: 1=unknown positive, 0=known negative
    scores: higher => more unknown
    """
    scores = scores.astype(np.float64)
    unk = unk.astype(np.int32)

    pos = (unk == 1)
    neg = (unk == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # sort by score desc
    order = np.argsort(-scores)
    unk_sorted = unk[order]

    tps = np.cumsum(unk_sorted == 1)
    fps = np.cumsum(unk_sorted == 0)

    tpr = tps / (n_pos + 1e-12)
    fpr = fps / (n_neg + 1e-12)

    # trapezoid on fpr-tpr curve
    # ensure starts at (0,0)
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    return float(np.trapz(tpr, fpr))


def sweep_thresholds(scores: np.ndarray, unk: np.ndarray,
                     y_true: np.ndarray, y_pred_closed: np.ndarray,
                     steps: int = 400) -> Dict:
    """
    扫 tau 得到：
    - Best unknown F1 (unknown detection)
    - Best NA (NA=(AKS+AUS)/2)
    同时返回各自最优 tau 与指标
    规则：
      pred_unknown = (score > tau)
      if not unknown: predicted class = y_pred_closed (nearest prototype)
    """
    scores = scores.astype(np.float32)
    unk = unk.astype(np.int32)
    y_true = y_true.astype(np.int32)
    y_pred_closed = y_pred_closed.astype(np.int32)

    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0

    n_unk = int((unk == 1).sum())
    n_known = int((unk == 0).sum())

    best_f1 = {"f1": -1.0, "tau": 0.0, "p": 0.0, "r": 0.0}
    best_na = {"na": -1.0, "tau": 0.0, "aks": 0.0, "aus": 0.0}

    taus = np.linspace(lo, hi, steps)

    for tau in taus:
        pred_unk = (scores > tau).astype(np.int32)  # 1 unknown

        # unknown detection confusion
        tp = int(((pred_unk == 1) & (unk == 1)).sum())
        fp = int(((pred_unk == 1) & (unk == 0)).sum())
        fn = int(((pred_unk == 0) & (unk == 1)).sum())

        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)

        if f1 > best_f1["f1"]:
            best_f1.update({"f1": float(f1), "tau": float(tau), "p": float(p), "r": float(r)})

        # AKS: known 样本中 “未拒判 且 分类正确” 的比例
        # AUS: unknown 样本中 “被拒判” 的比例
        if n_known > 0:
            known_mask = (unk == 0)
            accept_known = (pred_unk == 0) & known_mask
            correct_accept = accept_known & (y_pred_closed == y_true)
            aks = float(correct_accept.sum() / (n_known + 1e-12))
        else:
            aks = 0.0

        if n_unk > 0:
            aus = float(tp / (n_unk + 1e-12))
        else:
            aus = 0.0

        na = 0.5 * (aks + aus)

        if na > best_na["na"]:
            best_na.update({"na": float(na), "tau": float(tau), "aks": float(aks), "aus": float(aus)})

    return {"best_f1": best_f1, "best_na": best_na}


@torch.no_grad()
def per_class_accuracy_known(y_true: np.ndarray, y_pred_closed: np.ndarray, unk: np.ndarray, num_known: int) -> List[float]:
    """
    只在 known 样本上做 closed-set per-class accuracy（不考虑拒判）
    """
    accs = []
    for c in range(num_known):
        m = (unk == 0) & (y_true == c)
        if m.sum() == 0:
            accs.append(float("nan"))
        else:
            accs.append(float((y_pred_closed[m] == y_true[m]).mean()))
    return accs


# -------------------------
# 5) 训练（双 CE + mixup KL-to-uniform open loss）
# -------------------------
def mixup_batch(x: torch.Tensor, alpha: float = 1.0):
    """
    returns x_mix, perm_index, lam
    """
    if alpha <= 0:
        return x, None, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, idx, float(lam)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device,
                    lam_pos: float = 1.0,
                    w_open: float = 0.1,
                    mixup_alpha: float = 1.0,
                    grad_clip: float = 0.0):
    model.train()
    total_loss = 0.0
    total = 0

    for x, y, _ in loader:
        # 训练只用 known
        mask = (y >= 0)
        if not mask.any():
            continue
        x = x[mask].to(device)
        y = y[mask].to(device)

        # 1) neg/pos 两个 CE（注入 margin）
        logit_neg, logit_pos, _, f_pos = model(x, y)
        loss_T = F.cross_entropy(logit_neg, y)
        loss_D = F.cross_entropy(logit_pos, y)
        loss_cls = loss_T + lam_pos * loss_D

        # 2) mixup open loss：让 mixup 样本预测接近均匀分布（最大熵）
        x_mix, _, _ = mixup_batch(x, alpha=mixup_alpha)
        # 推理模式 logits（不注入 margin），更稳定
        _, logit_pos_mix, _, _ = model(x_mix, None)
        p = F.softmax(logit_pos_mix, dim=1)
        u = torch.full_like(p, 1.0 / p.size(1))
        loss_open = F.kl_div(p.log(), u, reduction="batchmean")

        loss = loss_cls + w_open * loss_open

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

    return total_loss / max(total, 1)


# -------------------------
# 6) main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_cut_10_v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # model
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--mneg", type=float, default=0.20)
    parser.add_argument("--mpos", type=float, default=0.40)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--no_pretrained", action="store_true")

    # train
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lam_pos", type=float, default=1.0)
    parser.add_argument("--w_open", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    # openset scoring
    parser.add_argument("--beta_gap", type=float, default=0.5, help="gap term weight in unknown score")
    parser.add_argument("--tau_steps", type=int, default=400)

    parser.add_argument("--save_path", type=str, default="negpos_openset_bestNA.pt")

    args = parser.parse_args()

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    # build data
    all_samples = build_samples(args.data_dir)
    if len(all_samples) == 0:
        raise RuntimeError(f"No images found in {args.data_dir}. Check folders match class names in ID2NAME.")

    train_samples, test_samples = split_train_test(all_samples, seed=args.seed, ratio=0.5)

    # map known orig -> 0..6
    known_id_map = {orig_id: i for i, orig_id in enumerate(KNOWN_ORIG_IDS)}
    known_names = [ID2NAME[i] for i in KNOWN_ORIG_IDS]

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ShipImageDataset(train_samples, train_tf, known_id_map, unknown_label=-1)
    test_ds  = ShipImageDataset(test_samples,  test_tf,  known_id_map, unknown_label=-1)

    def collate(batch):
        xs, ys, origs = zip(*batch)
        return torch.stack(xs, 0), torch.stack(ys, 0), torch.tensor(origs, dtype=torch.long)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                              collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                             collate_fn=collate, drop_last=False)

    # model
    model = NegPosOpenSetNet(
        num_known=len(KNOWN_ORIG_IDS),
        embed_dim=args.embed_dim,
        mneg=args.mneg,
        mpos=args.mpos,
        s=args.scale,
        pretrained=(not args.no_pretrained),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_na = -1.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, device,
            lam_pos=args.lam_pos,
            w_open=args.w_open,
            mixup_alpha=args.mixup_alpha,
            grad_clip=args.grad_clip
        )

        # prototypes from train
        protos = compute_prototypes(model, train_loader, device, num_known=len(KNOWN_ORIG_IDS))

        # scores on test
        pack = collect_scores_and_preds(model, test_loader, device, protos, beta_gap=args.beta_gap)
        scores = pack["score"]
        unk = pack["unk"]
        y_true = pack["y_true"]
        y_pred_closed = pack["y_pred_closed"]

        # diagnosis: per-class accuracy (known, closed-set)
        accs = per_class_accuracy_known(y_true, y_pred_closed, unk, num_known=len(KNOWN_ORIG_IDS))

        # metrics: AUROC + best F1 + best NA
        auc = auroc(scores, unk)
        sweep = sweep_thresholds(scores, unk, y_true, y_pred_closed, steps=args.tau_steps)
        bf1 = sweep["best_f1"]
        bna = sweep["best_na"]

        print(f"\nEpoch {epoch:03d}/{args.epochs} | loss={loss:.4f}")
        print("🔍 [Diagnosis] Per-Class Accuracy (Known Classes):")
        for i, a in enumerate(accs):
            nm = known_names[i]
            if np.isnan(a):
                print(f"   - Class {i} ({nm}): N/A")
            else:
                print(f"   - Class {i} ({nm}): {a*100:.2f}%")

        print(f"   >>> [Result] AUROC: {auc:.4f} | Best F1: {bf1['f1']:.4f} (tau={bf1['tau']:.3f})")
        print(f"   >>> [Metrics] Best NA: {bna['na']:.4f} (AKS: {bna['aks']:.4f}, AUS: {bna['aus']:.4f}, Tau: {bna['tau']:.3f})")

        # save by best NA
        if bna["na"] > best_na:
            best_na = bna["na"]
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "best_na": best_na,
                "best_na_detail": bna,
                "best_f1_detail": bf1,
                "auroc": auc,
                "prototypes": protos.detach().cpu(),
                "known_orig_ids": KNOWN_ORIG_IDS,
                "known_names": known_names,
            }
            torch.save(ckpt, args.save_path)
            print("   >>> ✨ Best Model Saved!\n")

    print("[Done] Training finished.")


if __name__ == "__main__":
    main()

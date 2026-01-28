import os, argparse, random
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
UNKNOWN_ORIG_IDS = {6, 8, 9}
KNOWN_ORIG_IDS = [i for i in range(10) if i not in UNKNOWN_ORIG_IDS]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---- model: only load embed.* from ckpt ----
class ResNet2DEmbed(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_neg = nn.Linear(in_dim, embed_dim)
        self.bn_neg = nn.BatchNorm1d(embed_dim)
        self.fc_pos = nn.Linear(embed_dim, embed_dim)
        self.bn_pos = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        h = self.backbone(x)
        f_neg = F.relu(self.bn_neg(self.fc_neg(h)), inplace=True)
        f_pos = F.relu(self.bn_pos(self.fc_pos(f_neg)), inplace=True)
        return f_neg, f_pos


@dataclass
class Sample:
    path: str
    orig_label: int


def build_samples(data_dir: str) -> List[Sample]:
    out = []
    for name in os.listdir(data_dir):
        d = os.path.join(data_dir, name)
        if not os.path.isdir(d): 
            continue
        if name not in NAME2ID:
            continue
        oid = NAME2ID[name]
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if os.path.isdir(fp): 
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXT:
                out.append(Sample(fp, oid))
    return out


class ImgDS(Dataset):
    def __init__(self, samples: List[Sample], tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.tfm(img)
        disp = "unknown" if s.orig_label in UNKNOWN_ORIG_IDS else ID2NAME[s.orig_label]
        return x, disp


def load_embed_only(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    filt = {k.replace("embed.", ""): v for k, v in state.items() if k.startswith("embed.")}
    missing, unexpected = model.load_state_dict(filt, strict=False)
    return missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_cut_10_v2")
    ap.add_argument("--ckpt", type=str, default="result/20260128_000605/openset_best.pt")
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_per_class", type=int, default=600)
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="tsne_openset.png")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] device:", device)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    samples = build_samples(args.data_dir)
    if not samples:
        raise RuntimeError("No images found.")

    # stratified sampling (7 known + unknown)
    buckets: Dict[str, List[Sample]] = {}
    for s in samples:
        key = "unknown" if s.orig_label in UNKNOWN_ORIG_IDS else ID2NAME[s.orig_label]
        buckets.setdefault(key, []).append(s)

    picked = []
    for k, lst in buckets.items():
        random.shuffle(lst)
        picked.extend(lst[: min(args.max_per_class, len(lst))])

    print("[Info] sampled:")
    for k in sorted(buckets.keys()):
        print(" ", k, "=>", min(args.max_per_class, len(buckets[k])))

    ds = ImgDS(picked, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = ResNet2DEmbed(embed_dim=args.embed_dim).to(device)
    missing, unexpected = load_embed_only(model, args.ckpt)
    print("[Info] ckpt loaded. missing:", len(missing), "unexpected:", len(unexpected))

    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, lab in dl:
            x = x.to(device)
            _, f_pos = model(x)
            f_pos = F.normalize(f_pos, dim=1)
            feats.append(f_pos.cpu().numpy())
            labels.extend(list(lab))

    X = np.concatenate(feats, axis=0)
    print("[Info] feature:", X.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        early_exaggeration=18,
        learning_rate=200,
        n_iter=1500,
        init="pca",
        random_state=args.seed,
    )
    Z = tsne.fit_transform(X)

    uniq = sorted(set(labels), key=lambda s: (s == "unknown", s))
    cmap = plt.get_cmap("tab10")
    color_map = {}
    ci = 0
    for name in uniq:
        if name == "unknown":
            color_map[name] = "black"
        else:
            color_map[name] = cmap(ci % 10)
            ci += 1

    plt.figure(figsize=(12, 8), dpi=150)
    for name in uniq:
        idx = np.array([i for i, lab in enumerate(labels) if lab == name], dtype=np.int64)
        pts = Z[idx]
        plt.scatter(
            pts[:, 0], pts[:, 1],
            s=36, alpha=0.9, c=[color_map[name]],
            label=name, linewidths=0
        )

    plt.title("t-SNE Visualization (7 Known Classes + 1 Unknown)")
    plt.xticks([])
    plt.yticks([])
    plt.legend(markerscale=1.6, fontsize=10, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print("[Saved]", args.out)


if __name__ == "__main__":
    main()

"""
开集识别主程序
命令行入口,包含参数解析、数据加载、模型初始化、训练
"""

import os
import argparse
import json
import time
import logging
from datetime import datetime
from typing import Tuple

import torch

from openset_config import (
    ID2NAME, NAME2ID, KNOWN_ORIG_IDS, UNKNOWN_ORIG_IDS,
    ModelConfig, TrainingConfig, PseudoUnknownConfig,
    ThresholdConfig, AugmentConfig
)
from openset_data import (
    build_samples, split_train_test, ShipImageFolderDataset,
    get_train_transforms, get_test_transforms, get_dataloader
)
from openset_models import NegPosNet2D
from openset_train import train
from openset_viz import plot_learning_curves, plot_tsne



# -------------------------
# 日志配置
# -------------------------
def setup_logger(log_file_path):
    """配置日志记录器"""
    logger = logging.getLogger("OpenSet")
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# -------------------------
# 参数解析
# -------------------------
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="开集识别训练")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="data_cut_10_v2",
                      help="数据集根目录")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                      help="训练集比例")

    # 模型参数
    parser.add_argument("--embed_dim", type=int, default=256,
                      help="嵌入维度")
    parser.add_argument("--mneg", type=float, default=0.20,
                      help="负margin(transferable)")
    parser.add_argument("--mpos", type=float, default=0.40,
                      help="正margin(discriminative)")
    parser.add_argument("--scale", type=float, default=30.0,
                      help="softmax温度参数")
    parser.add_argument("--no_pretrained", action="store_true",
                      help="不使用预训练模型")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                      help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256,
                      help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                      help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                      help="权重衰减")
    parser.add_argument("--lam", type=float, default=1.0,
                      help="L_T和L_D的权重比")
    parser.add_argument("--seed", type=int, default=42,
                      help="随机种子")
    parser.add_argument("--num_workers", type=int, default=8,
                      help="数据加载工作进程数")
    parser.add_argument("--grad_clip", type=float, default=1,
                      help="梯度裁剪阈值")

    # 原型参数
    parser.add_argument("--use_rectified_proto", action="store_true", default=False,
                      help="使用修正原型(论文公式4-6)")
    parser.add_argument("--percentile", type=float, default=90.0,
                      help="阈值分位数")
    parser.add_argument("--use_class_tau", action="store_true", default=False,
                      help="使用按类阈值")

    # 伪未知类参数
    parser.add_argument("--use_pseudo_unknown", action="store_true", default=True,
                      help="使用伪未知类生成")
    parser.add_argument("--pseudo_ratio", type=float, default=1.0,
                      help="伪未知类比例")

    # 边界损失参数
    parser.add_argument("--use_boundary_loss", action="store_true", default=True,
                      help="使用边界约束损失")
    parser.add_argument("--intra_weight", type=float, default=1.0,
                      help="类内聚合损失权重")
    parser.add_argument("--inter_weight", type=float, default=1.0,
                      help="类间分离损失权重")
    parser.add_argument("--open_weight", type=float, default=10.0,
                      help="开放空间约束权重")

    # 保存参数
    parser.add_argument("--save_path", type=str, default="openset_best.pt",
                      help="模型保存路径")
    parser.add_argument("--log_interval", type=int, default=1,
                      help="日志打印间隔")

    args = parser.parse_args()
    return args


# -------------------------
# 主函数
# -------------------------
def main():
    """主函数"""
    args = parse_args()

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("result", timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    # 初始化日志
    log_file = os.path.join(result_dir, "train.log")
    logger = setup_logger(log_file)
    logger.info(f"[Info] Results will be saved to: {result_dir}")
    
    # 更新 save_path 到 result_dir
    if os.path.dirname(args.save_path) == "":
        args.save_path = os.path.join(result_dir, args.save_path)
    else:
        # 如果用户指定了路径，我们还是强制放到 result_dir 下
        basename = os.path.basename(args.save_path)
        args.save_path = os.path.join(result_dir, basename)

    # 保存配置
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 固定随机种子
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Info] device = {device}")

    # 打印已知类和未知类
    logger.info(f"[Info] Known orig IDs: {KNOWN_ORIG_IDS} => new IDs 0..{len(KNOWN_ORIG_IDS)-1}")
    logger.info(f"[Info] Unknown orig IDs: {sorted(list(UNKNOWN_ORIG_IDS))}")

    # 1. 准备数据
    logger.info("\n[Step 1] Loading data...")
    all_samples = build_samples(args.data_dir)
    if len(all_samples) == 0:
        raise RuntimeError(
            f"No images found under {args.data_dir}. "
            f"Check folder names match: {list(NAME2ID.keys())}"
        )

    logger.info(f"  Total samples: {len(all_samples)}")

    # 分层划分训练集和测试集
    train_samples, test_samples = split_train_test(
        all_samples, seed=args.seed, ratio=args.train_ratio
    )
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"  Test samples: {len(test_samples)}")

    # 已知类映射
    known_id_map = {orig_id: i for i, orig_id in enumerate(KNOWN_ORIG_IDS)}
    num_known = len(KNOWN_ORIG_IDS)

    # 2. 数据增强
    logger.info("\n[Step 2] Preparing data transforms...")
    train_transform = get_train_transforms(
        image_size=AugmentConfig.image_size,
        horizontal_flip_prob=AugmentConfig.horizontal_flip_prob,
        color_jitter_brightness=AugmentConfig.color_jitter_brightness,
        color_jitter_contrast=AugmentConfig.color_jitter_contrast,
        color_jitter_saturation=AugmentConfig.color_jitter_saturation,
        color_jitter_hue=AugmentConfig.color_jitter_hue,
        normalize_mean=AugmentConfig.normalize_mean,
        normalize_std=AugmentConfig.normalize_std
    )

    test_transform = get_test_transforms(
        image_size=AugmentConfig.image_size,
        normalize_mean=AugmentConfig.normalize_mean,
        normalize_std=AugmentConfig.normalize_std
    )

    # 3. 数据加载器
    logger.info("\n[Step 3] Creating data loaders...")
    train_ds = ShipImageFolderDataset(
        train_samples, train_transform, known_id_map, unknown_label=-1
    )
    test_ds = ShipImageFolderDataset(
        test_samples, test_transform, known_id_map, unknown_label=-1
    )

    train_loader = get_dataloader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = get_dataloader(
        test_ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    # 4. 模型初始化
    logger.info("\n[Step 4] Initializing model...")
    model = NegPosNet2D(
        num_known_classes=num_known,
        embed_dim=args.embed_dim,
        mneg=args.mneg,
        mpos=args.mpos,
        scale_factor=args.scale,
        pretrained=(not args.no_pretrained)
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # 5. 优化器
    logger.info("\n[Step 5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    logger.info(f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    logger.info(f"  Scheduler: CosineAnnealingLR (T_max={args.epochs})")

    # 6. 打印训练配置
    logger.info("\n[Step 6] Training configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.lr}")
    logger.info(f"  - Margin: mneg={args.mneg}, mpos={args.mpos}, scale={args.scale}")
    logger.info(f"  - Lambda: {args.lam}")
    logger.info(f"  - Use rectified proto: {args.use_rectified_proto}")
    logger.info(f"  - Use pseudo unknown: {args.use_pseudo_unknown}")
    logger.info(f"  - Use boundary loss: {args.use_boundary_loss}")
    if args.use_boundary_loss:
        logger.info(f"    * intra_weight: {args.intra_weight}")
        logger.info(f"    * inter_weight: {args.inter_weight}")
        logger.info(f"    * open_weight: {args.open_weight}")
    logger.info(f"  - Results directory: {result_dir}")

    # 7. 开始训练
    logger.info("\n" + "="*80)
    logger.info("[Step 7] Starting training...")
    logger.info("="*80)

    history, best_state = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,  # 传入调度器
        device=device,
        num_known=num_known,
        num_epochs=args.epochs,
        lam=args.lam,
        use_rectified_proto=args.use_rectified_proto,
        use_pseudo_unknown=args.use_pseudo_unknown,
        use_boundary_loss=args.use_boundary_loss,
        pseudo_ratio=args.pseudo_ratio,
        use_class_tau=args.use_class_tau,
        percentile=args.percentile,
        boundary_weights=(args.intra_weight, args.inter_weight, args.open_weight),
        grad_clip=args.grad_clip,
        save_path=args.save_path,
        log_interval=args.log_interval,
        logger=logger  # 传入 logger
    )

    # 保存训练历史
    with open(os.path.join(result_dir, "history.json"), "w") as f:
        json.dump(history, f)

    # 8. 绘制图表
    logger.info("\n[Step 8] Generating visualizations...")
    plot_learning_curves(history, result_dir)
    
    # 重新加载最佳模型进行 t-SNE
    logger.info("Loading best model for t-SNE...")
    model.load_state_dict(best_state["model"])
    plot_tsne(model, test_loader, device, result_dir, title=f"t-SNE (Epoch {best_state['epoch']})")

    # 9. 打印最终结果
    logger.info("\n" + "="*80)
    logger.info("[Training Summary]")
    logger.info("="*80)
    logger.info(f"Best epoch: {best_state['epoch']}")
    logger.info(f"Best open_f1: {best_state['open_f1']:.4f}")
    logger.info(f"  - Known accuracy: {best_state['known_acc']:.4f}")
    logger.info(f"  - Open precision: {best_state['open_precision']:.4f}")
    logger.info(f"  - Open recall: {best_state['open_recall']:.4f}")
    logger.info(f"  - Open accuracy: {best_state['open_acc']:.4f}")
    logger.info(f"  - Best NA: {best_state.get('na', 0.0):.4f}")
    logger.info(f"    * AKS: {best_state.get('aks', 0.0):.4f}")
    logger.info(f"    * AUS: {best_state.get('aus', 0.0):.4f}")
    logger.info(f"  - AUROC: {best_state.get('auroc', 0.0):.4f}")
    logger.info(f"\nAll results saved to: {result_dir}")

    return best_state


# -------------------------
# 入口
# -------------------------
if __name__ == "__main__":
    main()

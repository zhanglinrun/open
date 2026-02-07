"""
孪生网络开集识别 - 完整流水线入口

实验流程 (对应会议纪要 4.1-4.3):
  Phase A: 基础训练 - 孪生网络度量学习
  Phase B: 构建特征库
  Phase C: 有偏/无偏判定 (4.1)
  Phase D: 无偏样本特征分布优化可视化 (4.2)
  Phase E: 有偏样本图像生成 + 特征更新 + 可视化 (4.3)

使用方式:
  # 完整流水线
  python siamese_main.py --data_dir MSTAR --epochs 100

  # 仅训练
  python siamese_main.py --data_dir MSTAR --epochs 100 --phase A

  # 从已有模型开始 (跳过训练)
  python siamese_main.py --data_dir MSTAR --ckpt result/xxx/siamese_best.pt --phase BCD
"""

import os
import argparse
import json
import random
import logging
from datetime import datetime
from typing import Dict

import numpy as np
import torch

from siamese_config import (
    ID2NAME, NAME2ID, KNOWN_ORIG_IDS, UNKNOWN_ORIG_IDS,
    SiameseModelConfig, SiameseTrainingConfig, AugmentConfig,
    BiasJudgeConfig,
)
from siamese_data import (
    build_samples, build_samples_from_split, split_train_test,
    get_train_transforms, get_test_transforms,
    create_single_loader, create_batch_hard_loader,
)
from siamese_model import SiameseFingerprintWithClassifier
from siamese_train import train_siamese, evaluate_openset, finetune_with_augmented_data
# CosFace 对抗边界已内嵌在 siamese_model.AddMarginProduct 中
from feature_library import FeatureLibrary
from bias_judge import BiasJudge
from feat_distribution_optimizer import FeatureDistributionOptimizer, DistributionMetrics
from biased_image_gen import BiasedImageGenerator
from siamese_viz import (
    plot_tsne_fingerprints,
    plot_tsne_bias_annotated,
    plot_tsne_before_after,
    plot_class_distance_matrix,
    plot_training_curves,
    plot_generation_effect,
    plot_generation_three_stage,
    plot_optimization_effect,
    plot_biased_generation_effect,
)


# =============================================================================
# 日志配置
# =============================================================================
def setup_logger(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger("SiameseOpenSet")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# =============================================================================
# 参数解析
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="孪生网络开集识别完整流水线")

    # 数据
    parser.add_argument("--data_dir", type=str, default="MSTAR",
                        help="数据集目录 (含 train/test 子目录, 或直接含类别子目录)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例 (若无 train/test 子目录)")

    # 模型
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="切片指纹维度")
    parser.add_argument("--no_pretrained", action="store_true")

    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 损失
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Triplet loss margin")
    parser.add_argument("--center_weight", type=float, default=0.01,
                        help="Center loss 权重")
    parser.add_argument("--cls_weight", type=float, default=0.5,
                        help="辅助分类损失权重")

    # 有偏判定 (4.1)
    parser.add_argument("--bias_percentile", type=float, default=95.0,
                        help="有偏判定阈值分位数")
    parser.add_argument("--bias_threshold", type=float, default=0.5,
                        help="综合有偏评分阈值")

    # 特征优化 (4.2)
    parser.add_argument("--opt_phase1_epochs", type=int, default=15,
                        help="Phase D-1 (对抗边界规整化) 轮数")
    parser.add_argument("--opt_phase2_epochs", type=int, default=5,
                        help="Phase D-2 (伪未知约束) 轮数")
    parser.add_argument("--opt_phase3_epochs", type=int, default=0,
                        help="(保留, 未使用)")

    # 图像生成 (4.3)
    parser.add_argument("--gen_rotation_range", type=float, default=30.0)
    parser.add_argument("--gen_rotation_steps", type=int, default=6)
    parser.add_argument("--gen_mixup_num", type=int, default=3)

    # 流程控制
    parser.add_argument("--phase", type=str, default="ABCDE",
                        help="运行哪些阶段 (A=训练, B=特征库, C=有偏判定, D=分布优化, E=图像生成)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="加载已有模型权重")

    # 输出
    parser.add_argument("--save_path", type=str, default="siamese_best.pt")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=1)

    return parser.parse_args()


# =============================================================================
# 主函数
# =============================================================================
def main():
    args = parse_args()

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("result_siamese", timestamp)
    os.makedirs(result_dir, exist_ok=True)

    # 日志
    logger = setup_logger(os.path.join(result_dir, "train.log"))
    logger.info(f"Results → {result_dir}")

    # 保存配置
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 已知/未知类
    known_id_map = {orig_id: i for i, orig_id in enumerate(KNOWN_ORIG_IDS)}
    num_known = len(KNOWN_ORIG_IDS)
    logger.info(f"Known classes: {KNOWN_ORIG_IDS} → 0..{num_known-1}")
    logger.info(f"Unknown classes: {sorted(UNKNOWN_ORIG_IDS)}")

    # =========================================================================
    # Step 1: 加载数据
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("[Step 1] 加载数据")
    logger.info("=" * 70)

    split_train = os.path.join(args.data_dir, "train")
    split_test = os.path.join(args.data_dir, "test")
    use_split = os.path.isdir(split_train) and os.path.isdir(split_test)

    if use_split:
        train_samples = build_samples_from_split(args.data_dir, "train")
        test_samples = build_samples_from_split(args.data_dir, "test")
        logger.info("  Using pre-split: train/test")
    else:
        all_samples = build_samples(args.data_dir)
        if not all_samples:
            raise RuntimeError(f"No images found under {args.data_dir}")
        train_samples, test_samples = split_train_test(
            all_samples, seed=args.seed, ratio=args.train_ratio
        )

    # 严格开集: 训练集只保留已知类
    train_samples_known = [s for s in train_samples if s.orig_label in set(KNOWN_ORIG_IDS)]
    logger.info(f"  Train (known only): {len(train_samples_known)}")
    logger.info(f"  Test (all): {len(test_samples)}")

    # 数据增强
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    # DataLoader
    train_loader = create_single_loader(
        train_samples_known, train_transform, known_id_map,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = create_single_loader(
        test_samples, test_transform, known_id_map,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    # =========================================================================
    # Step 2: 初始化模型
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("[Step 2] 初始化模型")
    logger.info("=" * 70)

    model = SiameseFingerprintWithClassifier(
        num_known_classes=num_known,
        backbone=args.backbone,
        embed_dim=args.embed_dim,
        pretrained=(not args.no_pretrained),
    ).to(device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        logger.info(f"  Loaded checkpoint: {args.ckpt}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Fingerprint dim: {args.embed_dim}")

    # 模型保存路径
    save_path = os.path.join(result_dir, args.save_path)

    # =========================================================================
    # Phase A: 基础训练
    # =========================================================================
    if "A" in args.phase.upper():
        logger.info("\n" + "=" * 70)
        logger.info("[Phase A] 孪生网络基础训练 (SupCon + CosFace)")
        logger.info("=" * 70)

        # 收集训练前特征 (用于三阶段可视化对比)
        logger.info("  收集训练前特征 (未训练模型)...")
        with torch.no_grad():
            model.eval()
            _feats_init, _labels_init, _origs_init = [], [], []
            for x, y, orig, _ in test_loader:
                x = x.to(device)
                fps = model.extract_fingerprint(x)
                _feats_init.append(fps.cpu())
                _labels_init.append(y)
                _origs_init.extend(orig.tolist())
            feats_pretrain = torch.cat(_feats_init)
            labels_pretrain = torch.cat(_labels_init)
            origs_pretrain = _origs_init
            known_mask_pre = labels_pretrain >= 0
            metrics_pretrain = DistributionMetrics.compute_all(
                feats_pretrain[known_mask_pre], labels_pretrain[known_mask_pre]
            )
            logger.info(f"  Pre-train metrics: Fisher={metrics_pretrain.get('fisher_ratio',0):.1f}")

        # 保存训练前数据 (供 Phase E 使用)
        torch.save({
            "feats": feats_pretrain, "labels": labels_pretrain,
            "origs": origs_pretrain, "metrics": metrics_pretrain,
        }, os.path.join(result_dir, "pretrain_feats.pt"))

        # BalancedBatchSampler: 7 类 x 16 样本 = 112/batch
        train_loader_balanced = create_batch_hard_loader(
            train_samples_known, train_transform, known_id_map,
            n_classes=num_known, n_samples=16, num_workers=args.num_workers,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        logger.info(f"  SupCon (t=0.12) + CosFace (s=30, m=0.4) + CenterLoss + SpreadReg")
        logger.info(f"  BalancedBatch: {num_known} classes x 16 samples = {num_known*16}/batch")

        history, best_state = train_siamese(
            model=model,
            train_loader=train_loader_balanced,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_known=num_known,
            num_epochs=args.epochs,
            center_weight=args.center_weight,
            cls_weight=args.cls_weight,
            grad_clip=args.grad_clip,
            save_path=save_path,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            supcon_temperature=0.12,
            spread_weight=0.5,
            logger=logger,
        )

        # 保存训练历史
        with open(os.path.join(result_dir, "history.json"), "w") as f:
            json.dump(history, f)

        # 绘制训练曲线
        plot_training_curves(history, os.path.join(result_dir, "training_curves.png"))

        # 加载最佳模型
        if best_state:
            model.load_state_dict(best_state["model"])
            logger.info(f"  Loaded best model (epoch {best_state['epoch']})")

    # =========================================================================
    # Phase B: 构建特征库
    # =========================================================================
    if "B" in args.phase.upper():
        logger.info("\n" + "=" * 70)
        logger.info("[Phase B] 构建特征库")
        logger.info("=" * 70)

        feature_lib = FeatureLibrary(embed_dim=args.embed_dim)
        feature_lib.build_from_model(
            model, train_loader, device, known_id_map, ID2NAME
        )
        feature_lib.save(os.path.join(result_dir, "feature_library.pt"))
        logger.info(feature_lib.summary())

        # t-SNE 可视化 (基础)
        logger.info("  生成 t-SNE 可视化...")
        with torch.no_grad():
            model.eval()
            all_feats, all_labels, all_origs = [], [], []
            for x, y, orig, _ in test_loader:
                x = x.to(device)
                fps = model.extract_fingerprint(x)
                all_feats.append(fps.cpu())
                all_labels.append(y)
                all_origs.extend(orig.tolist())

            feats_tensor = torch.cat(all_feats)
            labels_tensor = torch.cat(all_labels)

        protos = feature_lib.get_prototypes()
        plot_tsne_fingerprints(
            feats_tensor, labels_tensor, all_origs,
            save_path=os.path.join(result_dir, "tsne_fingerprints.png"),
            title="Slice Fingerprints (t-SNE)",
            prototypes=protos,
        )

        # 类间距离矩阵
        class_names = [ID2NAME.get(KNOWN_ORIG_IDS[i], f"C{i}") for i in range(num_known)]
        plot_class_distance_matrix(
            protos, class_names,
            save_path=os.path.join(result_dir, "class_distance_matrix.png"),
        )

    # =========================================================================
    # Phase C: 有偏/无偏判定 (4.1)
    # =========================================================================
    if "C" in args.phase.upper():
        logger.info("\n" + "=" * 70)
        logger.info("[Phase C] 库内有偏/无偏判定 (4.1)")
        logger.info("=" * 70)

        # 加载或使用已有特征库
        lib_path = os.path.join(result_dir, "feature_library.pt")
        if os.path.exists(lib_path):
            feature_lib = FeatureLibrary.load(lib_path)
        else:
            feature_lib = FeatureLibrary(embed_dim=args.embed_dim)
            feature_lib.build_from_model(model, train_loader, device, known_id_map, ID2NAME)

        # 创建判定器
        judge = BiasJudge(
            feature_library=feature_lib,
            percentile=args.bias_percentile,
        )
        judge.fit(verbose=True)

        # 批量判定 (在训练集上)
        results, class_stats = judge.judge_batch(model, train_loader, device)
        report = judge.report(class_stats)
        logger.info(report)

        # 保存报告
        with open(os.path.join(result_dir, "bias_report.txt"), "w") as f:
            f.write(report)

        # 分离有偏/无偏
        unbiased_feats, biased_feats, biased_paths, _ = \
            judge.split_biased_unbiased(model, train_loader, device)

        # 可视化: 有偏/无偏标注
        with torch.no_grad():
            model.eval()
            viz_feats, viz_labels, viz_biased = [], [], []
            for x, y, _, _ in train_loader:
                mask = (y >= 0)
                if not mask.any():
                    continue
                x_k = x[mask].to(device)
                y_k = y[mask]
                fps = model.extract_fingerprint(x_k).cpu()
                for i in range(fps.size(0)):
                    cid = int(y_k[i].item())
                    result = judge.judge(fps[i], predicted_class=cid)
                    viz_feats.append(fps[i])
                    viz_labels.append(cid)
                    viz_biased.append(result.is_biased)

            if viz_feats:
                plot_tsne_bias_annotated(
                    torch.stack(viz_feats),
                    torch.tensor(viz_labels),
                    viz_biased,
                    save_path=os.path.join(result_dir, "tsne_bias_annotated.png"),
                    title="Biased vs Unbiased Slice Fingerprints (4.1)",
                )

        # 保存有偏/无偏分离结果 (含逐样本标签, 用于 4.3 可视化)
        torch.save({
            "unbiased_feats": unbiased_feats,
            "biased_feats": biased_feats,
            "biased_paths": biased_paths,
            "sample_is_biased": viz_biased,   # Phase C 判定的每个训练样本有偏标签
            "sample_feats": torch.stack(viz_feats) if viz_feats else None,
            "sample_labels": torch.tensor(viz_labels) if viz_labels else None,
        }, os.path.join(result_dir, "bias_split.pt"))

    # =========================================================================
    # Phase D: 无偏样本特征分布优化 (4.2)
    # =========================================================================
    if "D" in args.phase.upper():
        logger.info("\n" + "=" * 70)
        logger.info("[Phase D] 无偏样本深度特征分布优化 (4.2)")
        logger.info("=" * 70)

        # 加载特征库
        lib_path = os.path.join(result_dir, "feature_library.pt")
        if os.path.exists(lib_path):
            feature_lib = FeatureLibrary.load(lib_path)
        else:
            feature_lib = FeatureLibrary(embed_dim=args.embed_dim)
            feature_lib.build_from_model(model, train_loader, device, known_id_map, ID2NAME)

        # 解耦优化: 冻结 backbone, 只训练 embed_head + classifier
        logger.info("  构建解耦优化器...")
        optimizer_inst = FeatureDistributionOptimizer(
            model=model,
            feature_library=feature_lib,
            device=device,
            lr=5e-4,             # 只训练小模块, 可用正常学习率
            supcon_temperature=0.12,
            pseudo_weight=0.1,
            fisher_drop_limit=0.10,  # Fisher 最多允许下降 10%
        )

        # 收集优化前特征
        feats_before, labels_before, origs_before = \
            optimizer_inst.collect_features(test_loader)

        # 解耦优化 (Phase 1: 对抗边界规整化, Phase 2: 伪未知约束)
        opt_result = optimizer_inst.optimize_full(
            train_loader=train_loader,
            test_loader=test_loader,
            phase1_epochs=args.opt_phase1_epochs,
            phase2_epochs=args.opt_phase2_epochs,
            phase3_epochs=args.opt_phase3_epochs,
            eval_interval=5,
            logger=logger,
        )

        # 收集优化后特征
        feats_after, labels_after, origs_after = \
            optimizer_inst.collect_features(test_loader)

        # 4.2 核心可视化: 尝试加载训练前特征 (散→聚 对比更明显)
        pretrain_path = os.path.join(result_dir, "pretrain_feats.pt")
        # 也检查原始训练目录
        if not os.path.exists(pretrain_path) and args.ckpt:
            ckpt_dir = os.path.dirname(args.ckpt)
            pretrain_path_alt = os.path.join(ckpt_dir, "pretrain_feats.pt")
            if os.path.exists(pretrain_path_alt):
                pretrain_path = pretrain_path_alt

        feats_viz_before = feats_before
        labels_viz_before = labels_before
        origs_viz_before = origs_before
        metrics_viz_before = opt_result.get("metrics_before")

        if os.path.exists(pretrain_path):
            logger.info(f"  加载训练前特征用于可视化: {pretrain_path}")
            pt_data = torch.load(pretrain_path, map_location="cpu", weights_only=False)
            feats_viz_before = pt_data["feats"]
            labels_viz_before = pt_data["labels"]
            origs_viz_before = pt_data["origs"]
            metrics_viz_before = pt_data.get("metrics")

        # 新版 4.2 可视化: 训练前(散)→优化后(聚) + 置信椭圆
        plot_optimization_effect(
            feats_viz_before, labels_viz_before, origs_viz_before,
            feats_after, labels_after, origs_after,
            save_path=os.path.join(result_dir, "tsne_before_after_optimization.png"),
            metrics_before=metrics_viz_before,
            metrics_after=opt_result.get("metrics_after"),
        )

        # 也保留原始 Phase D 前后对比图 (细粒度)
        plot_tsne_before_after(
            feats_before, labels_before, origs_before,
            feats_after, labels_after, origs_after,
            save_path=os.path.join(result_dir, "tsne_phaseD_detail.png"),
            title_prefix="Phase D Detail (4.2)",
            metrics_before=opt_result.get("metrics_before"),
            metrics_after=opt_result.get("metrics_after"),
        )

        # 保存优化后模型
        torch.save(model.state_dict(), os.path.join(result_dir, "model_optimized.pt"))

        # 更新特征库
        feature_lib = FeatureLibrary(embed_dim=args.embed_dim)
        feature_lib.build_from_model(model, train_loader, device, known_id_map, ID2NAME)
        feature_lib.save(os.path.join(result_dir, "feature_library_optimized.pt"))

    # =========================================================================
    # Phase E: 有偏样本图像生成 (4.3)
    # =========================================================================
    if "E" in args.phase.upper():
        logger.info("\n" + "=" * 70)
        logger.info("[Phase E] 有偏样本图像生成与特征更新 (4.3)")
        logger.info("=" * 70)

        # 加载特征库
        lib_opt_path = os.path.join(result_dir, "feature_library_optimized.pt")
        lib_path = os.path.join(result_dir, "feature_library.pt")
        if os.path.exists(lib_opt_path):
            feature_lib = FeatureLibrary.load(lib_opt_path)
        elif os.path.exists(lib_path):
            feature_lib = FeatureLibrary.load(lib_path)
        else:
            feature_lib = FeatureLibrary(embed_dim=args.embed_dim)
            feature_lib.build_from_model(model, train_loader, device, known_id_map, ID2NAME)

        # 用当前模型重新提取有偏/无偏特征 (避免使用 Phase C 的过时特征)
        logger.info("  重新提取有偏/无偏特征 (当前模型)...")
        judge_e = BiasJudge(feature_lib, percentile=args.bias_percentile)
        judge_e.fit(verbose=False)
        unbiased_feats, biased_feats, biased_paths, _ = \
            judge_e.split_biased_unbiased(model, train_loader, device)

        # ---- Step E.1: 收集微调前的测试集特征 ----
        logger.info("  [E.1] 收集微调前特征...")
        gen_optimizer = FeatureDistributionOptimizer(
            model=model, feature_library=feature_lib, device=device,
            optimize_mode=False,  # 仅用于收集特征, 不冻结 backbone
        )
        feats_before_gen, labels_before_gen, origs_before_gen = \
            gen_optimizer.collect_features(test_loader)

        known_mask = labels_before_gen >= 0
        metrics_before_gen = DistributionMetrics.compute_all(
            feats_before_gen[known_mask], labels_before_gen[known_mask]
        )
        logger.info(f"  Metrics BEFORE: {metrics_before_gen}")

        # ---- Step E.2: 生成有偏样本增广 ----
        logger.info("  [E.2] 生成有偏样本增广...")
        generator = BiasedImageGenerator(
            feature_library=feature_lib,
            rotation_range=args.gen_rotation_range,
            rotation_steps=args.gen_rotation_steps,
            mixup_num=args.gen_mixup_num,
        )

        gen_stats = generator.generate_all_classes(
            model=model,
            unbiased_feats=unbiased_feats,
            biased_feats=biased_feats,
            biased_paths=biased_paths,
            device=device,
            transform=train_transform,
            logger=logger,
        )

        with open(os.path.join(result_dir, "generation_stats.json"), "w") as f:
            json.dump({str(k): v for k, v in gen_stats.items()}, f, indent=4)

        # 整理生成的特征 (用于微调和可视化)
        generated_feats_dict = {}
        for cid in feature_lib.class_ids:
            bf = biased_feats.get(cid, [])
            uf = unbiased_feats.get(cid, [])
            if bf:
                gen_f = generator.generate_features(cid, bf, uf)
                if gen_f.numel() > 0:
                    generated_feats_dict[cid] = gen_f

        # ---- Step E.2.5: 收集训练集特征+有偏标签 (用于4.3可视化, 须在finetune前) ----
        logger.info("  [E.2.5] 收集训练集特征 (finetune 前, 用于4.3可视化)...")

        # 构建有偏路径集合 (优先用 Phase C 保存的原始判定, 有偏数量更多效果更明显)
        biased_path_set = set()
        bias_split_path = os.path.join(result_dir, "bias_split.pt")
        if os.path.exists(bias_split_path):
            bias_data = torch.load(bias_split_path, map_location="cpu", weights_only=False)
            saved_bp = bias_data.get("biased_paths", {})
            for cid, paths in saved_bp.items():
                for p in paths:
                    biased_path_set.add(p)
            logger.info(f"  加载 Phase C 有偏路径: {len(biased_path_set)} 个")
        else:
            # 回退: 用当前 biased_paths
            for cid, paths in biased_paths.items():
                for p in paths:
                    biased_path_set.add(p)
            logger.info(f"  当前有偏路径: {len(biased_path_set)} 个")

        # 遍历 train_loader, 同时记录特征、标签、原始标签和有偏标签
        model.eval()
        train_feats_list, train_labels_list, train_origs_list, is_biased_train = [], [], [], []
        with torch.no_grad():
            for x, y, orig_labels, paths in train_loader:
                fps = model.extract_fingerprint(x.to(device)).cpu()
                train_feats_list.append(fps)
                train_labels_list.append(y)
                train_origs_list.extend(orig_labels.tolist())
                for p in paths:
                    is_biased_train.append(p in biased_path_set)

        train_feats_before = torch.cat(train_feats_list, dim=0)
        train_labels_before = torch.cat(train_labels_list, dim=0)
        train_origs_before = train_origs_list

        n_biased_total = sum(is_biased_train)
        logger.info(f"  训练集有偏样本数: {n_biased_total}/{len(is_biased_train)}")

        # 增广前开集评估
        logger.info("  增广前开集评估...")
        eval_before = evaluate_openset(model, train_loader, test_loader, device, num_known)
        logger.info(f"  BEFORE finetune → Known={eval_before['known_acc']:.4f}, "
                     f"F1={eval_before['open_f1']:.4f}, "
                     f"NA={eval_before['na']:.4f}, AUROC={eval_before['auroc']:.4f}")

        # ---- Step E.3: 用增广数据微调模型 ----
        logger.info("  [E.3] 增广数据微调模型 (含伪未知拒判训练)...")
        ft_history = finetune_with_augmented_data(
            model=model,
            train_loader=train_loader,
            generated_feats=generated_feats_dict,
            device=device,
            num_known=num_known,
            epochs=20,
            lr=args.lr * 0.1,  # 适度学习率 (提高增益)
            supcon_temperature=0.12,
            logger=logger,
        )

        # ---- Step E.4: 收集微调后的测试集特征 ----
        logger.info("  [E.4] 收集微调后特征...")
        feats_after_gen, labels_after_gen, origs_after_gen = \
            gen_optimizer.collect_features(test_loader)

        known_mask_after = labels_after_gen >= 0
        metrics_after_gen = DistributionMetrics.compute_all(
            feats_after_gen[known_mask_after], labels_after_gen[known_mask_after]
        )
        logger.info(f"  Metrics AFTER: {metrics_after_gen}")

        # ---- Step E.5: 保存与可视化 ----
        logger.info("  [E.5] 生成可视化...")
        feature_lib_final = FeatureLibrary(embed_dim=args.embed_dim)
        feature_lib_final.build_from_model(model, train_loader, device, known_id_map, ID2NAME)
        feature_lib_final.save(os.path.join(result_dir, "feature_library_final.pt"))

        # 加载训练前特征 (Phase A 前保存的)
        pretrain_path = os.path.join(result_dir, "pretrain_feats.pt")
        if os.path.exists(pretrain_path):
            pretrain_data = torch.load(pretrain_path, map_location="cpu", weights_only=False)

            # 三阶段可视化: 训练前 | 训练后+原型 | 增广后+增广★
            logger.info("  生成三阶段可视化...")
            from siamese_viz import plot_pipeline_three_stage
            protos_for_viz = feature_lib.get_prototypes()
            plot_pipeline_three_stage(
                feats_pretrain=pretrain_data["feats"],
                labels_pretrain=pretrain_data["labels"],
                origs_pretrain=pretrain_data["origs"],
                metrics_pretrain=pretrain_data["metrics"],
                feats_trained=feats_before_gen,
                labels_trained=labels_before_gen,
                origs_trained=origs_before_gen,
                metrics_trained=metrics_before_gen,
                feats_final=feats_after_gen,
                labels_final=labels_after_gen,
                origs_final=origs_after_gen,
                metrics_final=metrics_after_gen,
                save_path=os.path.join(result_dir, "tsne_pipeline_three_stage.png"),
                prototypes=protos_for_viz,
                augmented_feats=generated_feats_dict,
            )

        # Phase E 前后对比 (共享 t-SNE + 增广样本星号标注)
        plot_tsne_before_after(
            feats_before_gen, labels_before_gen, origs_before_gen,
            feats_after_gen, labels_after_gen, origs_after_gen,
            save_path=os.path.join(result_dir, "tsne_before_after_generation.png"),
            title_prefix="Biased Sample Generation + Finetune (4.3)",
            metrics_before=metrics_before_gen,
            metrics_after=metrics_after_gen,
            augmented_feats=generated_feats_dict,
        )

        # 4.3 核心可视化: 有偏→增广→优化 (三面板, 有偏/无偏区分+置信椭圆)
        # 使用训练集特征 (有偏标签更丰富: 121个有偏 vs 测试集仅41个)
        logger.info("  生成 4.3 核心可视化 (训练集有偏/无偏 + 置信椭圆)...")

        # 收集训练集特征 (微调后)
        train_feats_after, train_labels_after, train_origs_after = \
            gen_optimizer.collect_features(train_loader)

        # 计算训练集 metrics
        known_mask_train_b = train_labels_before >= 0
        known_mask_train_a = train_labels_after >= 0
        metrics_train_before = DistributionMetrics.compute_all(
            train_feats_before[known_mask_train_b], train_labels_before[known_mask_train_b])
        metrics_train_after = DistributionMetrics.compute_all(
            train_feats_after[known_mask_train_a], train_labels_after[known_mask_train_a])

        plot_biased_generation_effect(
            train_feats_before, train_labels_before, train_origs_before,
            is_biased_train,
            train_feats_after, train_labels_after, train_origs_after,
            save_path=os.path.join(result_dir, "tsne_biased_generation_effect.png"),
            augmented_feats=generated_feats_dict,
            metrics_before=metrics_train_before,
            metrics_after=metrics_train_after,
        )

        # 保存微调后模型
        torch.save(model.state_dict(), os.path.join(result_dir, "model_after_generation.pt"))

        # ---- Step E.6: 增广后开集评估 (对比) ----
        logger.info("  [E.6] 增广后开集评估...")
        eval_after = evaluate_openset(model, train_loader, test_loader, device, num_known)
        logger.info(f"  AFTER finetune  → Known={eval_after['known_acc']:.4f}, "
                     f"F1={eval_after['open_f1']:.4f}, "
                     f"NA={eval_after['na']:.4f}, AUROC={eval_after['auroc']:.4f}")

        # 打印改善总结
        logger.info("\n" + "=" * 60)
        logger.info("[Generation + Finetune Improvement Summary]")
        logger.info("-" * 60)
        logger.info("  [特征分布指标]")
        for key in ["intra_compactness", "inter_separation",
                     "silhouette_approx", "fisher_ratio"]:
            before = metrics_before_gen.get(key, 0)
            after = metrics_after_gen.get(key, 0)
            delta = after - before
            direction = "↓" if key == "intra_compactness" else "↑"
            _ok = (delta < 0) if key == "intra_compactness" else (delta > 0)
            symbol = "BETTER" if _ok else "worse"
            logger.info(f"    {key}: {before:.4f} -> {after:.4f} "
                        f"({direction} {abs(delta):.4f}) [{symbol}]")
        logger.info("-" * 60)
        logger.info("  [开集识别指标]")
        for key, name in [("known_acc", "Known Accuracy"),
                          ("open_f1", "Open F1"),
                          ("na", "NA (Normalized Accuracy)"),
                          ("auroc", "AUROC"),
                          ("aks", "AKS"),
                          ("aus", "AUS")]:
            before = eval_before.get(key, 0)
            after = eval_after.get(key, 0)
            delta = after - before
            _ok = delta >= 0
            symbol = "BETTER" if _ok and delta > 0 else ("same" if delta == 0 else "worse")
            logger.info(f"    {name}: {before:.4f} -> {after:.4f} "
                        f"({'↑' if delta >= 0 else '↓'} {abs(delta):.4f}) [{symbol}]")
        logger.info("=" * 60)

    # =========================================================================
    # 最终评估
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("[Final Evaluation]")
    logger.info("=" * 70)

    final_metrics = evaluate_openset(model, train_loader, test_loader, device, num_known)
    logger.info(f"  Known Accuracy:  {final_metrics['known_acc']:.4f}")
    logger.info(f"  Open F1:         {final_metrics['open_f1']:.4f}")
    logger.info(f"  Open Precision:  {final_metrics['open_precision']:.4f}")
    logger.info(f"  Open Recall:     {final_metrics['open_recall']:.4f}")
    logger.info(f"  AKS:             {final_metrics['aks']:.4f}")
    logger.info(f"  AUS:             {final_metrics['aus']:.4f}")
    logger.info(f"  NA:              {final_metrics['na']:.4f}")
    logger.info(f"  AUROC:           {final_metrics['auroc']:.4f}")

    # 保存最终结果
    with open(os.path.join(result_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    logger.info(f"\nAll results saved to: {result_dir}")


if __name__ == "__main__":
    main()

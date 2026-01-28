"""
训练主程序
包含单轮训练、验证、完整训练循环
"""

from typing import Dict, Optional, Tuple

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from openset_losses import adversarial_margin_loss, total_boundary_loss
from openset_pseudo_unknown import (PseudoUnknownGenerator, generate_pseudo_unknown_batch,
                                    pseudo_unknown_classification_loss)
from openset_proto import (compute_rectified_prototypes, compute_simple_prototypes,
                            compute_class_thresholds_p95, compute_global_threshold_p95)
from openset_evaluator import evaluate_with_proto, evaluate_with_proto_class_tau


# -------------------------
# 单轮训练
# -------------------------
def train_one_epoch(model: nn.Module,
                    loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    lam: float = 1.0,
                    use_pseudo_unknown: bool = False,
                    pseudo_ratio: float = 0.3,
                    use_boundary_loss: bool = False,
                    prototypes: Optional[torch.Tensor] = None,
                    boundary_weights: Tuple[float, float, float] = (1.0, 1.0, 0.5),
                    grad_clip: float = 0.0) -> Dict:
    """
    单轮训练

    Args:
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        device: 设备
        lam: L_T和L_D的权重
        use_pseudo_unknown: 是否使用伪未知类
        pseudo_ratio: 伪未知比例
        use_boundary_loss: 是否使用边界损失
        prototypes: 类原型(边界损失需要)
        boundary_weights: (intra_weight, inter_weight, open_weight)
        grad_clip: 梯度裁剪阈值

    Returns:
        包含各损失的字典
    """
    model.train()
    total_loss_adv = 0.0
    total_loss_boundary = 0.0
    total_loss_pseudo = 0.0
    total_samples = 0

    for x, y, _, _ in loader:
        mask = (y >= 0)  # 训练只用已知类
        if not mask.any():
            continue

        x = x[mask].to(device)
        y = y[mask].to(device)

        # 避免BN报错 (batch size=1)
        if x.size(0) <= 1:
            continue

        # 1. 提取特征和logits
        logit_neg, logit_pos, f_neg, f_pos = model(x, y)

        # 2. 对抗性margin损失 (L_Adv = L_T + λ * L_D)
        loss_adv = adversarial_margin_loss(logit_neg, logit_pos, y, lam)

        # 3. (可选) 边界约束损失
        loss_boundary = 0.0
        if use_boundary_loss and prototypes is not None:
            # 归一化特征
            f_pos_norm = F.normalize(f_pos, dim=1)

            # 计算三类边界损失
            loss_boundary = total_boundary_loss(
                f_pos_norm,
                prototypes,
                y,
                intra_weight=boundary_weights[0],
                inter_weight=boundary_weights[1],
                open_weight=boundary_weights[2]
            )

        # 4. (可选) 伪未知类生成和损失
        loss_pseudo = 0.0
        if use_pseudo_unknown:
            # 生成伪未知特征
            pseudo_feats, _ = generate_pseudo_unknown_batch(
                f_pos, y,
                num_classes=model.cls_neg.weight.size(0),
                ratio=pseudo_ratio
            )

            # 计算伪未知类损失:让伪未知远离所有已知类原型
            if prototypes is not None:
                loss_pseudo = pseudo_unknown_classification_loss(
                    pseudo_feats, prototypes, margin=0.5
                )
                # Apply open_weight to loss_pseudo
                loss_pseudo = loss_pseudo * boundary_weights[2]

        # 总损失
        loss = loss_adv + loss_boundary + loss_pseudo

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 统计
        bs = x.size(0)
        total_loss_adv += float(loss_adv.item()) * bs
        
        # 处理可能的float类型(当不使用边界损失或伪未知时)
        if isinstance(loss_boundary, torch.Tensor):
            total_loss_boundary += float(loss_boundary.item()) * bs
        else:
            total_loss_boundary += float(loss_boundary) * bs
            
        if isinstance(loss_pseudo, torch.Tensor):
            total_loss_pseudo += float(loss_pseudo.item()) * bs
        else:
            total_loss_pseudo += float(loss_pseudo) * bs
            
        total_samples += bs

    return {
        "loss_adv": total_loss_adv / max(total_samples, 1),
        "loss_boundary": total_loss_boundary / max(total_samples, 1),
        "loss_pseudo": total_loss_pseudo / max(total_samples, 1),
        "loss_total": (total_loss_adv + total_loss_boundary + total_loss_pseudo) / max(total_samples, 1)
    }


# -------------------------
# 验证
# -------------------------
@torch.no_grad()
def validate(model: nn.Module,
             train_loader: torch.utils.data.DataLoader,
             test_loader: torch.utils.data.DataLoader,
             device: torch.device,
             num_known: int,
             use_rectified_proto: bool = True,
             use_class_tau: bool = True,
             percentile: float = 95.0) -> Dict:
    """
    验证: 计算原型、阈值,并在测试集上评估

    Args:
        model: 模型
        train_loader: 训练集加载器(用于计算原型)
        test_loader: 测试集加载器
        device: 设备
        num_known: 已知类数量
        use_rectified_proto: 是否使用修正原型
        use_class_tau: 是否使用按类阈值
        percentile: 阈值分位数

    Returns:
        包含原型、阈值和测试指标的字典
    """
    model.eval()

    # 1. 计算原型
    if use_rectified_proto:
        protos = compute_rectified_prototypes(
            model, train_loader, device, num_known, use_pos_branch=True
        )
    else:
        protos = compute_simple_prototypes(
            model, train_loader, device, num_known, use_pos_branch=True
        )

    # 2. 计算阈值
    if use_class_tau:
        tau_c = compute_class_thresholds_p95(
            model, train_loader, device, protos, num_known, percentile
        )

        # 3. 使用按类阈值评估
        metrics = evaluate_with_proto_class_tau(
            model, test_loader, device, protos, tau_c
        )
        metrics["tau_c"] = tau_c
        metrics["use_class_tau"] = True
    else:
        tau = compute_global_threshold_p95(
            model, train_loader, device, protos, percentile
        )

        # 3. 使用全局阈值评估
        metrics = evaluate_with_proto(
            model, test_loader, device, protos, tau
        )
        metrics["tau"] = tau
        metrics["use_class_tau"] = False

    # 添加原型信息
    metrics["prototypes"] = protos.detach().cpu()

    return metrics


# -------------------------
# 完整训练循环
# -------------------------
def train(model: nn.Module,
           train_loader: torch.utils.data.DataLoader,
           test_loader: torch.utils.data.DataLoader,
           optimizer: torch.optim.Optimizer,
           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
           device: torch.device = torch.device("cpu"),
           num_known: int = 7,
           num_epochs: int = 50,
           lam: float = 1.0,
           use_rectified_proto: bool = True,
           use_pseudo_unknown: bool = False,
           use_boundary_loss: bool = False,
           pseudo_ratio: float = 0.3,
           use_class_tau: bool = True,
           percentile: float = 95.0,
           boundary_weights: Tuple[float, float, float] = (1.0, 1.0, 0.5),
           grad_clip: float = 0.0,
           save_path: str = "best_model.pt",
           log_interval: int = 1,
           logger: Optional[logging.Logger] = None) -> Dict:
    """
    完整训练循环

    Args:
        model: 模型
        train_loader: 训练集加载器
        test_loader: 测试集加载器
        optimizer: 优化器
        scheduler: 学习率调度器 (可选)
        device: 设备
        num_known: 已知类数量
        num_epochs: 训练轮数
        lam: L_T和L_D的权重
        use_rectified_proto: 是否使用修正原型
        use_pseudo_unknown: 是否使用伪未知类
        use_boundary_loss: 是否使用边界损失
        pseudo_ratio: 伪未知比例
        use_class_tau: 是否使用按类阈值
        percentile: 阈值分位数
        boundary_weights: 边界损失权重
        grad_clip: 梯度裁剪阈值
        save_path: 模型保存路径
        log_interval: 日志打印间隔
        logger: 日志记录器

    Returns:
        训练历史和最佳模型信息
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    best_na = -1.0
    best_state = None
    history = {
        "loss_total": [],
        "loss_adv": [],
        "loss_boundary": [],
        "loss_pseudo": [],
        "open_f1": [],
        "known_acc": [],
        "open_precision": [],
        "open_recall": [],
        "open_acc": [],
        "aks": [],
        "aus": [],
        "na": [],
        "auroc": []
    }

    # 初始化原型为None
    current_prototypes = None

    for epoch in range(1, num_epochs + 1):
        # 训练:第一轮不使用边界损失(还没有原型),从第二轮开始使用上一轮的原型
        train_losses = train_one_epoch(
            model, train_loader, optimizer, device,
            lam=lam,
            use_pseudo_unknown=use_pseudo_unknown,
            pseudo_ratio=pseudo_ratio,
            use_boundary_loss=use_boundary_loss and (epoch > 1),
            prototypes=current_prototypes,  # 传入上一轮的原型
            boundary_weights=boundary_weights,
            grad_clip=grad_clip
        )

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 验证:计算原型和阈值
        metrics = validate(
            model, train_loader, test_loader, device,
            num_known=num_known,
            use_rectified_proto=use_rectified_proto,
            use_class_tau=use_class_tau,
            percentile=percentile
        )

        # 保存当前原型供下一轮训练使用
        current_prototypes = metrics["prototypes"].to(device)

        # 记录历史
        history["loss_total"].append(train_losses["loss_total"])
        history["loss_adv"].append(train_losses["loss_adv"])
        history["loss_boundary"].append(train_losses["loss_boundary"])
        history["loss_pseudo"].append(train_losses.get("loss_pseudo", 0.0))
        history["open_f1"].append(metrics["open_f1"])
        history["known_acc"].append(metrics["known_acc"])
        history["open_precision"].append(metrics["open_precision"])
        history["open_recall"].append(metrics["open_recall"])
        history["open_acc"].append(metrics["open_acc"])
        history["aks"].append(metrics.get("aks", 0.0))
        history["aus"].append(metrics.get("aus", 0.0))
        history["na"].append(metrics.get("na", 0.0))
        history["auroc"].append(metrics.get("auroc", 0.0))

        # 日志
        if epoch % log_interval == 0:
            logger.info(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"loss={train_losses['loss_total']:.4f} "
                f"(adv={train_losses['loss_adv']:.4f}, "
                f"boundary={train_losses['loss_boundary']:.4f}, "
                f"pseudo={train_losses.get('loss_pseudo', 0.0):.4f}) | "
                f"known_acc={metrics['known_acc']:.4f} | "
                f"open_f1={metrics['open_f1']:.4f} "
                f"(P={metrics['open_precision']:.4f}, "
                f"R={metrics['open_recall']:.4f}) | "
                f"open_acc={metrics['open_acc']:.4f} | "
                f"NA={metrics.get('na', 0.0):.4f} "
                f"(AKS={metrics.get('aks', 0.0):.4f}, "
                f"AUS={metrics.get('aus', 0.0):.4f})"
            )

        # 保存最佳模型
        current_na = metrics.get("na", 0.0)
        if current_na > best_na:
            best_na = current_na
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "open_f1": metrics["open_f1"],
                "known_acc": metrics["known_acc"],
                "open_precision": metrics["open_precision"],
                "open_recall": metrics["open_recall"],
                "open_acc": metrics["open_acc"],
                "aks": metrics.get("aks", 0.0),
                "aus": metrics.get("aus", 0.0),
                "na": metrics.get("na", 0.0),
                "prototypes": metrics["prototypes"],
                "history": history
            }

            if use_class_tau:
                best_state["tau_c"] = metrics["tau_c"]
            else:
                best_state["tau"] = metrics.get("tau", 0.5)

            torch.save(best_state, save_path)
            logger.info(f"[Save] best checkpoint -> {save_path} (best_na={best_na:.4f})")

    logger.info("[Done] Training finished.")
    logger.info(f"Best NA: {best_na:.4f}")

    return history, best_state


# -------------------------
# 测试代码
# -------------------------
if __name__ == "__main__":
    print("训练模块测试")
    print("  - train_one_epoch(): 单轮训练")
    print("  - validate(): 验证")
    print("  - train(): 完整训练循环")

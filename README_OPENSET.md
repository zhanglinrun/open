# 开集识别代码使用说明

## 概述

本项目实现了基于Prototype_enhance_FSCAMR论文的开集识别算法,包含以下核心功能:

1. **Rectified Prototypical Learning (RPL)**: 基于样本典型性的加权原型计算(论文公式4-6)
2. **Pseudo Unknown Generation**: 通过凸组合生成伪未知类,提升拒判能力
3. **Boundary Constraint Loss**: 三类边界约束(类内聚合、类间分离、开放空间约束)
4. **Adversarial Margin Learning**: 双头网络(负margin+正margin)

## 文件结构

```
/home2/lrzhang/pythonProject/yolo/
├── openset_config.py          # 配置和常量
├── openset_data.py            # 数据加载
├── openset_models.py          # 模型定义
├── openset_losses.py          # 损失函数
├── openset_proto.py           # 原型计算
├── openset_pseudo_unknown.py # 伪未知类生成
├── openset_evaluator.py      # 评估函数
├── openset_train.py          # 训练循环
├── openset_main.py          # 入口程序
├── run_experiments.sh        # 批量实验脚本
├── analyze_results.py        # 结果分析脚本
├── train_negpos_openset.py  # 原始代码(备份)
└── train.py                 # 原始FSC代码(保留)
```

## 快速开始

### 1. 基础训练 (Baseline)

使用简单平均原型,不使用任何改进:

```bash
python openset_main.py \
    --data_dir data_cut_10_v2 \
    --epochs 50 \
    --batch_size 64 \
    --save_path baseline.pt
```

### 2. 使用修正原型

激活Rectified Prototypical Learning:

```bash
python openset_main.py \
    --data_dir data_cut_10_v2 \
    --epochs 50 \
    --batch_size 64 \
    --use_rectified_proto \
    --save_path rectified.pt
```

### 3. 添加伪未知类

激活伪未知类生成:

```bash
python openset_main.py \
    --data_dir data_cut_10_v2 \
    --epochs 50 \
    --batch_size 64 \
    --use_pseudo_unknown \
    --pseudo_ratio 0.3 \
    --use_rectified_proto \
    --save_path pseudo_unknown.pt
```

### 4. 添加边界约束损失

激活三类边界损失:

```bash
python openset_main.py \
    --data_dir data_cut_10_v2 \
    --epochs 50 \
    --batch_size 64 \
    --use_boundary_loss \
    --use_rectified_proto \
    --intra_weight 1.0 \
    --inter_weight 1.0 \
    --open_weight 0.5 \
    --save_path boundary_loss.pt
```

### 5. 完整模型 (Full Model)

启用所有改进:

```bash
python openset_main.py \
    --data_dir data_cut_10_v2 \
    --epochs 50 \
    --batch_size 64 \
    --use_pseudo_unknown \
    --pseudo_ratio 0.3 \
    --use_boundary_loss \
    --use_rectified_proto \
    --use_class_tau \
    --intra_weight 1.0 \
    --inter_weight 1.0 \
    --open_weight 0.5 \
    --save_path full_model.pt
```

## 批量实验

运行所有对比实验:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

这将依次运行5个实验:
1. Baseline
2. +Rectified Proto
3. +Pseudo Unknown
4. +Boundary Loss
5. Full Model

## 结果分析

分析实验结果:

```bash
python analyze_results.py
```

这将输出所有实验的对比表格和改进幅度。

## 核心参数说明

### 模型参数

- `--embed_dim`: 嵌入维度(默认256)
- `--mneg`: 负margin(默认0.20),用于transferable features
- `--mpos`: 正margin(默认0.40),用于discriminative features
- `--scale`: softmax温度参数(默认30.0)

### 训练参数

- `--epochs`: 训练轮数(默认50)
- `--batch_size`: 批次大小(默认64)
- `--lr`: 学习率(默认1e-4)
- `--lam`: L_T和L_D的权重比(默认1.0)

### 原型参数

- `--use_rectified_proto`: 使用修正原型(论文公式4-6)
- `--use_class_tau`: 使用按类阈值(否则用全局阈值)
- `--percentile`: 阈值分位数(默认95.0)

### 伪未知类参数

- `--use_pseudo_unknown`: 使用伪未知类生成
- `--pseudo_ratio`: 伪未知比例(默认0.3)

### 边界损失参数

- `--use_boundary_loss`: 使用边界约束损失
- `--intra_weight`: 类内聚合损失权重(默认1.0)
- `--inter_weight`: 类间分离损失权重(默认1.0)
- `--open_weight`: 开放空间约束权重(默认0.5)

## 数据集说明

数据集包含10类船舶图像:

### 已知类 (7类,训练时使用)
- 1: Warship
- 2: Bulk-Carrier
- 3: Oil-Tanker
- 4: Container-Ship
- 5: Cargo-Ship
- 7: Tug
- 8: Vehicles-Carrier

### 未知类 (3类,测试时拒判)
- 0: Aircraft-Carrier
- 6: Passenger-Cruise-Ship
- 9: Blurred

## 评估指标

### 开集识别指标
- **open_f1**: F1分数(主要指标)
- **open_precision**: 精确率
- **open_recall**: 召回率
- **open_acc**: 开集准确率

### 分类指标
- **known_acc**: 已知类分类准确率

### 其他指标
- **auroc**: ROC曲线下面积(在openset_evaluator.py中实现)

## 性能目标

从原始代码的基线:
- open_f1 ≈ 0.018-0.052 (极差)
- open_recall ≈ 0.05-0.60 (几乎识别不到未知类)

改进目标:
- open_f1 > 0.30
- open_recall > 0.50
- open_precision > 0.40
- known_acc > 0.70 (保持已知类性能)

## 超参数调优建议

可以尝试的超参数组合:

### margin调优
```bash
for mneg in 0.1 0.2 0.3; do
    for mpos in 0.3 0.4 0.5; do
        python openset_main.py \
            --mneg $mneg --mpos $mpos \
            --use_rectified_proto \
            --save_path "mneg_${mneg}_mpos_${mpos}.pt"
    done
done
```

### 伪未知比例调优
```bash
for ratio in 0.2 0.3 0.4; do
    python openset_main.py \
        --use_pseudo_unknown \
        --pseudo_ratio $ratio \
        --use_rectified_proto \
        --save_path "pseudo_ratio_${ratio}.pt"
done
```

### 边界损失权重调优
```bash
for open_w in 0.3 0.5 0.7; do
    python openset_main.py \
        --use_boundary_loss \
        --open_weight $open_w \
        --use_rectified_proto \
        --save_path "open_weight_${open_w}.pt"
done
```

## 故障排查

### 问题1: CUDA out of memory
**解决方案**: 减小batch_size或embed_dim
```bash
python openset_main.py --batch_size 32 --embed_dim 128 ...
```

### 问题2: 开集识别效果差(open_f1很低)
**解决方案**:
1. 启用所有改进: `--use_rectified_proto --use_pseudo_unknown --use_boundary_loss`
2. 调整margin: `--mneg 0.15 --mpos 0.35`
3. 增加伪未知比例: `--pseudo_ratio 0.4`
4. 增加训练轮数: `--epochs 100`

### 问题3: 已知类准确率下降太多
**解决方案**:
1. 减小伪未知比例: `--pseudo_ratio 0.2`
2. 降低边界损失权重: `--open_weight 0.3`
3. 调整lambda: `--lam 0.5` (更重视L_T)

## 引用

本实现基于以下论文:
- **Prototype_enhance_FSCAMR.pdf**: Rectified Prototypical Learning for Few-Shot Incremental Learning

## 联系

如有问题,请检查:
1. 数据集路径是否正确
2. CUDA是否可用
3. Python环境依赖是否完整 (torch, torchvision, numpy, scikit-learn)

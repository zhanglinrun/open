#!/bin/bash
# 开集识别对比实验脚本
# 对比不同配置下的性能

echo "=========================================="
echo "开集识别对比实验"
echo "=========================================="

# 通用参数
DATA_DIR="data_cut_10_v2"
EPOCHS=50
BATCH_SIZE=256
LR=1e-4

# 实验1: Baseline (简单平均原型,无伪未知,无边界损失)
echo ""
echo "=========================================="
echo "实验1: Baseline (简单原型)"
echo "=========================================="

python openset_main.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_path "exp1_baseline.pt"

# 实验2: +Rectified Proto (使用修正原型)
echo ""
echo "=========================================="
echo "实验2: +Rectified Proto (修正原型)"
echo "=========================================="

python openset_main.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --use_rectified_proto \
    --save_path "exp2_rectified_proto.pt"

# 实验3: +Pseudo Unknown (添加伪未知类生成)
echo ""
echo "=========================================="
echo "实验3: +Pseudo Unknown (伪未知类)"
echo "=========================================="

python openset_main.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --use_pseudo_unknown \
    --pseudo_ratio 0.3 \
    --use_rectified_proto \
    --save_path "exp3_pseudo_unknown.pt"

# 实验4: +Boundary Loss (添加边界约束损失)
echo ""
echo "=========================================="
echo "实验4: +Boundary Loss (边界约束)"
echo "=========================================="

python openset_main.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --use_boundary_loss \
    --use_rectified_proto \
    --intra_weight 1.0 \
    --inter_weight 1.0 \
    --open_weight 0.5 \
    --save_path "exp4_boundary_loss.pt"

# 实验5: Full Model (全部改进组合)
echo ""
echo "=========================================="
echo "实验5: Full Model (全部改进)"
echo "=========================================="

python openset_main.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --use_pseudo_unknown \
    --pseudo_ratio 0.3 \
    --use_boundary_loss \
    --use_rectified_proto \
    --use_class_tau \
    --intra_weight 1.0 \
    --inter_weight 1.0 \
    --open_weight 0.5 \
    --save_path "exp5_full_model.pt"

echo ""
echo "=========================================="
echo "所有实验完成!"
echo "=========================================="
echo "模型文件:"
ls -lh exp*.pt

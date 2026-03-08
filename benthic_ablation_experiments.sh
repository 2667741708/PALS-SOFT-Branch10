#!/bin/bash
# ==============================================================================
# Benthic消融实验方案 - 目标: 测试集准确率 > 77%
# ==============================================================================
# 策略: 基于最小修改原则，系统探索关键超参数和组件
# 硬件: 2x RTX 5080, 每卡并行2个实验
# ==============================================================================

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_cuda128_whm

# 基础配置
SCRIPT="结合投影_众包_自节点.py"
DATASET="Benthic"
TRAIN_ROOT="./data"
OUT_DIR="./topology_daes"
NETWORK="R50"
EPOCHS=50
BATCH_SIZE=32
LR=0.05
WD=5e-4
LR_SCHEDULER="step"
LR_DECAY_RATE=0.2
LPI=10
SEEDS="1 2 3"
DETAILED_LOG="--detailed_log"

# 实验记录文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./experiment_logs/benthic_${TIMESTAMP}"
mkdir -p ${LOG_DIR}

echo "===================================================================="
echo "🚀 Benthic消融实验 - 启动时间: ${TIMESTAMP}"
echo "🎯 目标: 测试集准确率 > 77%"
echo "📁 日志目录: ${LOG_DIR}"
echo "===================================================================="

# ==============================================================================
# 阶段1: 基线实验 (2个实验)
# ==============================================================================
echo ""
echo "📊 阶段1: 基线实验"
echo "--------------------------------------------------------------------"

# Exp1: 用户原始配置（baseline）
EXP_NAME="Benthic_Ablation/Stage1_Baseline/exp01_original_k5_d1.0_lsr0.0"
echo "[Exp1] Baseline: k_val=5, delta=1.0, lsr=0.0, topology_daes"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 0 \
    --k_val 5 \
    --history_len 100 \
    --knn_heads 1 \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp01.log 2>&1 &

sleep 2

# Exp2: 启用软传播 + 模型融合（关键改进）
EXP_NAME="Benthic_Ablation/Stage1_Baseline/exp02_soft_prop_model_fuse"
echo "[Exp2] +Soft Propagation +Model Fusion"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 0 \
    --k_val 5 \
    --history_len 100 \
    --knn_heads 1 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --knn1_soft_prop_eps 1e-8 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp02.log 2>&1 &

sleep 5

# ==============================================================================
# 阶段2: KNN参数消融 (4个实验)
# ==============================================================================
echo ""
echo "📊 阶段2: KNN参数消融"
echo "--------------------------------------------------------------------"

# Exp3: k_val=10 + soft_prop
EXP_NAME="Benthic_Ablation/Stage2_KNN/exp03_k10_soft_prop"
echo "[Exp3] k_val=10 (更多邻居) +Soft Prop"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 1 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 1 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp03.log 2>&1 &

sleep 2

# Exp4: k_val=15 + soft_prop
EXP_NAME="Benthic_Ablation/Stage2_KNN/exp04_k15_soft_prop"
echo "[Exp4] k_val=15 (最大邻域) +Soft Prop"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 1 \
    --k_val 15 \
    --history_len 100 \
    --knn_heads 1 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp04.log 2>&1 &

sleep 2

# Exp5: delta=0.75 + k=10
EXP_NAME="Benthic_Ablation/Stage2_KNN/exp05_delta0.75_k10"
echo "[Exp5] delta=0.75 (更严格筛选) k=10"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 0.75 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 0 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 1 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp05.log 2>&1 &

sleep 2

# Exp6: lsr=0.1 + k=10
EXP_NAME="Benthic_Ablation/Stage2_KNN/exp06_lsr0.1_k10"
echo "[Exp6] lsr=0.1 (标签平滑) k=10"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.1 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 0 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 1 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp06.log 2>&1 &

sleep 10

# ==============================================================================
# 阶段3: 组合优化 (6个实验)
# ==============================================================================
echo ""
echo "📊 阶段3: 组合优化"
echo "--------------------------------------------------------------------"

# Exp7: lsr=0.2 + k=10
EXP_NAME="Benthic_Ablation/Stage3_Combo/exp07_lsr0.2_k10"
echo "[Exp7] lsr=0.2 (更强平滑) k=10"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.2 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 1 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 1 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp07.log 2>&1 &

sleep 2

# Exp8: knn_heads=2 + k=10
EXP_NAME="Benthic_Ablation/Stage3_Combo/exp08_heads2_k10"
echo "[Exp8] knn_heads=2 (多头注意力) k=10"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 1 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 2 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp08.log 2>&1 &

sleep 2

# Exp9: gamma=1.0 + k=10
EXP_NAME="Benthic_Ablation/Stage3_Combo/exp09_gamma1.0_k10"
echo "[Exp9] topology_rel_gamma=1.0 (低惩罚) k=10"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 0 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 1 \
    --topology_rel_gamma 1.0 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp09.log 2>&1 &

sleep 2

# Exp10: gamma=3.0 + k=10
EXP_NAME="Benthic_Ablation/Stage3_Combo/exp10_gamma3.0_k10"
echo "[Exp10] topology_rel_gamma=3.0 (高惩罚) k=10"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 0 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 1 \
    --topology_rel_gamma 3.0 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp10.log 2>&1 &

sleep 2

# Exp11: 更早学习率衰减 + lsr=0.1
EXP_NAME="Benthic_Ablation/Stage3_Combo/exp11_early_decay25_lsr0.1_k10"
echo "[Exp11] lr_decay_epochs=25 + lsr=0.1 k=10"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.1 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 25 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 1 \
    --k_val 10 \
    --history_len 100 \
    --knn_heads 1 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp11.log 2>&1 &

sleep 2

# Exp12: 综合最优配置
EXP_NAME="Benthic_Ablation/Stage3_Combo/exp12_best_combo_k15_lsr0.1_delta0.75_heads2"
echo "[Exp12] 综合优化: k=15, lsr=0.1, delta=0.75, heads=2"
nohup python ${SCRIPT} \
    --train_root ${TRAIN_ROOT} \
    --dataset ${DATASET} \
    --lpi ${LPI} \
    --out ${OUT_DIR} \
    --exp_name ${EXP_NAME} \
    --network ${NETWORK} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 0.75 \
    --lsr 0.1 \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_decay_epochs 30 \
    --lr_decay_rate ${LR_DECAY_RATE} \
    --seeds ${SEEDS} \
    --cuda_dev 1 \
    --k_val 15 \
    --history_len 100 \
    --knn_heads 2 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --enable_knn1_model_fuse \
    ${DETAILED_LOG} \
    > ${LOG_DIR}/exp12.log 2>&1 &

echo ""
echo "===================================================================="
echo "✅ 所有12个实验已提交"
echo "📊 并行配置: GPU0运行6个, GPU1运行6个"
echo "📝 日志位置: ${LOG_DIR}/exp*.log"
echo "===================================================================="
echo ""
echo "💡 监控命令:"
echo "  - 查看特定实验: tail -f ${LOG_DIR}/exp01.log"
echo "  - 查看GPU使用: watch -n 1 nvidia-smi"
echo "  - 统计运行中实验: ps aux | grep 结合投影_众包_自节点.py | grep -v grep | wc -l"
echo ""
echo "⏰ 预计完成时间: ~2-3小时"
echo "===================================================================="
echo ""
echo "📄 日志目录已保存到: ${LOG_DIR}"

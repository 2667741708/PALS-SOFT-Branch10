#!/bin/bash
# =============================================================================
# 众包数据集通用参数验证消融实验调度脚本
# 目标数据集: Plankton, Treeversity, Benthic
# 已知最优基线参数: delta=1.0, lsr=0.0, epochs=100, k_val=5, knn_heads=1
# 核心消融变量:
#   1. enable_knn1_model_fuse (ON vs OFF) -> 模型融合是否有利
#   2. lr_scheduler (cosine vs step)
#   3. lpi (3 vs 10) -> 标注数量对性能的影响
# =============================================================================

PYTHON="/home/c201/miniconda3/envs/torch_rtx5080_pals/bin/python"
BASE_DIR="/home/c201/公共/whm/PALS-SOFT/单流拓扑共识"
SCRIPT="结合投影_众包_自节点.py"
LOG_DIR="./topology_daes/crowdsource_ablation_logs"

# ===========================================================================
# 通用基线参数 (所有数据集统一使用这些参数)
# ===========================================================================
COMMON_BASE="--out ./topology_daes --seeds 1 2 3 --lsr 0.0 --detailed_log \
--wd 5e-4 --consistency_weight 1.0 \
--sim_mode_1 topology_daes --sim_mode_2 topology_daes --num_workers 4 \
--k_val 5 --history_len 100 --knn_heads 1 --delta 1.0"

# 数据集专属参数
PLANKTON_SPEC="--dataset Plankton --train_root ./data --network R50 --epochs 100 --batch_size 32 --lr 0.05"
TREEVERSITY_SPEC="--dataset Treeversity --train_root ./data --network R50 --epochs 100 --batch_size 32 --lr 0.05"
BENTHIC_SPEC="--dataset Benthic --train_root ./data --network R50 --epochs 100 --batch_size 32 --lr 0.05"

cd $BASE_DIR || exit 1
mkdir -p $LOG_DIR

# ===========================================================================
# GPU0 任务队列: Treeversity + Plankton 基线消融
# ===========================================================================
run_gpu0() {
    echo "=== [GPU0] Treeversity & Plankton Baseline Ablation ==="

    # --- Treeversity ---
    # T1: 基线 (lpi=10, cosine, 无模型融合)
    $PYTHON $SCRIPT $COMMON_BASE $TREEVERSITY_SPEC --cuda_dev 0 --lpi 10 \
        --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/Treeversity/Universal/T1_LPI10_Cosine_NoFuse

    # T2: 模型融合消融 (lpi=10, cosine, 开启模型融合)
    $PYTHON $SCRIPT $COMMON_BASE $TREEVERSITY_SPEC --cuda_dev 0 --lpi 10 \
        --lr_scheduler cosine --enable_knn1_model_fuse \
        --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8 \
        --exp_name 结合投影_众包_自节点/Treeversity/Universal/T2_LPI10_Cosine_WithFuse

    # T3: 调度器消融 (lpi=10, step, 无模型融合)
    $PYTHON $SCRIPT $COMMON_BASE $TREEVERSITY_SPEC --cuda_dev 0 --lpi 10 \
        --lr_scheduler step \
        --exp_name 结合投影_众包_自节点/Treeversity/Universal/T3_LPI10_Step_NoFuse

    # T4: lpi 消融 (lpi=3, cosine, 无模型融合) - 验证标注数量影响
    $PYTHON $SCRIPT $COMMON_BASE $TREEVERSITY_SPEC --cuda_dev 0 --lpi 3 \
        --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/Treeversity/Universal/T4_LPI3_Cosine_NoFuse

    # --- Plankton ---
    # P1: 基线 (lpi=10, cosine, 无模型融合)
    $PYTHON $SCRIPT $COMMON_BASE $PLANKTON_SPEC --cuda_dev 0 --lpi 10 \
        --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/Plankton/Universal/P1_LPI10_Cosine_NoFuse

    # P2: 模型融合消融 (lpi=10, cosine, 开启模型融合)
    $PYTHON $SCRIPT $COMMON_BASE $PLANKTON_SPEC --cuda_dev 0 --lpi 10 \
        --lr_scheduler cosine --enable_knn1_model_fuse \
        --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8 \
        --exp_name 结合投影_众包_自节点/Plankton/Universal/P2_LPI10_Cosine_WithFuse

    # P3: 调度器消融 (lpi=10, step, 无模型融合)
    $PYTHON $SCRIPT $COMMON_BASE $PLANKTON_SPEC --cuda_dev 0 --lpi 10 \
        --lr_scheduler step \
        --exp_name 结合投影_众包_自节点/Plankton/Universal/P3_LPI10_Step_NoFuse

    # P4: lpi 消融 (lpi=3, cosine, 无融合) 
    $PYTHON $SCRIPT $COMMON_BASE $PLANKTON_SPEC --cuda_dev 0 --lpi 3 \
        --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/Plankton/Universal/P4_LPI3_Cosine_NoFuse
}

# ===========================================================================
# GPU1 任务队列: Benthic 全消融 (fold2 测试集)
# ===========================================================================
run_gpu1() {
    echo "=== [GPU1] Benthic fold2 Full Ablation ==="

    # B1: 基线 (lpi=10, cosine, 无模型融合)
    $PYTHON $SCRIPT $COMMON_BASE $BENTHIC_SPEC --cuda_dev 1 --lpi 10 \
        --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/Benthic/Universal/B1_LPI10_Cosine_NoFuse

    # B2: 模型融合消融 (启用模型融合, 核心消融对比)
    $PYTHON $SCRIPT $COMMON_BASE $BENTHIC_SPEC --cuda_dev 1 --lpi 10 \
        --lr_scheduler cosine --enable_knn1_model_fuse \
        --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8 \
        --exp_name 结合投影_众包_自节点/Benthic/Universal/B2_LPI10_Cosine_WithFuse

    # B3: 调度器消融 (step)
    $PYTHON $SCRIPT $COMMON_BASE $BENTHIC_SPEC --cuda_dev 1 --lpi 10 \
        --lr_scheduler step \
        --exp_name 结合投影_众包_自节点/Benthic/Universal/B3_LPI10_Step_NoFuse

    # B4: lpi 消融 (lpi=3, LPI=3代表最少3人投票，噪声最大)
    $PYTHON $SCRIPT $COMMON_BASE $BENTHIC_SPEC --cuda_dev 1 --lpi 3 \
        --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/Benthic/Universal/B4_LPI3_Cosine_NoFuse

    # B5: lpi=3 + 模型融合 (双消融交叉对比)
    $PYTHON $SCRIPT $COMMON_BASE $BENTHIC_SPEC --cuda_dev 1 --lpi 3 \
        --lr_scheduler cosine --enable_knn1_model_fuse \
        --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8 \
        --exp_name 结合投影_众包_自节点/Benthic/Universal/B5_LPI3_Cosine_WithFuse
}

echo "==================================================================="
echo "启动众包数据集通用消融实验 (GPU0: Treeversity+Plankton, GPU1: Benthic)"
echo "实验日志目录: $LOG_DIR"
echo "==================================================================="

nohup bash -c "$(declare -f run_gpu0); run_gpu0" > $LOG_DIR/gpu0_treeversity_plankton.log 2>&1 &
nohup bash -c "$(declare -f run_gpu1); run_gpu1" > $LOG_DIR/gpu1_benthic.log 2>&1 &

echo "GPU0 PID: $!"
echo "所有任务已提交至后台! 查看进度："
echo "  tail -f $LOG_DIR/gpu0_treeversity_plankton.log"
echo "  tail -f $LOG_DIR/gpu1_benthic.log"

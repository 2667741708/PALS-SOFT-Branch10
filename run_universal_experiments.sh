#!/bin/bash
# =========================================================================
# 通用实验调度脚本 v2 — CUB200 vs Benthic 跨数据集一致性验证
# 核心约束：enable_knn1_model_fuse 和 enable_knn1_soft_prop 两个参数
# 在所有数据集中保持完全相同的设置。
# lsr 统一固定为 0.0 以确保结果可比性。
# =========================================================================

PYTHON="/home/c201/miniconda3/envs/torch_rtx5080_pals/bin/python"
BASE_DIR="/home/c201/公共/whm/PALS-SOFT/单流拓扑共识"
SCRIPT="结合投影_众包_自节点.py"

# =================================================================
# 🔑 关键统一参数（所有数据集保持完全相同的这两个开关）
#   --enable_knn1_soft_prop     : 第二阶段 KNN 输入为软分布（非独热）
#   不使用 --enable_knn1_model_fuse : 避免模型融合干扰纯拓扑基线
#   --lsr 0.0                   : 统一关闭标签平滑，各数据集保持一致
# =================================================================
UNIFIED_FLAGS="--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8 --lsr 0.0"

# --- CUB200 专属公共参数 (pr/nr 用于合成噪声，去除 lpi) ----------
CUB_COMMON="--dataset CUB200 --train_root ./data --pr 0.05 --nr 0.2 --out ./topology_daes \
--batch_size 64 --lr 0.05 --wd 5e-4 --consistency_weight 1.0 --seeds 1 2 3 \
--detailed_log --network R18 --epochs 250 --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
--num_workers 4 --k_val 15 --history_len 15 --delta 0.5"

# --- Benthic 专属公共参数 (使用 lpi 和 knn_heads) ----------------
# 注：Benthic 的 lpi 依然保留，它是众包专属的迭代机制
BEN_COMMON="--dataset Benthic --train_root ./data --lpi 10 --out ./topology_daes \
--batch_size 32 --lr 0.05 --wd 5e-4 --consistency_weight 1.0 --seeds 1 2 3 \
--detailed_log --network R50 --epochs 100 --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
--num_workers 4 --k_val 5 --history_len 100 --knn_heads 1 --delta 1.0 --lsr 0.0"

cd $BASE_DIR || exit 1

run_gpu0() {
    echo "=== GPU0: CUB200 实验 (Cosine + Soft Prop, 无 Model Fuse) ==="
    # C1: CUB200 Cosine 调度
    $PYTHON $SCRIPT $CUB_COMMON $UNIFIED_FLAGS --cuda_dev 0 --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2e250/C1_SoftProp_Cosine_NoModelFuse

    # C2: CUB200 Step 调度
    $PYTHON $SCRIPT $CUB_COMMON $UNIFIED_FLAGS --cuda_dev 0 --lr_scheduler step \
        --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2e250/C2_SoftProp_Step_NoModelFuse
}

run_gpu1() {
    echo "=== GPU1: Benthic fold2 实验 (Cosine + Soft Prop, 无 Model Fuse，验证通用性) ==="
    # B1: Benthic fold2 Cosine 调度 (与 CUB200 使用相同两个开关)
    $PYTHON $SCRIPT $BEN_COMMON $UNIFIED_FLAGS --cuda_dev 1 --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/Benthic/fold2LPI10/Universal_SoftProp_Cosine

    # B2: Benthic fold2 Step 调度
    $PYTHON $SCRIPT $BEN_COMMON $UNIFIED_FLAGS --cuda_dev 1 --lr_scheduler step \
        --exp_name 结合投影_众包_自节点/Benthic/fold2LPI10/Universal_SoftProp_Step
}

echo "Submitting Universal Cross-Dataset experiments..."
mkdir -p ./topology_daes

nohup bash -c "$(declare -f run_gpu0); run_gpu0" > ./topology_daes/universal_gpu0.log 2>&1 &
nohup bash -c "$(declare -f run_gpu1); run_gpu1" > ./topology_daes/universal_gpu1.log 2>&1 &

echo "Tasks submitted! Logs:"
echo "  GPU0 (CUB200):  ./topology_daes/universal_gpu0.log"
echo "  GPU1 (Benthic): ./topology_daes/universal_gpu1.log"

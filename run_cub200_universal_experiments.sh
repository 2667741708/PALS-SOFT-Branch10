#!/bin/bash
# -------------------------------------------------------------------------
# CUB200 通用鲁棒方案消融实验并行调度脚本 (pr=0.05, nr=0.2)
# 核心验证：软传播反哺是否对于 CUB200 合成强噪同样为通用最优解 (lsr 0.0, lpi 10)
# -------------------------------------------------------------------------

PYTHON="/home/c201/miniconda3/envs/torch_rtx5080_pals/bin/python"
SCRIPT="结合投影_众包_自节点.py"
BASE_DIR="/home/c201/公共/whm/PALS-SOFT/单流拓扑共识"

# 全局通用参数：对齐 Benthic 的强噪标配，保证架构一致性
# 关键：lpi=10, lsr=0.0, k_val=15, delta=0.5 -- 这些参数统一用于 CUB200 和 Benthic
COMMON="--dataset CUB200 --train_root ./data --lpi 10 --pr 0.05 --nr 0.2 --out ./topology_daes \
--batch_size 64 --lr 0.05 --wd 5e-4 --consistency_weight 1.0 --seeds 1 2 3 --lsr 0.0 \
--detailed_log --network R18 --epochs 250 --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
--num_workers 4 --k_val 15 --history_len 15 --delta 0.5"

cd $BASE_DIR || exit 1

run_gpu0() {
    echo "=== GPU0: 控制组 (关闭软反哺) Cosine vs Step ==="

    # Branch 1: Cosine, No SoftProp (控制组基准)
    $PYTHON $SCRIPT $COMMON --cuda_dev 0 --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2e250/B1_Cosine_NoSoft

    # Branch 2: Step, No SoftProp
    $PYTHON $SCRIPT $COMMON --cuda_dev 0 --lr_scheduler step \
        --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2e250/B2_Step_NoSoft
}

run_gpu1() {
    echo "=== GPU1: 鲁棒组 (开启软反哺) Cosine vs Step ==="

    # 软反哺核心参数
    SOFT_ARGS="--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8"

    # Branch 3: Cosine + SoftProp (核心探究：看是否跨数据集通用)
    $PYTHON $SCRIPT $COMMON $SOFT_ARGS --cuda_dev 1 --lr_scheduler cosine \
        --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2e250/B3_Cosine_Soft

    # Branch 4: Step + SoftProp (双重兜底)
    $PYTHON $SCRIPT $COMMON $SOFT_ARGS --cuda_dev 1 --lr_scheduler step \
        --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2e250/B4_Step_Soft
}

echo "Submitting Universal CUB200 ablation tasks to background..."
mkdir -p ./topology_daes

nohup bash -c "$(declare -f run_gpu0); run_gpu0" > ./topology_daes/cub200_univ_gpu0.log 2>&1 &
nohup bash -c "$(declare -f run_gpu1); run_gpu1" > ./topology_daes/cub200_univ_gpu1.log 2>&1 &

echo "Tasks submitted! Logs: ./topology_daes/cub200_univ_gpu0.log and cub200_univ_gpu1.log"

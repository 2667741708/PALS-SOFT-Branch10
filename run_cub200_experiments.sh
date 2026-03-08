# -------------------------------------------------------------------------
# CUB200 消融实验并行调度脚本 (pr=0.05, nr=0.2)
# 目标：双卡 5080 测试 250 epoch 下 Cosine vs Step 以及 通用鲁棒方案性能
# -------------------------------------------------------------------------

# 环境与相对路径配置
PYTHON="/home/c201/miniconda3/envs/torch_rtx5080_pals/bin/python"
SCRIPT="结合投影_众包_自节点.py"
BASE_DIR="/home/c201/公共/whm/PALS-SOFT/单流拓扑共识"

# 数据集基础公用参数 (彻底剔除 LPI，固定 pr=0.05, nr=0.2, lsr=0.0)
COMMON="--dataset CUB200 --train_root ./data --pr 0.05 --nr 0.2 --out ./topology_daes --batch_size 64 --lr 0.05 --wd 5e-4 --consistency_weight 1.0 --seeds 1 2 3 --lsr 0.0 --detailed_log --network R18 --epochs 250 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --num_workers 4"

cd $BASE_DIR || exit 1

run_gpu0() {
    echo "=========================================================="
    echo "Starting GPU0: Baseline Check (Cosine vs Step, k=15, del=0.5)"
    echo "=========================================================="
    
    # Branch 1: Baseline Cosine
    CMD1="$PYTHON $SCRIPT $COMMON --cuda_dev 0 --lr_scheduler cosine --k_val 15 --history_len 15 --delta 0.5 --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2/B1_Baseline_Cosine_k15_del0.5"
    echo "Running: $CMD1"
    eval $CMD1
    
    # Branch 2: Baseline Step
    CMD2="$PYTHON $SCRIPT $COMMON --cuda_dev 0 --lr_scheduler step --k_val 15 --history_len 15 --delta 0.5 --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2/B2_Baseline_Step_k15_del0.5"
    echo "Running: $CMD2"
    eval $CMD2
}

run_gpu1() {
    echo "=========================================================="
    echo "Starting GPU1: Universal Robust Scheme (k=5, del=0.75, soft_prop)"
    echo "=========================================================="
    
    # Branch 3: Robust Cosine
    CMD3="$PYTHON $SCRIPT $COMMON --cuda_dev 1 --lr_scheduler cosine --k_val 5 --history_len 15 --delta 0.75 --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8 --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2/B3_Robust_Cosine_k5_del0.75_softprop"
    echo "Running: $CMD3"
    eval $CMD3
    
    # Branch 4: Robust Step
    CMD4="$PYTHON $SCRIPT $COMMON --cuda_dev 1 --lr_scheduler step --k_val 5 --history_len 15 --delta 0.75 --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --knn1_soft_prop_eps 1e-8 --exp_name 结合投影_众包_自节点/CUB200/pr0.05nr0.2/B4_Robust_Step_k5_del0.75_softprop"
    echo "Running: $CMD4"
    eval $CMD4
}

echo "Submitting CUB200 ablation tasks to background..."
mkdir -p ./topology_daes

nohup bash -c "$(declare -f run_gpu0); run_gpu0" > ./topology_daes/cub200_gpu0.log 2>&1 &
nohup bash -c "$(declare -f run_gpu1); run_gpu1" > ./topology_daes/cub200_gpu1.log 2>&1 &
echo "Tasks submitted successfully! Logs will be saved to ./topology_daes/cub200_gpu0.log and cub200_gpu1.log."

#!/bin/bash
# Branch 3: Agreement-based Global Curriculum (基于 结合投影_众包_自节点_加权和.py) — GPU 1
# 仅执行用户指定的两条命令

cd "$(dirname "$0")"

PYTHON="/home/c201/miniconda3/envs/torch_rtx5080_pals/bin/python"
SCRIPT="结合投影_众包_自节点_加权和.py"
GPU=1

echo "=== [GPU$GPU] Branch3: CIFAR100 (500 epochs, Global Curriculum) ==="
$PYTHON "$SCRIPT" \
    --dataset CIFAR100 \
    --out topology_daes \
    --exp_name "Branch3/CIFAR100/pr0.05nr0.5e500top_top_hl15_exp6_knn1_geofuse" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.5 --epochs 500 --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --fusion_mode weighted_sum --consensus_power 2.0 \
    --cuda_dev $GPU

echo "=== [GPU$GPU] Branch3: CIFAR10 (100 epochs, Global Curriculum) ==="
$PYTHON "$SCRIPT" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "Branch3/CIFAR10/pr0.5nr0.3e100topdaes_topdaes_hl15_kh1_fuse" \
    --pr 0.5 --nr 0.3 --epochs 100 --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --knn_heads 1 --seeds 1 2 3 --enable_knn1_model_fuse \
    --fusion_mode weighted_sum --consensus_power 2.0 \
    --cuda_dev $GPU

echo "=== [GPU$GPU] Branch3: All experiments done! ==="

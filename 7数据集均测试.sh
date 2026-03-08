#!/bin/bash

# 遇到错误时继续执行下一个实验（如果你希望报错就停止，可以取消下面这行的注释）
# set -e

echo "=============================================================================="
echo "开始执行所有极端噪声下的脚本实验..."
echo "=============================================================================="

# ============================================================================
# 第一部分：3 个众包数据集 (Treeversity, Benthic, Plankton)
# ============================================================================
echo ">>> 开始执行众包数据集实验..."

# --- Treeversity ---
echo "Running Treeversity - Exp 1"
python 结合投影_众包_自节点.py \
    --dataset Treeversity \
    --lpi 10 \
    --out ./topology_daes \
    --exp_name 结合投影_众包_自节点/Treeversity/LPI10_Run_1del1_exp_top \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --sim_mode_1 topology \
    --sim_mode_2 topology \
    --delta 0.75 \
    --lr_scheduler step \
    --lr_decay_epochs 60 \
    --lr_decay_rate 0.2 \
    --seeds 1 \
    --detailed_log

echo "Running Treeversity - Exp 2"
python 结合投影_众包_自节点.py \
    --dataset Treeversity \
    --lpi 10 \
    --out ./topology_daes \
    --exp_name 结合投影_众包_自节点/Treeversity/LPI10_Run_1del1_top_daes_top_daesdel_1.0_lsr0.0_k5 \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler step \
    --lr_decay_epochs 60 \
    --lr_decay_rate 0.2 \
    --seeds 1 2 3 \
    --detailed_log \
    --cuda_dev 0 \
    --k_val 5

# --- Benthic ---
echo "Running Benthic - Exp 1"
python 结合投影_众包_自节点.py \
    --train_root ./data \
    --dataset Benthic \
    --lpi 3 \
    --out ./topology_daes \
    --exp_name 结合投影_众包_自节点/Benthic/LPI3/del1.0_lsr0.0k5_单头_可投到众包_hl100 \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler step \
    --lr_decay_epochs 60 \
    --lr_decay_rate 0.2 \
    --seeds 1 2 3 \
    --detailed_log \
    --cuda_dev 1 \
    --k_val 5 \
    --history_len 100 \
    --knn_heads 1 

echo "Running Benthic - Exp 2 (fold2)"
python 结合投影_众包_自节点.py \
    --train_root ./data \
    --dataset Benthic \
    --lpi 3 \
    --out ./topology_daes \
    --exp_name 结合投影_众包_自节点/Benthic/LPI3/del1.0_lsr0.0k5_单头_可投到众包_hl100_fold2 \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --sim_mode_1 topology_daes \
    --sim_mode_2 topology_daes \
    --delta 1.0 \
    --lsr 0.0 \
    --lr_scheduler step \
    --lr_decay_epochs 60 \
    --lr_decay_rate 0.2 \
    --seeds 1 2 3 \
    --detailed_log \
    --cuda_dev 1 \
    --k_val 5 \
    --history_len 100 \
    --knn_heads 1 

# --- Plankton ---
echo "Running Plankton - Exp 1"
python 结合投影_众包_自节点.py \
    --dataset Plankton \
    --lpi 10 \
    --out ./topology_daes \
    --exp_name 结合投影_众包_自节点/Plankton/LPI10_Run_1del1_exp_top \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --sim_mode_1 topology \
    --sim_mode_2 topology \
    --delta 0.75 \
    --lr_scheduler step \
    --lr_decay_epochs 60 \
    --lr_decay_rate 0.2 \
    --seeds 1 \
    --detailed_log


# ============================================================================
# 第二部分：4 个普通数据集 (CUB200, CIFAR10, CIFAR100, CIFAR100H)
# ============================================================================
echo ">>> 开始执行普通数据集极端噪声实验..."

# --- CUB200 ---
echo "Running CUB200"
python "结合投影.py" \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 \
    --nr 0.0 \
    --out ./topology_daes \
    --exp_name 结合投影/CUB200pr0.05nr0.0e250exp_topdaes_hl15_del0.5lsr0.5_123_cosine \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.5 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs 250 \
    --cuda_dev 0 \
    --sim_mode_1 exp \
    --sim_mode_2 topology_daes \
    --num_workers 4 \
    --k_val 15 \
    --history_len 15 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --knn1_soft_prop_eps 1e-8 \
    --enable_knn1_model_fuse

# --- CIFAR100 ---
echo "Running CIFAR100 - [1] EXP6 only (baseline)"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR100 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.5e500top_top_hl15_exp6_knn1_geofuse" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.5 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0

echo "Running CIFAR100 - [2] EXP8 only"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR100 \
    --out topology_daes \
    --exp_name "三方共识更新_exp9_knn1_softprop/pr0.05nr0.5e500top_top_hl15_exp8_knn1_softprop_lsr0.1geofuse23" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.5 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.1 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --enable_knn1_model_fuse --seeds 2 3

echo "Running CIFAR100 - [3] EXP6 + EXP8"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR100 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.5e500top_top_hl15_exp6+8_geofuse+softprop" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.5 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

echo "Running CIFAR100 - [4] 100 epochs EXP6 only"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR100 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.5e100top_top_hl15_exp6_knn1_geofuse" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.5 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0

echo "Running CIFAR100 - [5] 100 epochs EXP8 only"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR100 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.5e100top_top_hl15_exp8_knn1_softprop" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.5 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

echo "Running CIFAR100 - [6] 100 epochs EXP6 + EXP8 (daes_结合)"
python "三方共识更新_exp10_knn1_softprop_topo_daes_结合.py" \
    --dataset CIFAR100 \
    --out topology_daes \
    --exp_name "三方共识更新_exp10_knn1_softprop_topo_daes_结合/pr0.05nr0.5e100topdaes_topdaes_hl15_geofuse+softprop23lsr0.5" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.5 --epochs 100 \
    --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.5 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --enable_knn1_model_fuse --seeds 2 3

# --- CIFAR10 ---
echo "Running CIFAR10 - [7] EXP6 only"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.4e100top_top_hl15_exp6_knn1_geofuse" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.4 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0

echo "Running CIFAR10 - [8] EXP8 only"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.4e500top_top_hl15_exp8_knn1_softprop" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.4 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

echo "Running CIFAR10 - [9] EXP6 + EXP8"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.4e500top_top_hl15_exp6+8_geofuse+softprop" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.4 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

echo "Running CIFAR10 - [10] 100 epochs EXP6 only"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.4e100top_top_hl15_exp6_knn1_geofuse" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.4 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0

echo "Running CIFAR10 - [11] 100 epochs EXP8 only"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "三方共识更新/pr0.05nr0.4e100top_top_hl15_exp8_knn1_softprop" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.05 --nr 0.4 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

echo "Running CIFAR10 - [12] 300 epochs EXP6 + EXP8"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "三方共识更新/cifar10pr0.5nr0.3e300top_top_hl15_exp6+8_geofuse+softprop" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.5 --nr 0.3 --epochs 300 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

echo "Running CIFAR10 - [Extra] daes_结合"
python "三方共识更新_exp10_knn1_softprop_topo_daes_结合.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "三方共识更新_exp10_knn1_softprop_topo_daes_结合/cifar10pr0.5nr0.3e100topdaes_topdaes_geofuse+softprop_lsr0.2" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.5 --nr 0.3 --epochs 100 \
    --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --seeds 1 2 3 \
    --knn1_soft_prop_eps 1e-8 \
    --enable_knn1_model_fuse

# --- CIFAR100H ---
echo "Running CIFAR100H"
python "三方共识更新_exp9_knn1_softprop.py" \
    --dataset CIFAR100H \
    --out topology_daes \
    --exp_name "三方共识更新/100Hpr0.5nr0.2e100top_top_hl15_exp6_knn1_geofuse" \
    --enable_dual_source_refinement \
    --enable_dynamic_sampling \
    --pr 0.5 --nr 0.2 --epochs 100 \
    --sim_mode_1 topology --sim_mode_2 topology \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0

echo "=============================================================================="
echo "所有实验已全部执行完毕！"
echo "=============================================================================="
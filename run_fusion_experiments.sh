#!/bin/bash
# 自动生成的对照实验脚本
# 实验1: 探究 LPI 3 与 LPI 10 下，加与不加 --enable_knn1_model_fuse 的差异
# 实验2: 探究极端噪声下，CIFAR10/100 加与不加融合机制的差异
# 数据集：Benthic & CIFAR

# ====== Part 1: Benthic LPI 3 vs LPI 10 ======
# [1] Benthic, LPI 3, No Fusion
python 结合投影_众包_自节点.py --train_root ./data --dataset Benthic --lpi 3 --out ./topology_daes --exp_name 结合投影_众包_自节点/Benthic/LPI3_NoFuse --network R50 --epochs 100 --batch_size 32 --lr 0.05 --wd 5e-4 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --delta 1.0 --lsr 0.0 --lr_scheduler step --lr_decay_epochs 60 --lr_decay_rate 0.2 --seeds 1 2 3 --detailed_log --cuda_dev 0 --k_val 5 --history_len 100 --knn_heads 1

# [2] Benthic, LPI 3, With Fusion
python 结合投影_众包_自节点.py --train_root ./data --dataset Benthic --lpi 3 --out ./topology_daes --exp_name 结合投影_众包_自节点/Benthic/LPI3_WithFuse --network R50 --epochs 100 --batch_size 32 --lr 0.05 --wd 5e-4 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --delta 1.0 --lsr 0.0 --lr_scheduler step --lr_decay_epochs 60 --lr_decay_rate 0.2 --seeds 1 2 3 --detailed_log --cuda_dev 0 --k_val 5 --history_len 100 --knn_heads 1 --enable_knn1_model_fuse

# [3] Benthic, LPI 10, No Fusion
python 结合投影_众包_自节点.py --train_root ./data --dataset Benthic --lpi 10 --out ./topology_daes --exp_name 结合投影_众包_自节点/Benthic/LPI10_NoFuse --network R50 --epochs 100 --batch_size 32 --lr 0.05 --wd 5e-4 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --delta 1.0 --lsr 0.0 --lr_scheduler step --lr_decay_epochs 60 --lr_decay_rate 0.2 --seeds 1 2 3 --detailed_log --cuda_dev 0 --k_val 5 --history_len 100 --knn_heads 1

# [4] Benthic, LPI 10, With Fusion
python 结合投影_众包_自节点.py --train_root ./data --dataset Benthic --lpi 10 --out ./topology_daes --exp_name 结合投影_众包_自节点/Benthic/LPI10_WithFuse --network R50 --epochs 100 --batch_size 32 --lr 0.05 --wd 5e-4 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --delta 1.0 --lsr 0.0 --lr_scheduler step --lr_decay_epochs 60 --lr_decay_rate 0.2 --seeds 1 2 3 --detailed_log --cuda_dev 0 --k_val 5 --history_len 100 --knn_heads 1 --enable_knn1_model_fuse

# ====== Part 2: CIFAR Extreme Noise ======
# [5] CIFAR10, pr 0.5, nr 0.5, No Fusion
python 结合投影_众包_自节点.py --dataset CIFAR10 --out topology_daes --exp_name 结合投影_众包_自节点/CIFAR10/ExtremeNoise_NoFuse --pr 0.5 --nr 0.5 --epochs 500 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 --knn_heads 1 --seeds 1 2 3 --cuda_dev 1

# [6] CIFAR10, pr 0.5, nr 0.5, With Fusion
python 结合投影_众包_自节点.py --dataset CIFAR10 --out topology_daes --exp_name 结合投影_众包_自节点/CIFAR10/ExtremeNoise_WithFuse --pr 0.5 --nr 0.5 --epochs 500 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 --knn_heads 1 --seeds 1 2 3 --cuda_dev 1 --enable_knn1_model_fuse

# [7] CIFAR100, pr 0.05, nr 0.6, No Fusion
python 结合投影_众包_自节点.py --dataset CIFAR100 --out topology_daes --exp_name 结合投影_众包_自节点/CIFAR100/ExtremeNoise_NoFuse --pr 0.05 --nr 0.6 --epochs 500 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 --knn_heads 1 --seeds 1 2 3 --cuda_dev 1

# [8] CIFAR100, pr 0.05, nr 0.6, With Fusion
python 结合投影_众包_自节点.py --dataset CIFAR100 --out topology_daes --exp_name 结合投影_众包_自节点/CIFAR100/ExtremeNoise_WithFuse --pr 0.05 --nr 0.6 --epochs 500 --sim_mode_1 topology_daes --sim_mode_2 topology_daes --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 --knn_heads 1 --seeds 1 2 3 --cuda_dev 1 --enable_knn1_model_fuse

"""
# [EXP9] 第二阶段输入控制：可选模型融合 + 软/硬输入
# CUB200 示例（不开启模型融合）
python "三方共识更新_exp9_softprop精简版.py" \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 \
    --nr 0.2 \
    --out ./topology_daes \
    --exp_name 三方共识更新_exp9_softprop精简版/CUB200pr0.05nr0.2e250top_top_hl15_exp9_fuse_softpropdel0.25lsr0.0_123 \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.0 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs 250 \
    --cuda_dev 0 \
    --sim_mode_1 topology \
    --sim_mode_2 topology \
    --num_workers 4 \
    --k_val 15 \
    --history_len 15 \
    --enable_knn1_soft_prop \
    --knn1_soft_prop_max_w 1.0 \
    --knn1_soft_prop_eps 1e-8 --enable_knn1_model_fuse

# CUB200 示例（开启模型融合）
# 追加参数：--enable_knn1_model_fuse
"""
"""
==============================================================================
[EXP8] KNN1 Soft Propagation - 保留 softmax 分布的熵信息进行第 2 轮传播
==============================================================================

核心改动:
  1. 第 1 轮传播后, 使用 KNN1 + 模型几何分数的 log 空间融合(与 EXP6 一致)
  2. 第 2 轮传播时, 不做 one-hot 硬化, 保持 softmax 分布传播
  3. 允许 topology 方法利用熵信息计算更准确的可靠性分数

参数:
  --enable_knn1_soft_prop: 启用 EXP8 soft propagation
  --knn1_soft_prop_max_w: 模型权重上限(默认 1.0)
  --knn1_soft_prop_eps: 数值稳定性 epsilon(默认 1e-8)

注意: EXP8 与 EXP6 独立, 可同时启用或分别启用

==============================================================================
运行命令示例
==============================================================================

# ============================================================================
# CIFAR-100 实验
# ============================================================================

# [1] CIFAR-100 + 500 epochs + EXP6 only (baseline)
python "三方共识更新_exp9_softprop精简版.py" \
--dataset CIFAR100 \
--out topology_daes \
--exp_name "三方共识更新_exp9_softprop精简版/pr0.05nr0.5e500top_top_hl15_exp6_knn1_geofuse" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.3 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 --seeds 1 2 3 

# [2] CIFAR-100 + 500 epochs + EXP8 only (soft propagation, no EXP6)
python "三方共识更新_exp9_softprop精简版.py" \
--dataset CIFAR100 \
--out topology_daes \
--exp_name "三方共识更新_exp9_softprop精简版/pr0.1nr0.0e500top_top_hl15_exp8_knn1_softprop_lsr0.0geofuse123" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.1 --nr 0.0 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --enable_knn1_model_fuse --seeds 1 2 3

# [3] CIFAR-100 + 500 epochs + EXP6 + EXP8 (both enabled)
python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR100 \
--out topology_daes \
--exp_name "三方共识更新/pr0.05nr0.5e500top_top_hl15_exp6+8_geofuse+softprop" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.5 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

# [4] CIFAR-100 + 100 epochs + EXP6 only (fast iteration)
python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR100 \
--out topology_daes \
--exp_name "三方共识更新_exp9_knn1_softprop/pr0.05nr0.5e500top_top_hl15" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.5 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 

# [5] CIFAR-100 + 100 epochs + EXP8 only
python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR100 \
--out topology_daes \
--exp_name "三方共识更新_exp9_knn1_softprop/pr0.05nr0.5e100top_top_hl15" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.5 --epochs 100 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 

# [6] CIFAR-100 + 100 epochs + EXP6 + EXP8
python "三方共识更新_exp9_softprop精简版.py" \
--dataset CIFAR100H \
--out topology_daes \
--exp_name "三方共识更新_exp9_softprop精简版/100Hpr0.5nr0.2e500top_exp_hl15_geofuse+softprop" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.5 --epochs 500 --sim_mode_1 topology --sim_mode_2 exp \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --enable_knn1_model_fuse

# ============================================================================
# CIFAR-10 实验 (对称噪声)
# ============================================================================

# [7] CIFAR-10 + 500 epochs + EXP6 only
python "三方共识更新_exp9_softprop精简版.py" \
--dataset CIFAR10 \
--out topology_daes \
--exp_name "三方共识更新_exp9_softprop精简版/pr0.5nr0.2e500top_top_hl15_exp6_knn1_geofuse" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.5 --nr 0.2 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 

# [8] CIFAR-10 + 500 epochs + EXP8 only
python "三方共识更新_exp9_softprop精简版.py" \
--dataset CIFAR10 \
--out topology_daes \
--exp_name "三方共识更新_exp9_softprop精简版/pr0.1nr0.1e500top_top_hl15_exp8_knn1_softprop_fuse" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.1 --nr 0.1 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --enable_knn1_model_fuse

# [9] CIFAR-10 + 500 epochs + EXP6 + EXP8
python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR10 \
--out topology_daes \
--exp_name "三方共识更新/pr0.05nr0.4e500top_top_hl15_exp6+8_geofuse+softprop" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.4 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

# [10] CIFAR-10 + 100 epochs + EXP6 only
python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR10 \
--out topology_daes \
--exp_name "三方共识更新/pr0.05nr0.4e100top_top_hl15_exp6_knn1_geofuse" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.4 --epochs 100 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \

# [11] CIFAR-10 + 100 epochs + EXP8 only
python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR10 \
--out topology_daes \
--exp_name "三方共识更新/pr0.05nr0.4e100top_top_hl15_exp8_knn1_softprop" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.05 --nr 0.4 --epochs 100 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0

# [12] CIFAR-10 + 100 epochs + EXP6 + EXP8
python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR10 \
--out topology_daes \
--exp_name "三方共识更新/ cifar10pr0.5nr0.3e300top_top_hl15_exp6+8_geofuse+softprop" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.5 --nr 0.3 --epochs 300 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0



python "三方共识更新_exp9_knn1_softprop.py" \
--dataset CIFAR10 \
--out topology_daes \
--exp_name "三方共识更新_exp9_knn1_softprop/ cifar10pr0.5nr0.3e500top_top_geofuse+softprop_lsr0.2" \
--enable_dual_source_refinement \
--enable_dynamic_sampling \
--pr 0.5 --nr 0.3 --epochs 500 --sim_mode_1 topology --sim_mode_2 topology \
--lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 \
--enable_knn1_soft_prop --knn1_soft_prop_max_w 1.0 --seeds 1 2 3 \
    --knn1_soft_prop_max_w 1.0 \
    --knn1_soft_prop_eps 1e-8 --enable_knn1_model_fuse
==============================================================================
"""


# =N============================================================================
#           HYBRID PALS-SSL FRAMEWORK - THREE-PHASE TRAINING
#
# Desc: This version implements a sequential, three-tier training strategy per epoch:
# 1. SimSiam Phase: Train on the lowest-quality samples (remaining set)
#    using a self-supervised, feature-learning objective (SimSiam).
# 2. Hybrid Phase: Train on a mix of reliable (supervised) and high-quality
#    unsupervised (consensus set) samples using Mixup, CE loss, and SoftMatch.
# ==============================================================================
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math # <--- Added
import argparse
import os
import time
from collections import deque
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import logging
from PIL import Image
from torch.amp import autocast, GradScaler
import torch.optim as optim
import wandb
import sys
import pandas as pd # CUB200依赖
import itertools
import datetime # <--- Added
from torchvision.models import resnet18, resnet50
# 假设您的工具函数在以下路径
from data.dataset import CUB200Partial, CIFAR10Partial, CIFAR100Partial
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy ,ImageNetPolicy
# from utils.prototype_manager import PrototypeManager
# from utils.diagnostics import log_tri_consensus_diagnostics
from data.crowdsource import *
# 5. 创建采样器 (直接使用对齐后的列表)
# 注意:sampling_weights_aligned 的长度必须等于 unified_dataset 的长度
from torch.utils.data import WeightedRandomSampler
# ==============================================================================
#                      Section 0: 环境设置 (Environment Setup)
# ==============================================================================
def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# In Section 0: 环境设置 (Environment Setup)

def setup_logger(log_dir, filename="run.log", is_master=False, to_console=False):
    """
    Modified to allow disabling console output explicitly.
    """
    logger_name = f"logger_{log_dir.replace('/', '_')}_{filename}"
    logger = logging.getLogger(logger_name)
    
    if logger.hasHandlers():
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
            
    logger.setLevel(logging.INFO)
    logger.propagate = False 
    
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S")
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    
    # File Handler (Always active)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler (Only active if to_console is True)
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='Ultimate Hybrid PALS-SSL Framework with Three-Phase Training')
    # 基本设置
    parser.add_argument('--exp_name', type=str, default='HybridPALS_ThreePhase_Run', help='Experiment name.')
    
    # 在 parse_args() 函数中修改:
    parser.add_argument('--dataset', type=str, default='CIFAR100', 
                        choices=['CIFAR10', 'CIFAR100', 'CIFAR100H', 'CUB200', 
                                'Treeversity', 'Benthic', 'Plankton',])
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--out', type=str, default='./out_ultimate', help='Directory for output')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1], help='List of random seeds.')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    # 部分标签 (PLL) 设置
    parser.add_argument('--pr', type=float, default=0.05, help='partial ratio (q)')
    parser.add_argument('--nr', type=float, default=0.5, help='noise ratio (eta)')
    parser.add_argument('--lpi', type=int, default=10, help='Labels Per Image (LPI) for crowdsource NPLL conversion')
    # 核心算法开关
    parser.add_argument('--reliable_selection_mode', type=str, default='pals', choices=['mine', 'pals'], help="Strategy for reliable set selection.")
    
    # 训练超参数
    parser.add_argument('--network', type=str, default='R18', help='Network architecture (R18, R50)')
    parser.add_argument('--epochs', type=int, default=500, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step'],
                        help='Type of learning rate scheduler (cosine or step).')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[60, 120, 160, 200],
                        help='Epoch milestones for the step learning rate scheduler.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='Decay rate (gamma) for the step learning rate scheduler.')    
    # 损失函数超参数
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Alpha for Mixup.')
    parser.add_argument('--lsr', type=float, default=0.5, help='Label smoothing rate.')
    parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for consistency loss.')

    # --- 🚀 消融实验开关 (Ablation Study Flags) ---
    parser.add_argument('--no_reliable_mixup', action='store_true', help='[Ablation] Disable Mixup on reliable set.')
    parser.add_argument('--no_rebalance', action='store_true', help='[Ablation] Disable class rebalancing on pseudo-labels.')
    parser.add_argument('--no_softmatch', action='store_true', help='[Ablation] Disable SoftMatch weighting (force weight=1.0).')
    parser.add_argument('--no_unreliable_mixup', action='store_true', help='[Ablation] Disable Mixup on unreliable set (use standard consistency).')
    
    # [新增] 完全禁用不可靠集训练
    parser.add_argument('--no_unreliable_training', action='store_true', help='[Ablation] COMPLETELY ignore unreliable set (Supervised Only).')
    
    # [新增] 消融:禁用 Middleware Rectification
    parser.add_argument('--no_rectify', action='store_true', help='[Ablation] Disable Middleware Gating/Rectification.')
    
    # [新增] 消融:EMA 因子
    parser.add_argument('--ema_alpha', type=float, default=0.999, help='EMA momentum factor (default: 0.999).')
    # ---------------------------------------------

    # KNN & 平衡参数
    parser.add_argument('--k_val', type=int, default=15, help='k for knn')
    parser.add_argument('--delta', type=float, default=0.25, help='example selection quantile')
    
    parser.add_argument('--history_len', type=int, default=3, help='example selection quantile')
    # --- 🚀 [Added for Ablation Master Control] ---
    parser.add_argument('--consensus_power', type=float, default=2.0, help='Power for consensus proportion in dynamic weight (default: 2.0)')
    parser.add_argument('--fix_dynamic_weight', action='store_true', help='[Ablation] Fix dynamic consistency weight to 1.0 (Disable curriculum)')
    # ----------------------------------------------
# --- 修改部分开始 ---
    parser.add_argument('--sim_mode_1', type=str, default='topology',  # <--- 修改默认值为 topology
                        choices=['topology', 'exp', 'daes', 'none'],   # <--- 确保包含所有选项
                        help='Similarity measure for Stage 1')

    parser.add_argument('--sim_mode_2', type=str, default='daes',      # <--- 第二阶段通常用 daes 或 topology
                        choices=['topology', 'exp', 'daes', 'none'],   # <--- 确保包含所有选项
                        help='Similarity measure for Stage 2')   

    # [Topology] 信誉度核:让 topology 在 one-hot 标签下也能工作
    parser.add_argument('--topology_rel_mode', type=str, default='masked_entropy',
                        choices=['masked_entropy', 'kl', 'agree'],
                        help='[Topology] Reliability score mode. Use kl/agree to support one-hot labels.')
    parser.add_argument('--topology_rel_gamma', type=float, default=2.0,
                        help='[Topology] Penalty strength for low-consensus nodes (default: 2.0).')
    parser.add_argument('--topology_rel_eps', type=float, default=1e-12,
                        help='[Topology] Epsilon for numerical stability.')

    parser.add_argument('--warmup_epochs', type=int, default=250, help='Epochs for linear LR warmup.')

    # 日志
    parser.add_argument('--detailed_log', action='store_true', help='Enable detailed diagnostic logging.')
    # 1. Teacher Gating 控制 (干预机制)
    parser.add_argument('--gating_start_ratio', type=float, default=0.2, 
                        help='[Gating] Ratio of epochs before teacher gating starts (default: 0.2, means start at 20% epoch).')
    parser.add_argument('--gating_max_alpha', type=float, default=1.0, 
                        help='[Gating] Max influence of teacher (0.0 to 1.0). Set <1.0 to always keep some geometry signal.')

    # 2. DAES 算法控制 (拓扑构建)
    parser.add_argument('--daes_clamp', type=float, default=0.25, 
                        help='[DAES] Max temperature clamp (Anti-oversmoothing lock). Lower is sharper.')
    parser.add_argument('--daes_entropy_weight', type=float, default=0.2, 
                        help='[DAES] Sensitivity to local entropy (tau = base + weight * H).')
    parser.add_argument('--daes_sharpening_power', type=float, default=2.0, 
                        help='[DAES] Sharpening power for local mean calculation (CUB=2.0, Standard=1.0).')
    
    # [新增] DAES 参数化控制
    parser.add_argument('--daes_spatial_temp', type=float, default=0.5, 
                        help='[DAES] Temperature for spatial weighting (default: 0.5).')
    parser.add_argument('--daes_base_tau', type=float, default=0.1, 
                        help='[DAES] Base temperature for affinity matrix (default: 0.1).')
    parser.add_argument('--daes_entropy_coeff', type=float, default=0.5, 
                        help='[DAES] Coefficient for entropy-based temperature adjustment (default: 0.5).')
    parser.add_argument('--daes_sim_power', type=float, default=2.0, 
                        help='[DAES] Power to raise similarity to (default: 2.0).')
    
    # [新增] 拓扑参考模式
    parser.add_argument('--daes_topology_ref_mode', type=str, default='hard', choices=['gated', 'hard'],
                        help='[DAES] Topology reference mode: "gated" (use teacher confidence gated signal) or "hard" (use raw hard signal).')

    # 3. KNN 图构建
    parser.add_argument('--knn_heads', type=int, default=4, 
                        help='[KNN] Number of  heads for metric learning (Robustness).')
    
    # 🚀 新增: 双源KNN交集筛选参数
    parser.add_argument('--enable_dual_source_refinement', action='store_true',
                        help='[Dual-Source] Enable dual-source KNN intersection refinement.')
    parser.add_argument('--refinement_start_epoch', type=int, default=10,
                        help='[Dual-Source] Epoch to start dual-source refinement (default: 10).')
    parser.add_argument('--threshold_knn', type=float, default=0.85,
                        help='[Dual-Source] Confidence threshold for original KNN propagation (default: 0.85).')
    parser.add_argument('--threshold_model', type=float, default=0.85,
                        help='[Dual-Source] Confidence threshold for model-KNN propagation (default: 0.85).')
    parser.add_argument('--sim_mode_3', type=str, default='exp', choices=['exp', 'daes', 'topology_entropy'],
                        help='[Dual-Source] Similarity mode for model-KNN propagation (default: exp).')
    parser.add_argument('--max_refinement_ratio', type=float, default=0.3,
                        help='[Dual-Source] Maximum ratio of unreliable samples to refine (default: 0.3, adaptive: 0.3→0.8).')
    
    # 🚀 新增: 动态全可靠集训练参数
    parser.add_argument('--enable_dynamic_sampling', action='store_true',
                        help='[Dynamic] Enable dynamic progressive sampling strategy.')
    parser.add_argument('--full_reliable_threshold', type=float, default=0.7,
                        help='[Dynamic] Reliable ratio threshold to switch to full-reliable mode (default: 0.7).')
    # 在 parse_args() 函数的 "核心算法开关" 或 "消融实验" 部分加入:

    parser.add_argument('--enable_dynamic_rescue', action='store_true',
                        help='[Rescue] Enable dynamic threshold rescue for high-confidence samples.')

    parser.add_argument('--rescue_quantile', type=float, default=0.5,
                        help='[Rescue] Quantile for threshold (0.5=Median for Robust, 0.9=Top10% for Aggressive).')
    # [新增] 消融:禁用一致性正则化
    parser.add_argument('--no_consistency', action='store_true', 
                        help='[Ablation] Disable Consistency Reg. Train on WEAK images only.')

    # [EXP9] 第二阶段输入: 可选模型预测融合 + 软/硬输入控制
    parser.add_argument('--enable_knn1_model_fuse', action='store_true',
                        help='[EXP9] Enable model-geometry fusion with KNN1 scores (log-space) before candidate projection.')
    parser.add_argument('--enable_knn1_soft_prop', action='store_true',
                        help='[EXP9] Use soft distribution as 2nd-stage input (otherwise one-hot).')
    parser.add_argument('--knn1_soft_prop_max_w', type=float, default=1.0,
                        help='[EXP9] Max model weight; ramps linearly from 0 to this value over epochs.')
    parser.add_argument('--knn1_soft_prop_eps', type=float, default=1e-8,
                        help='[EXP9] Epsilon for fusion/normalization stability.')

    return parser.parse_args()
# (在 Section 2: 数据处理与模型)

class PrototypeManager:
    def __init__(self, num_classes, feature_dim, ema_alpha=0.9, device='cuda'):
        """
        Args:
            ema_alpha: 历史原型的保留比例 (0.9 表示新中心只占 0.1 权重)
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.ema_alpha = ema_alpha
        self.device = device
        
        # 初始化原型 (N_class, Dim)
        self.prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.is_initialized = False

    def update(self, features, reliable_mask, reliable_labels):
        """
        利用当前的可靠样本更新原型
        features: [N, Dim]
        reliable_mask: [N] bool
        reliable_labels: [N] (可能是伪标签或干净标签，取决于传入什么)
        """
        features = features.detach()
        # 确保特征归一化 (配合余弦相似度)
        features = F.normalize(features, dim=1)
        
        # 筛选可靠样本
        rel_feats = features[reliable_mask]
        rel_targets = reliable_labels[reliable_mask]
        
        if len(rel_feats) == 0:
            return

        # 计算当前 Batch/Epoch 的新类中心
        new_protos = torch.zeros_like(self.prototypes)
        
        # 这种写法比循环快
        # Numerator: Sum features per class
        # Denominator: Count per class
        one_hot = F.one_hot(rel_targets.long(), self.num_classes).float() # [N_rel, C]
        
        # [C, N_rel] @ [N_rel, Dim] -> [C, Dim]
        sum_features = torch.mm(one_hot.T, rel_feats) 
        counts = one_hot.sum(dim=0).unsqueeze(1) + 1e-8
        
        current_means = sum_features / counts
        current_means = F.normalize(current_means, dim=1) # 再次归一化

        # EMA 更新
        if not self.is_initialized:
            self.prototypes = current_means
            self.is_initialized = True
        else:
            # 只有当前 batch 出现过的类别才更新，没出现的保持原样
            # mask: [C, 1]
            active_classes = (one_hot.sum(dim=0).unsqueeze(1) > 0).float()
            
            updated_protos = self.ema_alpha * self.prototypes + (1 - self.ema_alpha) * current_means
            
            # 组合：活跃类用更新的，不活跃类用旧的
            self.prototypes = active_classes * updated_protos + (1 - active_classes) * self.prototypes
            
        # 保持原型在单位球面上
        self.prototypes = F.normalize(self.prototypes, dim=1)

    def predict(self, query_features):
        """
        基于余弦相似度进行预测
        return: 
            sims: [N, C] 相似度分数
            preds: [N] 预测类别
        """
        query_features = F.normalize(query_features, dim=1)
        # [N, Dim] @ [Dim, C] -> [N, C]
        sims = torch.mm(query_features, self.prototypes.T)
        preds = sims.argmax(dim=1)
        return sims, preds


def get_pals_transforms(dataset_name):
    # --- (新增) 众包数据集的 Mean/Std ---
    if dataset_name == 'Treeversity':
        mean = [0.4439581940620345, 0.4509297096690951, 0.3691211738638277]
        std = [0.23407518616927706, 0.22764417468550843, 0.2600833107790479]
    elif dataset_name == 'Benthic':
        mean = [0.34728872821176615, 0.40013687864974884, 0.4110478166769647]
        std = [0.1286915489786319, 0.13644626747739305, 0.14258506692263767]
    elif dataset_name == 'Plankton':
        mean = [0.9663359216202008, 0.9663359216202008, 0.9663359216202008]
        std = [0.10069729102981237, 0.10069729102981237, 0.10069729102981237]
    elif dataset_name == 'CUB200':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else: # 默认为 CIFAR
        mean, std = ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]) if '100' in dataset_name else ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    # --- (修改) 扩展 Transform 逻辑 ---
    
    # (新增) CUB200 (使用 PALS 原始的强 Aug)
    if dataset_name == 'CUB200':
        weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            # CIFAR10Policy(), 
            ImageNetPolicy(),
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=56), # <-- 关键！
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
        ])
        
    elif dataset_name == 'Treeversity':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(int(224/0.875)), # (256)
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset_name == 'Benthic':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((112,112)),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name == 'Plankton':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            # CIFAR10Policy(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else: # CIFAR
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    # 包含您截图中的所有新数据集
    if dataset_name in ['Turkey', 'Pig', 'MiceBone', 'QualityMRI', 'Synthetic', 
                        'verse_blended-vps', 'verse_mask1-vps', 'CIFAR10H']:
        
        # 使用 ImageNet 统计数据作为通用初始化
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # 如果是 CIFAR10H,可能图片很小 (32x32),需要特殊处理
        if 'CIFAR' in dataset_name or 'Synthetic' in dataset_name:
            resize_size = 32
            crop_size = 32
        else:
            # 其他真实世界数据集 (Turkey, Pig等) 使用标准 224
            resize_size = 256
            crop_size = 224

        weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(), # 强增强
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return weak_transform, strong_transform, test_transform
def get_base_encoder(network_name, dataset_name):
    
    
    # --- (修改) 扩展使用预训练权重的条件 ---
    # use_pretrained = dataset_name in ['CUB200', 'Treeversity', 'Benthic', 'Plankton','Synthetic',]
    use_pretrained = dataset_name in ['CUB200', 'Treeversity', 'Benthic', 'Plankton',]
    
    if network_name == 'R50':
        base_model = resnet50(weights='IMAGENET1K_V1' if use_pretrained else None)
    else: # Default to R18
        base_model = resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)

    feature_dim = base_model.fc.in_features
    
    # if 'CIFAR' in dataset_name:
    if 'CIFAR' in dataset_name or 'Synthetic' in dataset_name:
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        
    encoder = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten())
    return encoder, feature_dim


## --- 🚀 MODIFICATION START: Removed SimSiam class --- ##
# class SimSiam(nn.Module): ... (Class Removed)
# def simsiam_loss_fn(p, z): ... (Function Removed)
## --- MODIFICATION END --- ##


class FeatureExtractionDataset(Dataset):
    def __init__(self, base_dataset, transform): 
        self.base_dataset, self.transform = base_dataset, transform
        
        # --- (修改) ---
        # 我们需要明确区分 CUB200 和 Crowdsource
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        # --- (修改结束) ---

    def __len__(self): 
        return len(self.base_dataset)
        
    def __getitem__(self, index):
        # 1. 获取原始图像
        
        # --- (修改) ---
        if self.is_cub:
            # CUB200: .data 是 DataFrame. 必须用 .data_paths
            # (这是在破坏"封装",但这是在你设定的约束下唯一可行的方法)
            img_path = os.path.join(self.base_dataset.root, 
                                    self.base_dataset.base_folder, 
                                    'images', 
                                    self.base_dataset.data_paths[index])
            img = Image.open(img_path).convert('RGB')
        
        elif self.is_crowd:
            # Crowdsource: .data 是 'list' of paths, 可以直接用 [index]
            img_path = self.base_dataset.data[index]
            img = Image.open(img_path).convert('RGB')


        else:
            # CIFAR: .data 是 numpy 数组
            img = Image.fromarray(self.base_dataset.data[index])
        # --- (修改结束) ---

        # 2. 应用 *这个类* 自己的 transform (即 test_t)
        return self.transform(img), index


# ==============================================================================
#      Section 2.5: 统一数据集 + 动态重要性采样器 (Unified Dataset + Importance Sampler)
# ==============================================================================

class UnifiedSSLDataset(Dataset):
    """
    统一的单流数据集，合并可靠集和不可靠集
    每个样本带有 is_reliable 标志用于区分训练策略
    """
    def __init__(self, base_dataset, data_list, weak_t, strong_t):
        """
        Args:
            base_dataset: 原始数据集 (CIFAR/CUB/Crowdsource)
            data_list: [(idx, label, is_reliable), ...]
                - idx: 原始索引
                - label: 伪标签（可靠集）或 -1（不可靠集）
                - is_reliable: True/False
            weak_t: 弱增强变换
            strong_t: 强增强变换
        """
        self.base_dataset = base_dataset
        self.data_list = data_list
        self.weak_t = weak_t
        self.strong_t = strong_t
        
        # 检测数据集类型
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        original_idx, label, is_reliable = self.data_list[idx]
        
        # 获取原始图像
        if self.is_cub:
            img_path = os.path.join(self.base_dataset.root,
                                    self.base_dataset.base_folder,
                                    'images',
                                    self.base_dataset.data_paths[original_idx])
            img = Image.open(img_path).convert('RGB')
        elif self.is_crowd:
            img_path = self.base_dataset.data[original_idx]
            img = Image.open(img_path).convert('RGB')
        else:  # CIFAR
            img = Image.fromarray(self.base_dataset.data[original_idx])
        
        return (self.weak_t(img), self.strong_t(img),
                label, is_reliable, original_idx)


class TemporalStateManager:
    def __init__(self, num_samples, num_classes, max_epochs, history_len=5, use_disambiguation=True):
        self.N, self.C = num_samples, num_classes
        self.max_epochs = max_epochs
        self.history_len = history_len
        self.use_disambiguation = use_disambiguation 
        # 🚀 三方队列共识硬标签快照队列
        # 存储格式:达成三方共识存入 Label(0-99),未达成存入 -1
        self.tri_consensus_history = deque(maxlen=history_len)
        
        # 为了判断 "始终不可靠",保留可靠性历史
        self.is_reliable_history = deque(maxlen=history_len)
        # 队列维护:记录每个样本的可靠性状态
        # self.is_reliable_history = deque(maxlen=history_len)
        
        # 记录 1: 剪枝后的 KNN 标签 (Topology-KNN)
        self.pruned_pl_history = deque(maxlen=history_len)
        
        # 记录 2: 基于模型预测的几何标签 (Model-KNN)
        self.geo_pl_history = deque(maxlen=history_len)
        
        # 🚀 记录 3: [新增] 基于类原型的预测标签 (Proto-PL)
        self.proto_pl_history = deque(maxlen=history_len)
        
        # 消歧参考:模型历史预测分布的移动平均 (EMA)
        self.prob_ema = torch.ones(num_samples, num_classes) / num_classes
        self.ema_m = 0.995 

    def update_ema(self, current_model_probs):
        """使用模型预测更新 EMA"""
        self.prob_ema = self.ema_m * self.prob_ema + (1 - self.ema_m) * current_model_probs.cpu()

    def update_history(self, is_reliable_mask, pruned_pl, geo_pl=None, proto_pl=None):
        """
        存入历史轨迹 
        Args:
            is_reliable_mask: 当前 epoch 是否被选为可靠
            pruned_pl: Phase 2 产生的 KNN 伪标签
            geo_pl: 基于去自身化邻居的模型几何预测标签
            proto_pl: [新增] 基于类原型的预测标签
        """
        self.is_reliable_history.append(is_reliable_mask.cpu().bool())
        self.pruned_pl_history.append(pruned_pl.cpu().long())
        
        if geo_pl is not None:
            self.geo_pl_history.append(geo_pl.cpu().long())
        else:
            self.geo_pl_history.append(torch.full_like(pruned_pl, -1).cpu().long())

        # 🚀 [新增] 记录 Proto 历史
        if proto_pl is not None:
            self.proto_pl_history.append(proto_pl.cpu().long())
        else:
            self.proto_pl_history.append(torch.full_like(pruned_pl, -1).cpu().long())

    def get_dynamic_disambiguation(self, epoch, device):
        if not self.use_disambiguation or epoch == 0:
            return torch.ones(self.N, self.C).to(device)
            
        alpha = (epoch / self.max_epochs) ** 2
        D = torch.pow(self.prob_ema + 1e-12, alpha)
        return D.to(device)

    def get_salvage_mask(self):
        """
        Tri-Consensus Salvage (三方共识打捞)
        要求: KNN (Topology), Model-Geo, 和 Prototype 在历史上都稳定且达成一致。
        """
        # 1. Base safety check
        curr_len = len(self.is_reliable_history)
        if curr_len == 0:
            return None, None, None, None

        # 2. Stack History
        rel_stack = torch.stack(list(self.is_reliable_history)) 
        pl_stack = torch.stack(list(self.pruned_pl_history))

        # 3. Calculate Metrics
        # A. Always Unreliable (始终是弃儿)
        always_unreliable = (rel_stack.sum(dim=0) == 0)

        # B. KNN Consistency (KNN 自身历史稳定)
        knn_consistency = (pl_stack == pl_stack[-1:]).all(dim=0)

        # C. Geo Consistency (Model-KNN 自身历史稳定)
        if len(self.geo_pl_history) > 0:
            geo_stack = torch.stack(list(self.geo_pl_history))
            valid_geo = (geo_stack != -1).all(dim=0)
            geo_consistency = (geo_stack == geo_stack[-1:]).all(dim=0) & valid_geo
        else:
            geo_consistency = torch.zeros_like(knn_consistency, dtype=torch.bool)
            geo_stack = pl_stack # Fallback

        # 🚀 D. [新增] Proto Consistency (Proto 自身历史稳定)
        if len(self.proto_pl_history) > 0:
            proto_stack = torch.stack(list(self.proto_pl_history))
            valid_proto = (proto_stack != -1).all(dim=0)
            proto_consistency = (proto_stack == proto_stack[-1:]).all(dim=0) & valid_proto
        else:
            # 如果没有 proto 历史,暂时放宽此条件或设为 False (视严格程度而定)
            # 建议: 如果启用了 Proto,这里应该是 False;为了兼容性先设 False
            proto_consistency = torch.zeros_like(knn_consistency, dtype=torch.bool)
            proto_stack = pl_stack # Fallback

        # F. Cross-Track Agreement(三方:KNN / Geo / Proto)
        cross_track_agreement = (pl_stack[-1] == geo_stack[-1]) & (geo_stack[-1] == proto_stack[-1])

        # 4. Source Attribution (归因分析,用于日志)
        knn_src_mask = always_unreliable & knn_consistency
        geo_src_mask = always_unreliable & geo_consistency

        # 5. Combine Masks (Strict Intersection)
        # 必须: 始终不可靠 AND 历史稳定 AND 三方意见一致
        salvage_mask = knn_src_mask & geo_src_mask & proto_consistency & cross_track_agreement

        # 6. Determine Labels
        # 既然三方一致,直接取任意一个(这里取 KNN)的标签即可
        final_salvaged_labels = self.pruned_pl_history[-1].clone()

        return salvage_mask, final_salvaged_labels, knn_src_mask, geo_src_mask


    def update_tri_consensus(self, is_reliable_mask, p_model, p_knn, p_proto):
        """计算并存入瞬时三方共识快照（Model / KNN / Proto）。

        注意：`is_reliable_history` 的维护在 `update_history(...)` 中完成，
        这里不再重复 append，避免出现"一个 epoch 写两次历史"的现象。
        """
        # 获取各方硬预测
        pl_m = p_model.argmax(dim=1).cpu()
        pl_k = p_knn.argmax(dim=1).cpu()
        pl_p = p_proto.argmax(dim=1).cpu()

        tri_mask = (pl_m == pl_k) & (pl_k == pl_p)

        # 构造快照:共识则留标签,否则 -1
        snapshot = torch.where(tri_mask, pl_m, torch.tensor(-1))
        self.tri_consensus_history.append(snapshot)


    def get_stable_tri_mask(self):
        """筛选出：

        1. 历史区间内从未进入可靠集（针对打捞场景）
        2. 在 history_len 内每一帧都达成三方共识（无 -1）
        3. 标签在 history_len 内完全锁定
        """
        if len(self.tri_consensus_history) < self.history_len or len(self.is_reliable_history) < self.history_len:
            return None, None

        rel_stack = torch.stack(list(self.is_reliable_history))
        tri_stack = torch.stack(list(self.tri_consensus_history))

        always_unreliable = (rel_stack.sum(dim=0) == 0)

        ever_disagree = (tri_stack == -1).any(dim=0)
        always_tri_agree = ~ever_disagree

        latest_label = tri_stack[-1]
        label_is_locked = (tri_stack == latest_label).all(dim=0)

        stable_mask = always_unreliable & always_tri_agree & label_is_locked
        return stable_mask, latest_label

# ==============================================================================
# 核心组件 1: 拓扑引导亲和矩阵计算 (核心核函数)
# ==============================================================================
def get_topology_guided_affinity(raw_D, neighbors_indices, current_soft_labels, num_classes,
                                rel_mode='masked_entropy', gamma=2.0, eps=1e-12):
    """计算基于拓扑一致性的动态亲和矩阵（支持 one-hot 标签）。

    raw_D: [N, K+1] 原始相似度
    neighbors_indices: [N, K+1]
    current_soft_labels: [N, C]（可以是 soft / multi-hot / one-hot）

    rel_mode:
      - 'masked_entropy': 旧版（one-hot 时会退化为恒 1 信誉度）
      - 'kl': 用 KL/交叉熵度量 self vs neighbor 分布（one-hot 时等价于 -log p_knn[y]）
      - 'agree': 用邻域对 self 的支持度（dot(p_knn, p_self)）构造分数
    """
    N, K_plus_1 = neighbors_indices.shape

    # --- Step 1: 纯线性平滑 KNN 估计 (只用邻居预测我) ---
    linear_weights = raw_D[:, 1:]
    linear_weights = linear_weights / (linear_weights.sum(dim=1, keepdim=True) + eps)

    neighbor_labels = F.embedding(neighbors_indices[:, 1:], current_soft_labels)  # [N, K, C]
    knn_scores_smooth = (linear_weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)  # [N, C]
    p_knn = knn_scores_smooth / (knn_scores_smooth.sum(dim=1, keepdim=True) + eps)

    # --- Step 2: 计算节点信誉度分数(对 one-hot 也有效) ---
    p_self = current_soft_labels
    p_self = p_self / (p_self.sum(dim=1, keepdim=True) + eps)

    if rel_mode == 'masked_entropy':
        masked_scores = p_knn * p_self
        masked_prob = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + eps)
        entropy = -torch.sum(masked_prob * torch.log(masked_prob + eps), dim=1)  # [N]
        norm_score = entropy / (np.log(num_classes) + eps)

    elif rel_mode == 'kl':
        # KL(p_self || p_knn) 在 one-hot 情况下退化为 -log p_knn[y]
        cross_entropy = -torch.sum(p_self * torch.log(p_knn + eps), dim=1)
        norm_score = cross_entropy / (np.log(num_classes) + eps)

    elif rel_mode == 'agree':
        # 邻域对 self 的支持度(one-hot: p_knn[y])
        agree_mass = torch.sum(p_knn * p_self, dim=1).clamp(min=eps, max=1.0)
        norm_score = (-torch.log(agree_mass)) / (np.log(num_classes) + eps)

    else:
        raise ValueError(f"Unknown rel_mode: {rel_mode}")

    gamma = float(gamma)
    reliability_scores = torch.exp(-gamma * (norm_score ** 2))  # [N]

    # --- Step 3: 生成最终亲和矩阵 ---
    reliability_scores_expanded = reliability_scores.unsqueeze(1)  # [N, 1]
    all_reliabilities = F.embedding(neighbors_indices, reliability_scores_expanded).squeeze(-1)
    refined_sim = raw_D * all_reliabilities

    return refined_sim, reliability_scores

def reliable_pseudolabel_selection_advanced(logger, args, device, trainloader, features, epoch, 
                                            state_manager, model_preds=None,proto_manager=None):
    """
    博士级增强版：双源感知可靠集筛选
    1. 原始可靠集 (Static Constraint)
    2. 历史修正共识集 (Dynamic History Consensus)
    """
    N = features.shape[0]
    dataset = trainloader.dataset
    
    # 获取原始静态约束
    if hasattr(dataset, 'original_soft_labels'):
        static_cand_mask = torch.tensor(dataset.original_soft_labels, device=device, dtype=torch.float64)
    else:
        static_cand_mask = torch.tensor(dataset.soft_labels, device=device, dtype=torch.float64)

    # 获取当前动态起点 (包含已覆写的标签)
    current_fixed_labels = torch.tensor(dataset.soft_labels, device=device).float()
    clean_labels = torch.tensor(dataset.clean_labels, device=device, dtype=torch.long)

    # ==============================================================================
    # 核心辅助逻辑:双源统一筛选闸门
    # ==============================================================================
    def _filter_logic(soft_probs):
        prob_temp = torch.clamp(soft_probs, min=1e-6, max=1-1e-6)
        discrepancy = -torch.log(prob_temp)
        max_p, max_idx = soft_probs.max(dim=1)
        
        # --- 🚀 双源准入判定 ---
        # 源 A: 满足原始候选集约束
        in_static_cand = (static_cand_mask.gather(1, max_idx.unsqueeze(1)).squeeze(1) == 1.0)
        
        # # 源 B: 满足历史修正共识 (已改动过 且 预测与改动后一致)
        # has_been_modified = torch.from_numpy(dataset.modified_mask).to(device)
        # history_fixed_idx = current_fixed_labels.argmax(dim=1)
        # is_modified_consensus = has_been_modified & (max_idx == history_fixed_idx)
        
        # # 合并所有候选者
        # total_cand_mask = in_static_cand | is_modified_consensus
        # ----------------------
        total_cand_mask = in_static_cand
        rel_mask = torch.zeros(N, device=device)
        # 统计各类别候选数量用于计算分位数
        counts = torch.bincount(max_idx[total_cand_mask], minlength=args.num_classes).double()
        limit = torch.quantile(counts, args.delta) if counts.numel() > 0 else 0
        
        for i in range(args.num_classes):
            idx_c_mask = total_cand_mask & (max_idx == i)
            if idx_c_mask.sum() == 0: continue
            
            k_val = min(limit.item(), idx_c_mask.sum().float().item())
            if k_val < 1: continue
            
            # 在同一类别内,无论来自哪个源,按置信度(discrepancy)公平竞争 Top-K
            _, top_idx = torch.topk(discrepancy[idx_c_mask, i], k=int(k_val), largest=False)
            rel_mask[idx_c_mask.nonzero().squeeze(1)[top_idx]] = 1.0
            
        n_selected = rel_mask.sum().item()
        acc = (max_idx[rel_mask.bool()] == clean_labels[rel_mask.bool()]).float().mean().item() if n_selected > 0 else 0.0
        return rel_mask, max_idx, acc, n_selected

    # 1. 拓扑构建
    D_mh, neighbors_mh = knn_search_pytorch_chunked(features, args.k_val, num_heads=args.knn_heads)
    raw_sim = F.relu(D_mh).float()

    # --- 📊 统计初始状态 (Before Propagation) ---
    # 为了对比 Step 1 的效果,我们先看一眼传播前的准确率
    with torch.no_grad():
        start_pred = current_fixed_labels.argmax(dim=1)
        acc_start = (start_pred == clean_labels).float().mean().item()
        # 简单估算初始"可靠"数量,这里用简单的阈值或直接复用 _filter_logic 太耗时,
        # 我们这里用 "满足静态约束且argmax正确" 的数量作为基准参考
        size_start = (static_cand_mask.gather(1, start_pred.unsqueeze(1)).squeeze(1) == 1.0).sum().item()

    # 2. 迭代拓扑传播 (使用当前已修正的标签作为传播起点)
    curr_soft_out = current_fixed_labels.clone()
    iterations = 2
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 [Epoch {epoch}] Reliable Selection - Stage Breakdown")
    logger.info(f"{'='*80}")

    # 📊 Stage 0: 初始状态 (传播前)
    with torch.no_grad():
        mask_init, pred_init, acc_init, size_init = _filter_logic(curr_soft_out)
        # 计算全体准确率
        overall_pred_init = curr_soft_out.argmax(dim=1)
        overall_acc_init = (overall_pred_init == clean_labels).float().mean().item()
        logger.info(f"📊 [Stage 0] Initial (Before Propagation)")
        logger.info(f"   ├─ Selected: {int(size_init)} samples | Acc: {acc_init*100:.2f}% | Overall Acc: {overall_acc_init*100:.2f}%")
        logger.info(f"   └─ Distribution: argmax of current_fixed_labels")

    # for i in range(iterations):
    #     mode = args.sim_mode_1 if i == 0 else args.sim_mode_2
    #     refined_sim = get_weight_matrix(mode, raw_sim, neighbors_mh, curr_soft_out, args)
        
    #     # 排除自身权重进行投票
    #     voting_weights = refined_sim[:, 1:]
    #     weighted_votes = torch.einsum('nk,nkc->nc', voting_weights, curr_soft_out[neighbors_mh[:, 1:]])
    #     curr_soft_out = F.softmax(weighted_votes, dim=1)

    for i in range(iterations):
        # --- 🚀 [修改点]:输入模式控制 ---
        if i == 0:
            # 第一轮:必须使用原始的、带有历史信息的软标签 (Soft)
            # 因为这是我们的先验知识
            propagation_input = curr_soft_out
        else:
            # 第二轮及以后:[EXP9] 先决定是否进行模型预测融合,再投影+归一化,最后决定软/硬输入
            eps = float(getattr(args, 'knn1_soft_prop_eps', 1e-8))
            max_w = float(getattr(args, 'knn1_soft_prop_max_w', 1.0))

            # epoch-ramp 权重(与 EXP6 保持一致)
            denom = float(max(getattr(args, 'epochs', 1) - 1, 1))
            w_model = max_w * float(epoch) / denom
            w_model = float(np.clip(w_model, 0.0, max_w))

            cand = static_cand_mask.float()
            knn1_raw = curr_soft_out

            if getattr(args, 'enable_knn1_model_fuse', False) and (model_preds is not None):
                # 1) 模型预测融合: 先在 log 空间融合,再回到正常空间
                neighbor_idx = neighbors_mh[:, 1:]
                w_raw = raw_sim[:, 1:]
                w_norm = w_raw / (w_raw.sum(dim=1, keepdim=True) + eps)
                neighbor_model_probs = model_preds[neighbor_idx]
                geo = torch.einsum('nk,nkc->nc', w_norm, neighbor_model_probs)
                geo = geo / (geo.sum(dim=1, keepdim=True) + eps)

                log_knn = torch.log(knn1_raw + eps)
                log_geo = torch.log(geo + eps)
                log_fused = log_knn + w_model * log_geo
                log_fused = log_fused - log_fused.max(dim=1, keepdim=True)[0]
                base_dist = torch.exp(log_fused)
            else:
                # 不启用模型融合:直接使用原始 KNN1 分数
                base_dist = knn1_raw

            # 2) 投影到候选集 + 归一化
            base_dist = base_dist * cand
            base_dist = base_dist / (base_dist.sum(dim=1, keepdim=True) + eps)

            # 3) 决定软/硬输入
            if getattr(args, 'enable_knn1_soft_prop', False):
                propagation_input = base_dist
                if logger is not None and epoch % 10 == 0:
                    fuse_tag = "Geo+KNN" if getattr(args, 'enable_knn1_model_fuse', False) else "KNN-only"
                    logger.info(f" [EXP9] {fuse_tag}-SoftProp-Cand | w_model={w_model:.3f} (max_w={max_w:.3f})")
            else:
                hard_preds = base_dist.argmax(dim=1)
                propagation_input = F.one_hot(hard_preds, num_classes=args.num_classes).float()

        # 1. 计算边权重 
        # 注意:这里传入 propagation_input,意味着 DAES/Topology 也会基于这个硬标签计算熵/一致性
        # (硬标签的自身熵为0,但邻域熵依然有效)
        mode = args.sim_mode_1 if i == 0 else args.sim_mode_2
        refined_sim = get_weight_matrix(mode, raw_sim, neighbors_mh, propagation_input, args)
        
        # 2. 邻居投票 (Message Passing)
        # 排除自身权重 (col 0),只用邻居投票
        voting_weights = refined_sim[:, 1:]
        
        # [关键]:邻居传过来的也是 propagation_input (即硬标签)
        neighbor_vals = propagation_input[neighbors_mh[:, 1:]]
        
        weighted_votes = torch.einsum('nk,nkc->nc', voting_weights, neighbor_vals)
        
        # 3. 更新状态 (归一化)
        curr_soft_out = F.softmax(weighted_votes, dim=1)

        # 📊 Stage 1: 第 1 轮传播后 (KNN only)
        if i == 0:
            with torch.no_grad():
                mask_knn1, pred_knn1, acc_knn1, size_knn1 = _filter_logic(curr_soft_out)
                # 计算全体准确率
                overall_pred_knn1 = curr_soft_out.argmax(dim=1)
                overall_acc_knn1 = (overall_pred_knn1 == clean_labels).float().mean().item()
                logger.info(f"📊 [Stage 1] After 1st Propagation (KNN only)")
                logger.info(f"   ├─ Selected: {int(size_knn1)} samples | Acc: {acc_knn1*100:.2f}% | Overall Acc: {overall_acc_knn1*100:.2f}%")
                logger.info(f"   ├─ Change from Stage 0: Δsize={int(size_knn1-size_init):+d}, Δacc={(acc_knn1-acc_init)*100:+.2f}%")
                logger.info(f"   └─ Method: {args.sim_mode_1} propagation")

        # ==============================================================================
        # [EXP6] KNN1 分数 + 模型几何分数(log 空间)融合:在第二轮传播前修正 KNN 分布
        # - 模型权重 w_model: 随 epoch 线性从 0 -> max_w
        # - KNN 权重固定为 1(log 空间相加等价于乘积融合)
        # - 严格约束在原始候选标签集(static_cand_mask)上
        # ==============================================================================
        if i == 0 and getattr(args, 'enable_knn1_geo_fuse', False) and (model_preds is not None):
            eps = float(getattr(args, 'knn1_geo_fuse_eps', 1e-8))
            max_w = float(getattr(args, 'knn1_geo_fuse_max_w', 1.0))

            # epoch: 0-indexed; 让最后一个 epoch 的权重达到 max_w
            denom = float(max(getattr(args, 'epochs', 1) - 1, 1))
            w_model = max_w * float(epoch) / denom
            w_model = float(np.clip(w_model, 0.0, max_w))

            cand = static_cand_mask.float()

            # 1) 第 1 轮 KNN 分数(传播后分布)投影回候选集
            knn1 = curr_soft_out
            knn1 = knn1 * cand
            knn1 = knn1 / (knn1.sum(dim=1, keepdim=True) + eps)

            # 2) 基于当前模型预测的"几何分数":邻居模型 soft prob 的相似度加权平均
            neighbor_idx = neighbors_mh[:, 1:]
            w_raw = raw_sim[:, 1:]
            w_norm = w_raw / (w_raw.sum(dim=1, keepdim=True) + eps)
            neighbor_model_probs = model_preds[neighbor_idx]  # [N, K, C]
            geo = torch.einsum('nk,nkc->nc', w_norm, neighbor_model_probs)
            geo = geo / (geo.sum(dim=1, keepdim=True) + eps)

            geo = geo * cand
            geo = geo / (geo.sum(dim=1, keepdim=True) + eps)

            # 3) log 空间融合并回投影
            log_knn = torch.log(knn1 + eps)
            log_geo = torch.log(geo + eps)
            log_fused = log_knn + w_model * log_geo

            # 非候选标签直接屏蔽,避免 argmax 跑出候选集
            log_fused = torch.where(cand > 0, log_fused, torch.full_like(log_fused, -1e9))

            log_fused = log_fused - log_fused.max(dim=1, keepdim=True)[0]
            fused = torch.exp(log_fused)
            fused = fused / (fused.sum(dim=1, keepdim=True) + eps)

            curr_soft_out = fused

            if logger is not None and epoch % 10 == 0:
                logger.info(f" 🧪 [EXP6] KNN1-GeoFuse | w_model={w_model:.3f} (max_w={max_w:.3f})")
            
            # 📊 Stage 1.5: EXP6 融合后 (KNN + Model Geo)
            with torch.no_grad():
                mask_exp6, pred_exp6, acc_exp6, size_exp6 = _filter_logic(curr_soft_out)
                # 计算全体准确率
                overall_pred_exp6 = curr_soft_out.argmax(dim=1)
                overall_acc_exp6 = (overall_pred_exp6 == clean_labels).float().mean().item()
                logger.info(f"📊 [Stage 1.5] After EXP6 Fusion (KNN1 + Model Geo)")
                logger.info(f"   ├─ Selected: {int(size_exp6)} samples | Acc: {acc_exp6*100:.2f}% | Overall Acc: {overall_acc_exp6*100:.2f}%")
                logger.info(f"   ├─ Model weight: w_model={w_model:.3f} (epoch-ramped)")
                logger.info(f"   ├─ Change from Stage 1: Δsize={int(size_exp6-size_knn1):+d}, Δacc={(acc_exp6-acc_knn1)*100:+.2f}%")
                logger.info(f"   └─ Constraint: Within static candidate set")

    # 3. 最终筛选执行
    mask_2, pred_2, acc_rel_2, size_rel_2 = _filter_logic(curr_soft_out)
    mask_2 = mask_2.float()
    
    # 计算全体准确率
    overall_pred_final = curr_soft_out.argmax(dim=1)
    overall_acc_final = (overall_pred_final == clean_labels).float().mean().item()
    
    # 📊 Stage 2: 最终结果 (第 2 轮传播后)
    logger.info(f"📊 [Stage 2] Final Selection (After 2nd Propagation)")
    logger.info(f"   ├─ Selected: {int(size_rel_2)} samples | Acc: {acc_rel_2*100:.2f}% | Overall Acc: {overall_acc_final*100:.2f}%")
    
    # 根据启用的功能显示对比
    if getattr(args, 'enable_knn1_geo_fuse', False) and (model_preds is not None):
        # EXP6 启用: 对比 Stage 1.5
        if 'size_exp6' in locals():
            logger.info(f"   ├─ Change from Stage 1.5: Δsize={int(size_rel_2-size_exp6):+d}, Δacc={(acc_rel_2-acc_exp6)*100:+.2f}%")
        if getattr(args, 'enable_knn1_soft_prop', False):
            logger.info(f"   ├─ Method: {args.sim_mode_2} propagation (EXP8: soft input)")
        else:
            logger.info(f"   ├─ Method: {args.sim_mode_2} propagation (one-hot input)")
    else:
        # EXP6 未启用: 对比 Stage 1
        if 'size_knn1' in locals():
            logger.info(f"   ├─ Change from Stage 1: Δsize={int(size_rel_2-size_knn1):+d}, Δacc={(acc_rel_2-acc_knn1)*100:+.2f}%")
        logger.info(f"   ├─ Method: {args.sim_mode_2} propagation (one-hot KNN1)")
    
    logger.info(f"   └─ Total improvement: Δsize={int(size_rel_2-size_init):+d}, Δacc={(acc_rel_2-acc_init)*100:+.2f}%")
    logger.info(f"{'='*80}\n")
    
    # # ==============================================================================
    # # 🚀 2. [关键修改] 原型更新与扩充 (Prototype Update & Expansion)
    # # ==============================================================================
    # if proto_manager is not None:
    #     # A. 立即更新原型 (即使是 Epoch 0,也会基于刚刚筛选出的 mask_2 初始化原型)
    #     #    注意:我们信任 pred_2 (经过拓扑传播后的标签)
    #     proto_manager.update(features, mask_2.bool(), pred_2)
        
    #     # B. 基于新原型的全局预测
    #     #    sims: [N, C], proto_preds: [N]
    #     _, proto_preds = proto_manager.predict(features)
        
    #     # C. 三方共识扩充逻辑 (Tri-Consensus Expansion)
    #     #    条件: Model (Raw) == KNN (Propagated) == Prototype (Global)
    #     raw_model_preds = model_preds.argmax(dim=1)
    #     knn_preds = pred_2 # 这是拓扑传播后的硬标签
        
    #     tri_consensus_mask = (raw_model_preds == knn_preds) & (knn_preds == proto_preds)
        
    #     #    找出那些"被 mask_2 遗漏"但"满足三方共识"的样本
    #     #    mask_2 == 0 表示当前被认为是不可靠
    #     expansion_candidates = tri_consensus_mask & (mask_2 == 0)
    #     num_expanded = expansion_candidates.sum().item()
        
    #     if num_expanded > 0:
    #         # 1. 计算扩充样本的准确率 (仅用于日志,不用于决策)
    #         acc_exp = (knn_preds[expansion_candidates] == clean_labels[expansion_candidates]).float().mean().item()
            
    #         # 2. 正式加入可靠集
    #         mask_2[expansion_candidates] = 1.0
            
    #         # 3. 日志记录
    #         logger.info(f" 🌟 [Proto-Expansion] Added {num_expanded} samples via Tri-Consensus!")
    #         logger.info(f"    └─ Precision: {acc_exp:.2%} (vs Base Reliable: {acc_rel_2:.2%})")
            
    #         # 4. 可选:更新统计变量以便后续打印
    #         pred_2 = curr_soft_out.argmax(dim=1) # 保持 pred_2 更新
    #     else:
    #         logger.info(f" 🌟 [Proto-Expansion] No additional samples found this epoch.")
    # # 4. 状态更新与打捞分析 (保持原逻辑以维护历史队列)
    # # 计算 Pruned PL 用于打捞逻辑
    with torch.no_grad():
        target_unrel_mask = (~mask_2.bool())
        w_pruned = raw_sim[:, 1:] * torch.gather(mask_2.view(-1, 1).expand(-1, args.k_val).float(), 0, neighbors_mh[:, 1:])
        w_p_sum = w_pruned.sum(dim=1)
        w_p_norm = w_pruned / (w_p_sum.unsqueeze(1) + 1e-12)
        soft_out_pruned = torch.sum(F.embedding(neighbors_mh[:, 1:], curr_soft_out) * w_p_norm.view(N, -1, 1), dim=1)
        pruned_pl = soft_out_pruned.argmax(dim=1)

    # 计算 Model Geometry PL 用于双轨打捞
    # 1. 获取邻居的模型硬预测标签 [N, K]
    model_hard_labels = model_preds.argmax(dim=1) 
    neighbor_model_labels = model_hard_labels[neighbors_mh[:, 1:]] # [N, K]

    # 2. 将邻居标签转为 One-hot 编码 [N, K, num_classes]
    neighbor_model_onehot = F.one_hot(neighbor_model_labels, num_classes=args.num_classes).float()

    # 3. 准备相似度权重 [N, K, 1] 以便进行广播乘法
    weights = raw_sim[:, 1:].unsqueeze(-1)

    # 4. 加权求和得到类别的分布 [N, num_classes]
    geo_soft_out = torch.sum(neighbor_model_onehot * weights, dim=1)

    # 5. 现在可以安全地在 dim=1 上执行 argmax 了
    model_geo_pl = geo_soft_out.argmax(dim=1)

    # 更新历史记录
    # 🚀 [新增] 获取 Proto PL 用于三轨打捞
    # 假设在此之前已经通过 proto_manager.predict 获取了 proto_preds
    # 如果没有,可以在这里补算一下
    if proto_manager is not None:
         # 注意:这里需要对所有样本预测,不仅是可靠集
        _, proto_pl_all = proto_manager.predict(features)
    else:
        proto_pl_all = None

    # 更新历史记录 (传入 proto_pl_all)
    # NOTE: 三方共识版本不再维护 EMA 概率(不把 EMA 作为共识参与方)
    state_manager.update_history(mask_2, pruned_pl, model_geo_pl, proto_pl=proto_pl_all) # <--- 修改这里
# ==============================================================================
    # 📊 6. 日志输出 (全维度诊断版 - 完美适配您的需求)
    # ==============================================================================
    with torch.no_grad():
        # --- 1. 基础数据准备 ---
        clean_labels = clean_labels.to(device)
        # A. 原始模型预测 (不受图影响)
        raw_model_preds = model_preds.argmax(dim=1)
        # B. 拓扑传播后软标签预测 (您问的"当前软标签")
        prop_preds = curr_soft_out.argmax(dim=1)
        
        # 计算全局可靠性指标
        probs = curr_soft_out
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        norm_entropy = entropy / np.log(args.num_classes)
        avg_reliability = (1.0 - norm_entropy).mean().item()
        
        # --- 2. 计算各种 KNN 投票结果 ---
        neighbor_indices = neighbors_mh[:, 1:] # [N, K]
        neighbor_is_reliable = mask_2[neighbor_indices].bool() # [N, K]
        w_raw = raw_sim[:, 1:] # [N, K]

        # 分离权重
        w_reliable_only = w_raw * neighbor_is_reliable.float()
        w_unreliable_only = w_raw * (~neighbor_is_reliable).float()

        # [关键] 计算"不可靠邻居"的投票结果 (Unreliable-KNN Pseudo Label)
        # 看看孤岛周围的"坏邻居"到底投出了什么
        # 使用 curr_soft_out 作为邻居的意见
        neighbor_soft_vals = curr_soft_out[neighbor_indices]
        unrel_votes = torch.einsum('nk,nkc->nc', w_unreliable_only, neighbor_soft_vals)
        unrel_knn_pl = unrel_votes.argmax(dim=1)

        # [关键] 计算"可靠邻居"的投票结果 (Reliable-KNN / Pruned PL)
        rel_votes = torch.einsum('nk,nkc->nc', w_reliable_only, neighbor_soft_vals)
        rel_knn_pl = rel_votes.argmax(dim=1)

        # --- 3. 区域划分 ---
        # 计算可靠邻居强度
        reliable_strength = w_reliable_only.sum(dim=1)
        
        # 定义真正孤岛 (True Islands): 不可靠 & 无可靠邻居
        true_island_mask = target_unrel_mask & (reliable_strength == 0)
        num_islands = true_island_mask.sum().item()
        
        # 定义连接样本 (Connected): 不可靠 & 有可靠邻居
        true_connected_mask = target_unrel_mask & (reliable_strength > 0)
        num_connected = true_connected_mask.sum().item()

        # --- 4. 统计准确率 ---
        # 全局与可靠集
        acc_final_all = (prop_preds == clean_labels).float().mean().item()
        num_rel = mask_2.sum().item()
        acc_rel = (prop_preds[mask_2.bool()] == clean_labels[mask_2.bool()]).float().mean().item() if num_rel > 0 else 0

        # 真正孤岛指标
        if num_islands > 0:
            acc_isl_model = (raw_model_preds[true_island_mask] == clean_labels[true_island_mask]).float().mean().item()
            acc_isl_prop = (prop_preds[true_island_mask] == clean_labels[true_island_mask]).float().mean().item()
            acc_isl_unrel_knn = (unrel_knn_pl[true_island_mask] == clean_labels[true_island_mask]).float().mean().item()
        else:
            acc_isl_model = 0; acc_isl_prop = 0; acc_isl_unrel_knn = 0

        # 连接样本指标
        if num_connected > 0:
            acc_conn_model = (raw_model_preds[true_connected_mask] == clean_labels[true_connected_mask]).float().mean().item()
            acc_conn_prop = (prop_preds[true_connected_mask] == clean_labels[true_connected_mask]).float().mean().item()
            acc_conn_rel_knn = (rel_knn_pl[true_connected_mask] == clean_labels[true_connected_mask]).float().mean().item()
        else:
            acc_conn_model = 0; acc_conn_prop = 0; acc_conn_rel_knn = 0

        # --- 5. 打印详细日志 ---
        logger.info(f"\n" + "═"*80)
        logger.info(f"📊 [Topology Diagnostics] Epoch {epoch} | Total: {N}")
        
        # Block 1: 迭代演进
        logger.info(f" 🟢 [Step 1: Propagation] Final Acc (All): {acc_final_all:.2%} | Avg Reliability: {avg_reliability:.4f}")

        # Block 2: 可靠集
        logger.info(f" 🔵 [Step 2: Reliable Set] Count: {num_rel:<6} ({num_rel/N:.1%}) | Acc: {acc_rel:.2%}")

        # Block 3: 真正的孤岛 (分析重点)
        logger.info(f" 🔴 [Step 3.1: True Islands] (Unreliable & No Reliable Neighbors)")
        logger.info(f"    - Count: {num_islands:<6} ({num_islands/N:.1%})")
        if num_islands > 0:
            logger.info(f"      ├─ [Self] Raw Model Acc:       {acc_isl_model:.2%} (Model independent view)")
            logger.info(f"      ├─ [Peer] Unreliable-KNN Acc:  {acc_isl_unrel_knn:.2%} (Vote from other islands)")
            logger.info(f"      └─ [Fuse] Curr Soft Label Acc: {acc_isl_prop:.2%} (After propagation)")
            # 智能提示
            if acc_isl_unrel_knn > acc_isl_model:
                logger.info(f"      💡 Discovery: Unreliable neighbors provide better signal ({acc_isl_unrel_knn:.2%} > {acc_isl_model:.2%})!")

        # Block 4: 连接样本
        logger.info(f" 🟠 [Step 3.2: Connected Unreliable] (Has Reliable Neighbors)")
        logger.info(f"    - Count: {num_connected:<6} ({num_connected/N:.1%})")
        if num_connected > 0:
            logger.info(f"      ├─ [Self] Raw Model Acc:       {acc_conn_model:.2%}")
            logger.info(f"      ├─ [Fuse] Curr Soft Label Acc: {acc_conn_prop:.2%}")
            logger.info(f"      └─ [Peer] Reliable-KNN Acc:    {acc_conn_rel_knn:.2%} (Target for Salvage)")
        
        # Block 5: 打捞
        salvage_mask, salvage_labels, knn_src, geo_src = state_manager.get_salvage_mask()
        
        if salvage_mask is not None and salvage_mask.sum() > 0:
             # --- 🚀 [修复]:将所有涉及计算的张量统统移到 device (GPU) ---
             salvage_mask = salvage_mask.to(device)
             salvage_labels = salvage_labels.to(device)
             # clean_labels 已经在函数开头移到了 device,这里直接用
             
             num_sal = salvage_mask.sum().item()
             
             # 现在两边都在 GPU 上,可以安全计算
             acc_sal = (salvage_labels[salvage_mask] == clean_labels[salvage_mask]).float().mean().item()
             
             logger.info(f" 🟡 [Step 4: Salvage Ops] Promoted: {num_sal} | Precision: {acc_sal:.2%}")
        else:
             logger.info(f" 🟡 [Step 4: Salvage Ops] None this epoch.")

        logger.info("═"*80 + "\n")

    # 5. 返回结果
    return mask_2.float(), pred_2, pred_2, pred_2, curr_soft_out.float()


@torch.no_grad()
def get_features(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval(); all_features, all_predictions, all_indices = [], [], []
    for images, indices in loader:
        images = images.to(device, non_blocking=True)
        with autocast('cuda'):
            features = encoder(images)
            predictions = F.softmax(classifier(features), dim=1)
        all_features.append(F.normalize(features.float())); all_predictions.append(predictions.float()); all_indices.append(indices.cpu())
    all_features, all_predictions, all_indices = torch.cat(all_features), torch.cat(all_predictions), torch.cat(all_indices)
    return all_features[torch.argsort(all_indices)], all_predictions[torch.argsort(all_indices)]

class SoftMatchWeightManager:
    def __init__(self, num_samples, num_classes, n_sigma=2.0, momentum=0.99, device='cuda'): self.n_sigma, self.momentum, self.device = n_sigma, momentum, device; self.prob_model = torch.ones(num_samples, num_classes, device=device) / num_classes
    def __call__(self, preds, index, return_stats=False):
        self.prob_model[index] = self.momentum * self.prob_model[index] + (1 - self.momentum) * preds.detach(); max_probs_model = self.prob_model[index].max(dim=1)[0]; mu = max_probs_model.mean(); std = max_probs_model.std() if max_probs_model.size(0) > 1 else torch.tensor(1e-8, device=self.device); weights = torch.exp(-torch.pow(F.relu(mu - preds.max(dim=1)[0]), 2) / (2 * self.n_sigma * std**2 + 1e-8))
        return (weights.detach(), mu.item(), std.item()) if return_stats else weights.detach()











@torch.no_grad()
def evaluate(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval(); correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast('cuda'): outputs = classifier(encoder(images))
        _, predicted = torch.max(outputs, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

# ==============================================================================
# 🛠️ Helper 1: 显存安全的高精度 KNN
# ==============================================================================
def knn_search_pytorch_chunked(feats, k, num_heads=1, chunk_size=4096):
    original_matmul_precision = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False 
        N, D_dim = feats.shape
        if not feats.is_contiguous(): feats = feats.contiguous()
        
        if num_heads > 1:
            head_dim = D_dim // num_heads
            feats_ready = F.normalize(feats.view(N, num_heads, head_dim), p=2, dim=2).reshape(N, D_dim) 
        else:
            feats_ready = F.normalize(feats, p=2, dim=1)

        final_sims, final_indices = [], []
        with torch.no_grad():
            database_t = feats_ready.t()
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                sim_matrix = torch.mm(feats_ready[i:end_idx], database_t)
                batch_sims, batch_indices = torch.topk(sim_matrix, k=min(k+1, N), dim=1, largest=True, sorted=True)
                final_sims.append(batch_sims); final_indices.append(batch_indices)
        return torch.cat(final_sims, dim=0) / float(num_heads), torch.cat(final_indices, dim=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_precision

# ==============================================================================
# 🛠️ Helper 3: DAES 亲和度矩阵计算 (参数化版 - 严格去自身化)
# ==============================================================================
def get_adaptive_affinity_matrix(raw_D, neighbors_indices, current_soft_labels, args):
    """
    计算 DAES 动态边权重 (参数化版)。
    [关键修改]: 假设输入的 raw_D 和 neighbors_indices 已经是 (N, k) 形状，
    即已经剔除了 Top-1 (自身)，完全基于邻居进行计算。
    """
    # 1. 直接使用传入的 raw_D (已经是纯邻居距离)
    # [参数化] 空间温度
    att_temp = getattr(args, 'daes_spatial_temp', 0.5)
    
    # 计算空间注意力权重 (N, k, 1)
    # 这里 softmax 是在 k 个邻居之间进行的
    spatial_weights = F.softmax(raw_D / att_temp, dim=1).unsqueeze(-1)
    
    # 2. 获取邻居标签 (N, k, C)
    # neighbors_indices 也不包含自身,所以获取的是纯粹的邻居标签
    neighbor_labels = F.embedding(neighbors_indices, current_soft_labels) 
    
    # 3. 计算 Local Mean (局部均值) 用于熵计算
    # 这里的均值完全由邻居决定,代表"环境的看法"
    local_mean_raw = (neighbor_labels * spatial_weights).sum(dim=1)
    
    # [参数化] 熵敏感度 & 基础温度
    base_tau = getattr(args, 'daes_base_tau', 0.1)
    entropy_coeff = getattr(args, 'daes_entropy_coeff', 0.5)
    
    # 4. 计算熵 (反映邻域的一致性/混乱程度)
    local_entropy = -torch.sum(local_mean_raw * torch.log(local_mean_raw + 1e-8), dim=1)
    norm_entropy = local_entropy / math.log(args.num_classes)
    
    # 动态温度 tau: 邻域越乱(熵越高),tau越大,权重分布越平滑
    tau_dynamic = base_tau + (torch.pow(norm_entropy, 2) * entropy_coeff) 
    tau_dynamic = tau_dynamic.unsqueeze(1) 

    # [参数化] 相似度非线性变换 (Power)
    sim_power = getattr(args, 'daes_sim_power', 2.0)
    
    # 5. 计算最终权重
    # raw_D 是 (N, k),计算出的 weights 也是 (N, k)
    scaled_sim = torch.pow(raw_D, sim_power) / tau_dynamic
    
    max_val, _ = scaled_sim.max(dim=1, keepdim=True)
    weights = torch.exp(scaled_sim - max_val.detach()) # Subtract max for stability

    return weights
# ==============================================================================
# 🛠️ Helper 4: 通用权重获取入口
# # ==============================================================================
# def get_weight_matrix(mode, raw_D, neighbors_indices, ref_soft_labels, args):
#     if mode == 'daes':
#         return get_adaptive_affinity_matrix(raw_D, neighbors_indices, ref_soft_labels, args)
#     elif mode == 'exp':
#         return torch.exp(raw_D / 0.1)
#     else:
#         return raw_D

# ==============================================================================
# 🛠️ Helper 4: 通用权重获取入口 (更新版)
# ==============================================================================
def get_weight_matrix(mode, raw_D, neighbors_indices, ref_soft_labels, args):
    """
    根据 sim_mode 生成亲和矩阵
    输入 raw_D, neighbors_indices 均为 [N, K+1] (包含自身在 col 0)
    输出 [N, K+1]
    """
    # 1. 拆分:分离自身和邻居
    # col 0 是自身 (sim usually 1.0 or max), col 1: 是邻居
    self_sim = raw_D[:, 0:1] # [N, 1]
    neighbor_sim = raw_D[:, 1:] # [N, K]
    neighbor_indices = neighbors_indices[:, 1:] # [N, K]

    refined_neighbors = neighbor_sim # 默认初始化

    # 2. 根据模式处理邻居部分
    if mode == 'daes':
        # DAES 逻辑:利用熵和空间分布重新加权
        refined_neighbors = get_adaptive_affinity_matrix(neighbor_sim, neighbor_indices, ref_soft_labels, args)
        
    elif mode == 'topology':
        # 拓扑一致性逻辑 (调用您代码中原有的 get_topology_guided_affinity)
        # 注意:get_topology_guided_affinity 需要完整的 [N, K+1] 输入
        # 这里我们做个特殊处理,直接调用原函数并取邻居部分
        full_refined, _ = get_topology_guided_affinity(
            raw_D, neighbors_indices, ref_soft_labels, args.num_classes,
            rel_mode=getattr(args, 'topology_rel_mode', 'masked_entropy'),
            gamma=getattr(args, 'topology_rel_gamma', 2.0),
            eps=getattr(args, 'topology_rel_eps', 1e-12),
        )
        refined_neighbors = full_refined[:, 1:]
        
    elif mode == 'exp':
        # 指数逻辑:简单锐化
        refined_neighbors = torch.exp(neighbor_sim / 0.1)
        
    elif mode == 'linear' or mode == 'none':
        # 线性/原始
        refined_neighbors = neighbor_sim
    
    # 3. 合并:将自身加回去
    # 对于自身权重,确保它在数值上占优,保证自身特征不丢失
    if mode == 'exp':
        self_refined = torch.exp(self_sim / 0.1)
    elif mode == 'daes' or mode == 'topology':
        # 在高级模式下,自身权重通常设为稍大于邻居最大权重
        self_refined = refined_neighbors.max(dim=1, keepdim=True)[0] * 1.5
        # 防止全0
        self_refined = torch.clamp(self_refined, min=1.0)
    else:
        self_refined = self_sim

    # 拼接回 [N, K+1]
    refined_D = torch.cat([self_refined, refined_neighbors], dim=1)
    return refined_D



def train_unified_single_stream(args, encoder, classifier, device,
                                unified_loader, optimizer, softmatch_manager,
                                logger, num_classes, knn_pl, model_pl, knn_scores,
                                saved_softmatch_weights, proto_manager=None):
    """
    统一单流训练：通过重要性采样后的 DataLoader 遍历所有数据
    
    Args:
        saved_softmatch_weights: [N] tensor，用于在训练过程中累积更新
    
    Returns:
        avg_unreliable_softmatch_weight: 本 epoch 不可靠集的平均 SoftMatch 权重
    """
    encoder.train()
    classifier.train()
    scaler = GradScaler()
    
    knn_pl, model_pl = knn_pl.to(device), model_pl.to(device)
    knn_scores = knn_scores.to(device)
    
    # 统计变量
    total_loss_s, total_loss_c_self = 0.0, 0.0
    num_sup, num_unsup = 0, 0
    
    # 累积不可靠集的 SoftMatch 权重
    unreliable_softmatch_sum = 0.0
    unreliable_count = 0
    
    # 遍历整个统一数据集(基于 epoch)
    for batch_data in unified_loader:
        weak_imgs, strong_imgs, labels, is_reliable, indices = batch_data
        weak_imgs = weak_imgs.to(device)
        strong_imgs = strong_imgs.to(device)
        labels = labels.to(device)
        is_reliable = is_reliable.to(device)
        indices = indices.to(device)
        
        optimizer.zero_grad()
        
        # 分离可靠集和不可靠集的索引
        reliable_mask = is_reliable.bool()
        unreliable_mask = ~reliable_mask
        
        inputs_forward = []
        
        # ========== 可靠集处理 ==========
        if reliable_mask.sum() > 0:
            rel_weak = weak_imgs[reliable_mask]
            rel_strong = strong_imgs[reliable_mask]
            rel_labels = labels[reliable_mask]
                        # 目的: 让类中心紧跟当前 Batch 训练样本的特征变化
            if proto_manager is not None:
                with torch.no_grad():
                    # 1. 提取当前模型的特征 (使用弱增强版本,更干净)
                    # 使用 .detach() 确保不影响梯度回传
                    rel_feats_curr = encoder(rel_weak).detach()
                    
                    # 2. 构造局部掩码 (在这里全是可靠的)
                    batch_rel_mask = torch.ones(rel_feats_curr.shape[0], device=device, dtype=torch.bool)
                    
                    # 3. 更新原型
                    # 注意: 由于是 Batch 级更新,建议 proto_manager 内部的 ema_alpha 设置得较高 (如 0.999)
                    # 或者在这里手动控制权重,但通常 EMA 机制本身能处理
                    proto_manager.update(rel_feats_curr, batch_rel_mask, rel_labels)
            # ------------------------------------------------------------------

            # Mixup + Label Smoothing
            s_labels = F.one_hot(rel_labels.long(), num_classes).float()
            s_labels = s_labels * (1 - args.lsr) + args.lsr / num_classes
            
            B_s = rel_weak.size(0)
            perm_w = torch.randperm(B_s, device=device)
            lam_w = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            mix_w = lam_w * rel_weak + (1 - lam_w) * rel_weak[perm_w]
            mix_l_w = lam_w * s_labels + (1 - lam_w) * s_labels[perm_w]
            
            perm_s = torch.randperm(B_s, device=device)
            lam_s = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            mix_s = lam_s * rel_strong + (1 - lam_s) * rel_strong[perm_s]
            mix_l_s = lam_s * s_labels + (1 - lam_s) * s_labels[perm_s]
            
            inputs_forward.extend([mix_w, mix_s])
            num_sup += B_s
        else:
            mix_l_w, mix_l_s = None, None
            B_s = 0
        

        # ========== 统一前向传播 ==========
        if len(inputs_forward) > 0:
            with autocast('cuda'):
                all_inputs = torch.cat(inputs_forward)
                all_logits = classifier(encoder(all_inputs))
                
                current_idx = 0
                loss_s = torch.tensor(0.0, device=device)
                loss_c_self = torch.tensor(0.0, device=device)
                
                # 计算可靠集损失
                if B_s > 0:
                    logits_w = all_logits[current_idx : current_idx + B_s]
                    current_idx += B_s
                    logits_s = all_logits[current_idx : current_idx + B_s]
                    current_idx += B_s
                    
                    loss_w = -torch.sum(F.log_softmax(logits_w, 1) * mix_l_w, 1).mean()
                    loss_s_strong = -torch.sum(F.log_softmax(logits_s, 1) * mix_l_s, 1).mean()
                    loss_s = (loss_w + loss_s_strong) * 0.5
                    total_loss_s += loss_s.item() * B_s
                

        
        # 总损失
        total_loss = loss_s
        
        if total_loss > 0 and not torch.isnan(total_loss):
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    # Epoch 结束后的统计
    avg_loss_s = total_loss_s / num_sup if num_sup > 0 else 0
    avg_loss_c_self = total_loss_c_self / num_unsup if num_unsup > 0 else 0
    avg_unreliable_softmatch = unreliable_softmatch_sum / max(unreliable_count, 1)
    
    logger.info(f"  -> [Train Loss] Sup={avg_loss_s:.4f}, SSL-Self={avg_loss_c_self:.4f}.")
    
    return avg_unreliable_softmatch

def log_tri_consensus_diagnostics(logger, epoch, true_labels, 
                                  unreliable_indices, 
                                  model_pl, knn_pl, proto_pl,
                                  device):
    """诊断：Model vs KNN vs Prototype 在不可靠集上的多方博弈（三方版）。

    说明：该脚本不再把 EMA 预测作为共识参与方，因此这里也不再统计 EMA。
    """
    if len(unreliable_indices) == 0:
        return

    logger.info(f"\n📐 [Triangle Diagnostics] Epoch {epoch} | Unreliable Set: {len(unreliable_indices)}")

    idx = torch.tensor(unreliable_indices, device=device).long()
    target = true_labels[idx]

    # 获取预测
    p_model = model_pl[idx]
    p_knn   = knn_pl[idx]
    p_proto = proto_pl[idx]

    # 1) 单体准确率对比
    acc_model = (p_model == target).float().mean().item() * 100
    acc_knn   = (p_knn == target).float().mean().item() * 100
    acc_proto = (p_proto == target).float().mean().item() * 100
    logger.info(f"  ├─ 🎯 Individual Acc: Model={acc_model:.2f}% | KNN={acc_knn:.2f}% | Proto={acc_proto:.2f}%")

    # 2) 两两共识分析
    def analyze_pair(pred1, pred2):
        agree_mask = (pred1 == pred2)
        num_agree = agree_mask.sum().item()
        rate = num_agree / len(pred1) * 100
        if num_agree > 0:
            acc_consensus = (pred1[agree_mask] == target[agree_mask]).float().mean().item() * 100
        else:
            acc_consensus = 0.0
        return rate, acc_consensus

    rate_mk, acc_mk = analyze_pair(p_model, p_knn)
    rate_mp, acc_mp = analyze_pair(p_model, p_proto)
    rate_kp, acc_kp = analyze_pair(p_knn, p_proto)

    logger.info(f"  ├─ 🤝 Consensus Analysis (Agreement Rate / Consensus Acc):")
    logger.info(f"  │   ├─ Model & KNN:   Rate={rate_mk:.1f}% | Acc={acc_mk:.2f}%")
    logger.info(f"  │   ├─ Model & Proto: Rate={rate_mp:.1f}% | Acc={acc_mp:.2f}%")
    logger.info(f"  │   └─ KNN & Proto:   Rate={rate_kp:.1f}% | Acc={acc_kp:.2f}%")

    # 3) 三方共识
    grand_mask = (p_model == p_knn) & (p_knn == p_proto)
    num_grand = grand_mask.sum().item()
    if num_grand > 0:
        acc_grand = (p_model[grand_mask] == target[grand_mask]).float().mean().item() * 100
    else:
        acc_grand = 0.0
    logger.info(f"  └─ 🌟 Grand Consensus (All 3 agree): {num_grand} samples | Acc={acc_grand:.2f}%")
def run_single_experiment(args):
    start_time = time.time()
    set_seed(args.seed)

    # # [EXP6/EXP8] 为避免不同实验互相覆盖输出目录,自动为 exp_name 增加后缀
    # _exp_suffix = None
    # if getattr(args, 'enable_knn1_geo_fuse', False):
    #     _exp_suffix = "_exp6_knn1_geofuse"
    # elif getattr(args, 'enable_knn1_soft_prop', False):
    #     _exp_suffix = "_exp8_knn1_softprop"
    # if isinstance(_exp_suffix, str) and isinstance(args.exp_name, str) and (not args.exp_name.endswith(_exp_suffix)):
    #     args.exp_name = args.exp_name + _exp_suffix

    log_dir = os.path.join(args.out, args.exp_name, f"seed_{args.seed}")
    logger = setup_logger(log_dir, to_console=True)
    
    # 0. 本脚本仅运行第一阶段(不做蒸馏更新/标签覆写/系统重置)
    total_epochs = args.epochs
    
    logger.info(f"--- Starting Dynamic Strategy Run with Seed: {args.seed} ---")
    logger.info(f"Settings: {vars(args)}")
    logger.info(f"📅 Schedule: Stage-1 only | Total {total_epochs} eps")

    device = torch.device(f"cuda:{args.cuda_dev}" if torch.cuda.is_available() else "cpu")
    args.seed_dataset = args.seed
    
    # 3. 数据加载与预处理
    num_classes_map = {'CIFAR10': 10, 'CIFAR100': 100, 'CIFAR100H': 100, 'CUB200': 200, 'Treeversity': 6, 'Benthic': 8, 'Plankton': 10, 'Synthetic': 6}
    num_classes = num_classes_map[args.dataset]
    args.num_classes = num_classes
    
    weak_t, strong_t, test_t = get_pals_transforms(args.dataset)
    
    # --- Dataset Loading Logic ---
    if args.dataset in ['CIFAR10', 'CIFAR100', 'CIFAR100H']:
        is_h = 'H' in args.dataset
        BaseClass = CIFAR100Partial if '100' in args.dataset else CIFAR10Partial
        base_train_ds = BaseClass(args, train=True, download=True, transform=None)
        
        # 初始化修改掩码
        if not hasattr(base_train_ds, 'modified_mask'):
            base_train_ds.modified_mask = np.zeros(len(base_train_ds), dtype=bool)
            
        if hasattr(base_train_ds, 'partial_noise'):
            logger.info(f"Generating simulated NPLL noise for {args.dataset} (pr={args.pr}, nr={args.nr})")
            if '100' in args.dataset:
                base_train_ds.partial_noise(args.pr, args.nr, heirarchical=is_h)
            else:
                base_train_ds.partial_noise(args.pr, args.nr)
                
        TestClass = datasets.CIFAR100 if '100' in args.dataset else datasets.CIFAR10
        test_ds = TestClass(root=args.train_root, train=False, download=True, transform=test_t)

    elif args.dataset == 'CUB200':
        base_train_ds = CUB200Partial(args, train=True, transform=None)
        base_train_ds.partial_noise(args.pr, args.nr)
        test_ds = CUB200Partial(args, train=False, transform=test_t)
        
    # 备份原始噪声标签 (Static Anchor),用于可靠集筛选的基准
    if not hasattr(base_train_ds, 'original_soft_labels'):
        base_train_ds.original_soft_labels = base_train_ds.soft_labels.copy()
        logger.info(" 🔒 [Backup] Original noisy soft labels backed up for robust screening.")
        
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 4. 模型与状态初始化
    encoder, feature_dim = get_base_encoder(args.network, args.dataset)
    encoder = encoder.to(device)
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # 注意:state_manager 的总长度设为 total_epochs
    state_manager = TemporalStateManager(len(base_train_ds), num_classes, total_epochs, history_len=args.history_len, use_disambiguation=True)
    softmatch_manager = SoftMatchWeightManager(len(base_train_ds), num_classes, device=device)
    
    # 差异化学习率策略 (Fine-tuning 范式)
    if args.dataset in ['CUB200', 'Treeversity', 'Benthic', 'Plankton']:
        # 预训练骨干网络使用较小的学习率 (通常为基础 LR 的 0.1 或 0.01)
        encoder_lr = args.lr * 0.01 
        logger.info(f"Fine-tuning mode: Encoder LR={encoder_lr}, Classifier LR={args.lr}")
        optimizer = optim.SGD([
            {'params': encoder.parameters(), 'lr': encoder_lr},
            {'params': classifier.parameters(), 'lr': args.lr}
        ], momentum=args.momentum, weight_decay=args.wd)
    else:
        # 从头训练 (CIFAR等) 使用统一学习率
        optimizer = optim.SGD(list(encoder.parameters()) + list(classifier.parameters()), 
                              lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # [关键] Phase 1 调度器:T_max = args.epochs
    # 动态解析学习率调度器
    if args.lr_scheduler == 'step':
        # 假设命令行或 args 中存在默认的衰减轮次，例如 [60, 120, 160]
        milestones = getattr(args, 'lr_decay_epochs', [60, 120, 160])
        gamma = getattr(args, 'lr_decay_rate', 0.2)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        logger.info(f"Using StepLR: milestones={milestones}, gamma={gamma}")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        logger.info("Using CosineAnnealingLR")
    saved_softmatch_weights = torch.ones(len(base_train_ds), dtype=torch.float32)

    best_test_acc = 0.0
    

    # 初始化特征空间原型 (Prototypes)
    proto_manager = PrototypeManager(num_classes, feature_dim, ema_alpha=0.9, device=device)
    
    # ==============================================================================
    # 5. 主训练循环(仅第一阶段)
    # ==============================================================================
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        logger.info(f"======== Epoch {epoch+1}/{total_epochs} ========")

        # 5.1 特征提取与可靠性筛选
        feature_loader = DataLoader(FeatureExtractionDataset(base_train_ds, test_t), batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
        features, model_preds = get_features(encoder, classifier, feature_loader, device)
        
        class MockTrainloader:
            def __init__(self, dataset): self.dataset = dataset
        
        # 执行高级筛选
        selected_mask, selected_labels, knn_pl, model_pl, knn_scores = \
            reliable_pseudolabel_selection_advanced(logger, args, device, MockTrainloader(base_train_ds), features, epoch, state_manager, model_preds, proto_manager)
        
        # 1. 更新原型 (仅使用可靠集)
        proto_manager.update(features, selected_mask.bool(), selected_labels)
        
        # 2. 全局原型预测 & 历史队列更新 (在重置判断之前执行,以确保获取最新的 Salvage Set)
        _, proto_preds_all = proto_manager.predict(features)
        
        # 3. 三方队列共识更新(Model / KNN / Proto)+ 稳定打捞集提取
        with torch.no_grad():
            proto_logits = torch.matmul(features, proto_manager.prototypes.T) / 0.1
            proto_soft_preds = F.softmax(proto_logits, dim=1)

        state_manager.update_tri_consensus(
            is_reliable_mask=selected_mask,
            p_model=model_preds,
            p_knn=knn_scores,
            p_proto=proto_soft_preds,
        )
        tri_stable_mask, tri_stable_labels = state_manager.get_stable_tri_mask()

        # ==============================================================================
        # Stage-1 only:不做 SYSTEM RESET / 标签覆写 / 蒸馏更新
        # ==============================================================================

        # 4. 诊断日志(三方版:不再统计 EMA 预测)
        reliable_indices = torch.where(selected_mask > 0)[0]
        unreliable_indices = torch.where(selected_mask == 0)[0]
        unrel_indices_list = unreliable_indices.cpu().tolist()

        log_tri_consensus_diagnostics(
            logger, epoch, 
            true_labels=torch.tensor(base_train_ds.clean_labels, device=device),
            unreliable_indices=unrel_indices_list,
            model_pl=model_preds.argmax(dim=1),
            knn_pl=knn_pl,
            proto_pl=proto_preds_all,
            device=device
        )
        

        # ==============================================================================
        # 5.4 构建采样器
        # ==============================================================================
        logger.info(" -> [Single-Stream] Constructing Sampler & Refinement...")

        # 准备变量:salvage_mask 基于"三方队列共识"稳定掩码
        salvage_mask = tri_stable_mask
        salvaged_labels = tri_stable_labels
        
        # 旧的对比逻辑 (仅用于日志)
        old_salvage_mask, _, _, _ = state_manager.get_salvage_mask()
        if salvage_mask is not None:
             with torch.no_grad():
                clean_l = torch.tensor(base_train_ds.clean_labels, device=device)
                acc_tri = (salvaged_labels.to(device)[salvage_mask.to(device).bool()] == clean_l[salvage_mask.to(device).bool()]).float().mean().item() * 100 if salvage_mask.sum() > 0 else 0.0
                num_old = old_salvage_mask.sum().item() if old_salvage_mask is not None else 0
                logger.info(f" 🔬 [Salvage Check] New Tri-Stable: {salvage_mask.sum().item()} (Acc: {acc_tri:.2f}%) | Old Multi-Track: {num_old}")

        # 决定是否在训练中使用打捞样本
        # 策略:只要能打捞出来,就视为 Active Sample (在 Phase 2 尤为重要)
        detected_count = salvage_mask.sum().item() if salvage_mask is not None else 0
        use_salvage_for_training = (detected_count > 0)
        
        salvage_indices = []
        if use_salvage_for_training:
            salvage_indices = torch.where(salvage_mask > 0)[0].cpu().tolist()
            logger.info(f" 🚀 [Active] Promoting {len(salvage_indices)} salvaged samples to training pool.")
        
        salvage_set = set(salvage_indices)
        
        # 排除冲突:Reliable Set 中剔除已经是 Salvage 的 (虽然上面做了互斥,这里双重保险)
        real_reliable_indices = [idx for idx in reliable_indices.cpu().tolist() if idx not in salvage_set]
        # 剩下的是不可靠
        final_unreliable_indices = [idx for idx in unreliable_indices.cpu().tolist() if idx not in salvage_set]

        unified_data_list = []
        sampling_weights_aligned = []
        
        # --- 统一权重计算 ---
        total_target = len(base_train_ds)
        total_active_count = len(salvage_indices) + len(real_reliable_indices)
        unified_weight = max(0.0, total_target / max(total_active_count, 1))
        
        # A. 打捞集 (Salvaged)
        for idx in salvage_indices:
            unified_data_list.append((idx, salvaged_labels[idx].item(), True)) 
            sampling_weights_aligned.append(unified_weight)
            
        # B. 不可靠集 (Unreliable) -> 权重 0.0
        for idx in final_unreliable_indices:
            unified_data_list.append((idx, -1, False)) 
            sampling_weights_aligned.append(0.0) 
            
        # C. 可靠集 (Reliable)
        for idx in real_reliable_indices:
            unified_data_list.append((idx, selected_labels[idx].item(), True))
            sampling_weights_aligned.append(unified_weight)

        logger.info(f" >> [Sampler] Salvaged: {len(salvage_indices)} | Reliable: {len(real_reliable_indices)} | Unified Weight: {unified_weight:.2f}x")

        # 构建 Loader
        unified_dataset = UnifiedSSLDataset(base_train_ds, unified_data_list, weak_t, strong_t)
        sampler = WeightedRandomSampler(weights=sampling_weights_aligned, num_samples=total_target, replacement=True)
        unified_loader = DataLoader(unified_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        # ==============================================================================
        # 5.6 🚀 执行混合单流训练
        # ==============================================================================
        avg_sm_weight = train_unified_single_stream(
            args, encoder, classifier, device, unified_loader, optimizer, softmatch_manager,
            logger, num_classes, knn_pl, model_pl, knn_scores,
             saved_softmatch_weights, proto_manager
        )

        scheduler.step()
        
        # 评估与保存
        test_acc = evaluate(encoder, classifier, test_loader, device)
        if test_acc > best_test_acc: best_test_acc = test_acc
        
        # 本脚本不使用 EMA teacher / EMA 共识(也不维护 EMA 预测均值)

        wandb.log({'Test Accuracy': test_acc, 'Best Accuracy': best_test_acc, 'LR': optimizer.param_groups[0]['lr']}, step=epoch+1)
        logger.info(f"Epoch {epoch+1} Summary: Acc={test_acc:.2f}% | Best={best_test_acc:.2f}% | Time: {time.time()-epoch_start_time:.2f}s\n")

    wandb.finish()
    return best_test_acc, test_acc, time.time() - start_time
# ==============================================================================
#                      MAIN (MODIFIED FOR DUAL STATS)
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    wandb.init(mode="disabled")
    all_best_accuracies, all_final_epoch_accuracies, all_durations = [], [], []
    
    master_log_dir = os.path.join(args.out, args.exp_name)
    master_logger = setup_logger(master_log_dir, "master_log.txt", is_master=True)
    master_logger.info("========================= Starting Experiment Series =========================")
    master_logger.info(f"Base Settings: {vars(args)}\n" + "="*80)

    for i, seed in enumerate(args.seeds):
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        
        master_logger.info(f"--- Starting Run {i+1}/{len(args.seeds)} with Seed: {seed} ---")
        
        # <<< --- MODIFICATION START --- >>>
        best_acc, final_acc, duration = run_single_experiment(run_args)
        
        all_best_accuracies.append(best_acc)
        all_final_epoch_accuracies.append(final_acc) # 收集 Final Acc
        all_durations.append(duration)
        
        master_logger.info(f"--- Run {i+1} Finished. Duration: {duration/60.0:.2f} min | Best Acc: {best_acc:.2f}% | Final Acc: {final_acc:.2f}% ---\n")
        # <<< --- MODIFICATION END --- >>>
        
    # <<< --- MODIFICATION START --- >>>
    mean_best_acc = np.mean(all_best_accuracies)
    std_best_acc = np.std(all_best_accuracies)
    mean_final_acc = np.mean(all_final_epoch_accuracies) # 计算 Final Acc 均值
    std_final_acc = np.std(all_final_epoch_accuracies)   # 计算 Final Acc 标准差
    avg_duration_minutes = np.mean(all_durations) / 60
    # <<< --- MODIFICATION END --- >>>

    master_logger.info("========================= FINAL SUMMARY =========================")
    master_logger.info(f"Experiment Name: {args.exp_name}\n")
    master_logger.info(f"Average Run Duration: {avg_duration_minutes:.2f} min\n")
    
    # <<< --- MODIFICATION START --- >>>
    master_logger.info(f"Individual Best Accuracies: {[f'{acc:.2f}%' for acc in all_best_accuracies]}")
    master_logger.info(f"Individual Final Epoch Accuracies: {[f'{acc:.2f}%' for acc in all_final_epoch_accuracies]}")
    
    master_logger.info(f"--> Final Reported (Best Acc): {mean_best_acc:.2f}% ± {std_best_acc:.2f}%")
    master_logger.info(f"--> Final Reported (Final Epoch Acc): {mean_final_acc:.2f}% ± {std_final_acc:.2f}%")
    # <<< --- MODIFICATION END --- >>>
    
    master_logger.info("="*80)


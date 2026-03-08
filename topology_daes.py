"""python topology_daes.py \
  --exp_name "PALS_softMatch/CIFAR100/consensus_power2/exp_daes/delta0.5_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus1.0_rectify" \
  --dataset CIFAR100 \
  --seeds 1 2 3 \
  --num_workers 1 \
  --pr 0.05 \
  --nr 0.5 \
  --lr 0.1 \
  --wd 0.001 \
  --feature_consistency_weight 0.0 \
  --delta 0.5 \
  --detailed_log \
  --sim_mode_1 topology_entropy

python topology_daes.py \
  --exp_name "PALS_softMatch/CIFAR100/consensus_power2/exp_daes/delta0.5_model_predict_123_p0.1n0.0_bs256_rand_lam_no_cross_mixup_noncensus1.0_rectify" \
  --dataset CIFAR100 \
  --seeds 1 2 3 \
  --num_workers 1 \
  --pr 0.1 \
  --nr 0.0 \
  --lr 0.1 \
  --wd 0.001 \
  --feature_consistency_weight 0.0 \
  --delta 0.5 \
  --detailed_log \
  --sim_mode_1 topology_entropy  

python topology_daes.py \
    --dataset Treeversity \
    --lpi 10 \
    --out ./topology_daes \
    --exp_name topology_daes/Treeversity/LPI10_Run_1del1_top_daes_top_daesdel_0.75_lsr0.0 \
    --network R50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --sim_mode_1 topology_entropy \
    --sim_mode_2 topology_entropy \
    --delta 0.75 --lsr 0.0 \
    --lr_scheduler step \
    --lr_decay_epochs 60 \
    --lr_decay_rate 0.2 \
    --seeds 1 \
    --detailed_log    --cuda_dev 1  
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
import argparse
import os
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import logging
from PIL import Image
import faiss
from torch.amp import autocast, GradScaler
import torch.optim as optim
import wandb
import sys
import pandas as pd # CUB200依赖
import itertools
from torchvision.models import resnet18, resnet50
# 假设您的工具函数在以下路径
from data.dataset import CUB200Partial, CIFAR10Partial, CIFAR100Partial
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy ,ImageNetPolicy
from data.crowdsource import *
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

# ==============================================================================
#                      Section 1: 参数与配置
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Ultimate Hybrid PALS-SSL Framework with Three-Phase Training')
    # 基本设置
    parser.add_argument('--exp_name', type=str, default='HybridPALS_ThreePhase_Run', help='Experiment name.')
    # parser.add_argument('--dataset', type=str, default='CIFAR100', 
    #                     choices=['CIFAR10', 'CIFAR100', 'CIFAR100H', 'CUB200', 
    #                              'Treeversity', 'Benthic', 'Plankton'])
    # 在 parse_args() 函数中修改：
    parser.add_argument('--dataset', type=str, default='CIFAR100', 
                        choices=['CIFAR10', 'CIFAR100', 'CIFAR100H', 'CUB200', 
                                'Treeversity', 'Benthic', 'Plankton',
                                ])
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--out', type=str, default='./topology_daes', help='Directory for output')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1], help='List of random seeds.')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    
    # 部分标签 (PLL) 设置
    parser.add_argument('--pr', type=float, default=0.1, help='partial ratio (q)')
    parser.add_argument('--nr', type=float, default=0.0, help='noise ratio (eta)')
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
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='Epochs for linear LR warmup.')
    parser.add_argument('--lr_warmup_multiplier', type=float, default=0, help='Initial LR multiplier for warmup.')
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
    parser.add_argument('--feature_consistency_weight', type=float, default=0.5, 
                        help='Weight for SimSiam feature consistency loss on remaining samples.')
    parser.add_argument('--inter_contrast_weight', type=float, default=0.3, 
                        help='Weight for the inter-sample supervised contrastive loss.')
    # KNN & 平衡参数
    # KNN & 平衡参数
    parser.add_argument('--k_val', type=int, default=15, help='k for knn')
    parser.add_argument('--knn_iterations', type=int, default=2, help='Number of KNN purification iterations.')
    parser.add_argument('--delta', type=float, default=0.25, help='example selection quantile')
    # --- 🚀 为方法 A (weighted) 添加这些参数 ---
    parser.add_argument('--start_correct', type=int, default=0, help='(Used by the selection function)')  
    parser.add_argument('--epoch', type=int, default=100, help='(Dummy epoch for selection function)')  
    parser.add_argument('--conf_th_h', type=float, default=1.0, help='(Used by the selection function)')
    parser.add_argument('--conf_th_l', type=float, default=1.0, help='(Used by the selection function)')
    # --- 添加结束 ---
    parser.add_argument('--unreliable_batch_size', type=int, default=32, 
                    help='Specific batch size for the unreliable set.')
    # 日志
    parser.add_argument('--detailed_log', action='store_true', help='Enable detailed diagnostic logging.')
    # ... 原有参数 ...
    # --- 🚀 实验控制参数 ---
    # 在你的 arg parser 设置中修改这两个参数
    parser.add_argument('--sim_mode_1', type=str, default='daes',  # <--- 默认设为 daes
                        choices=['D', 'exp', 'daes','topology_entropy'], # <--- 加入 daes
                        help='Similarity measure for Stage 1')

    parser.add_argument('--sim_mode_2', type=str, default='daes', 
                        choices=['D', 'exp',  'daes','topology_entropy'], 
                        help='Similarity measure for Stage 2')
    # ...
    return parser.parse_args()

# (在 Section 2: 数据处理与模型)

# (在 Section 2: 数据处理与模型)

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
            # (这是在破坏“封装”，但这是在你设定的约束下唯一可行的方法)
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
class SSLReadyDataset(Dataset):
    def __init__(self, base_dataset, indices, labels, weak_t, strong_t):
        self.base_dataset, self.indices, self.labels = base_dataset, indices, labels
        self.weak_t, self.strong_t = weak_t, strong_t
        
        # --- (修改) ---
        # 我们需要明确区分 CUB200 和 Crowdsource
        self.is_cub = isinstance(self.base_dataset, CUB200Partial)
        self.is_crowd = isinstance(self.base_dataset, Crowdsource)
        # --- (修改结束) ---

    def __len__(self): 
        return len(self.indices)
        
    def __getitem__(self, item_idx):
        original_idx, label = self.indices[item_idx], self.labels[item_idx]
        
        # 1. 获取原始图像
        
        # --- (修改) ---
        if self.is_cub:
            # CUB200: .data 是 DataFrame. 必须用 .data_paths
            img_path = os.path.join(self.base_dataset.root, 
                                    self.base_dataset.base_folder, 
                                    'images', 
                                    self.base_dataset.data_paths[original_idx])
            img = Image.open(img_path).convert('RGB')

        elif self.is_crowd:
            # Crowdsource: .data 是 'list' of paths
            img_path = self.base_dataset.data[original_idx]
            img = Image.open(img_path).convert('RGB')

        else: 
            # CIFAR: .data 是 numpy 数组
            img = Image.fromarray(self.base_dataset.data[original_idx])
        # --- (修改结束) ---
            
        # 2. 应用 *这个类* 自己的 transform (即 weak_t 和 strong_t)
        return self.weak_t(img), self.strong_t(img), label, original_idx    
class HybridDataset(Dataset):
    def __init__(self, base_dataset, data_list, weak_t, strong_t): self.base_dataset, self.data_list, self.weak_t, self.strong_t = base_dataset, data_list, weak_t, strong_t
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        original_idx, label, is_reliable = self.data_list[idx]
        if isinstance(self.base_dataset, CUB200Partial): img, _, _ = self.base_dataset[original_idx]
        else: img = Image.fromarray(self.base_dataset.data[original_idx])
        return self.weak_t(img), self.strong_t(img), label, is_reliable, original_idx

# ==============================================================================
#                      Section 3: 核心算法辅助
# ==============================================================================
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

def get_topology_guided_affinity(raw_D, neighbors_indices, current_soft_labels, num_classes, device):
    """
    实现拓扑一致性引导的亲和矩阵 (修复维度版)
    Returns: [N, K+1] 维度的权重矩阵
    """
    
    # neighbors_indices: [N, K+1]
    N, K_plus_1 = neighbors_indices.shape
    
    # -----------------------------------------------------------
    # Step 1: 纯线性平滑 KNN 估计 (只用邻居预测我)
    # -----------------------------------------------------------
    # 我们只用 K 个邻居来预测当前样本，不包含自己，这样才客观
    # neighbors_indices[:, 1:]: [N, K]
    linear_weights = raw_D[:, 1:] 
    linear_weights = linear_weights / (linear_weights.sum(dim=1, keepdim=True) + 1e-12)
    
    neighbor_labels = F.embedding(neighbors_indices[:, 1:], current_soft_labels)
    # [N, K, 1] * [N, K, C] -> sum -> [N, C]
    knn_scores_smooth = (linear_weights.unsqueeze(-1) * neighbor_labels).sum(dim=1)
    p_knn = knn_scores_smooth / (knn_scores_smooth.sum(dim=1, keepdim=True) + 1e-12)
    
    # -----------------------------------------------------------
    # Step 2: 拓扑一致性检查 (计算熵)
    # -----------------------------------------------------------
    my_candidate_mask = current_soft_labels.clone()
    masked_scores = p_knn * my_candidate_mask
    masked_prob = masked_scores / (masked_scores.sum(dim=1, keepdim=True) + 1e-12)
    entropy = -torch.sum(masked_prob * torch.log(masked_prob + 1e-12), dim=1) # [N]
    
    # -----------------------------------------------------------
    # Step 3: 计算样本信誉度 (Reliability)
    # -----------------------------------------------------------
    max_entropy = np.log(num_classes)
    norm_entropy = entropy / max_entropy
    gamma = 2.0 
    reliability_scores = torch.exp(-gamma * (norm_entropy ** 2)) # [N]
    
    # -----------------------------------------------------------
    # Step 4: 生成最终亲和矩阵 (维度修复关键点)
    # -----------------------------------------------------------
    # 我们需要返回 [N, K+1] 的矩阵，对应 neighbors_indices 的形状
    
    # 1. 扩展 reliability_scores 维度以便查表: [N] -> [N, 1]
    reliability_scores_expanded = reliability_scores.unsqueeze(1)
    
    # 2. 查表获取所有 K+1 个节点（包含自身和邻居）的信誉度
    # F.embedding 输入 [N, K+1], 输出 [N, K+1, 1]
    # squeeze 后变成 [N, K+1]
    # 这里意味着：如果我是噪声(reliability低)，那么在矩阵中 W_ii (自我连接) 也会变小
    all_reliabilities = F.embedding(neighbors_indices, reliability_scores_expanded).squeeze(-1)
    
    # 3. 原始相似度 raw_D 本身就是 [N, K+1]
    # 亲和度 = 原始相似度 * 节点的信誉度
    refined_sim = raw_D * all_reliabilities
    
    # -----------------------------------------------------------
    # Step 5: 最终归一化
    # -----------------------------------------------------------
    final_weights = refined_sim / (refined_sim.sum(dim=1, keepdim=True) + 1e-12)
    
    return final_weights

@torch.no_grad()
def iterative_pseudo_labelling(features, partial_labels, k, iterations, device):
    # 1. KNN Search (PyTorch)
    features = features.to(device)
    # Norm
    features = F.normalize(features, p=2, dim=1)
    
    # Simple Chunked KNN (to avoid OOM on large sets)
    chunk_size = 4096
    N = features.shape[0]
    sims_list, idx_list = [], []
    
    with torch.no_grad():
        database_t = features.t()
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            # Mm
            sim_matrix = torch.mm(features[i:end_idx], database_t)
            # Top K+1 (including self)
            batch_sims, batch_indices = torch.topk(sim_matrix, k=min(k+1, N), dim=1, largest=True, sorted=True)
            sims_list.append(batch_sims)
            idx_list.append(batch_indices)
            
    sim = torch.cat(sims_list, dim=0)
    neighbor_idx = torch.cat(idx_list, dim=0)

    # 2. Skip Self (Column 0)
    sim = sim[:, 1:]
    neighbor_idx = neighbor_idx[:, 1:]
    
    # 3. Candidates
    current_candidates = partial_labels.to(device).float()

    for i in range(iterations):
        # neighbor_idx: [N, K]
        # current_candidates: [N, C]
        # We need to gather neighbors' candidates: [N, K, C]
        
        # Since F.embedding works with 2D indices:
        neighbor_candidates = F.embedding(neighbor_idx, current_candidates) # [N, K, C]
        
        # Weighted Voting
        # sim: [N, K] -> [N, K, 1]
        weighted_votes = torch.sum(sim.unsqueeze(-1) * neighbor_candidates, dim=1)
        
        pseudo_labels = torch.argmax(weighted_votes, dim=1)
        
        if i < iterations - 1:
            current_candidates = F.one_hot(pseudo_labels, partial_labels.shape[1]).float()
            
    return pseudo_labels, F.softmax(weighted_votes, dim=1)



class SoftMatchWeightManager:
    def __init__(self, num_samples, num_classes, n_sigma=2.0, momentum=0.99, device='cuda'): self.n_sigma, self.momentum, self.device = n_sigma, momentum, device; self.prob_model = torch.ones(num_samples, num_classes, device=device) / num_classes
    def __call__(self, preds, index, return_stats=False):
        self.prob_model[index] = self.momentum * self.prob_model[index] + (1 - self.momentum) * preds.detach(); max_probs_model = self.prob_model[index].max(dim=1)[0]; mu = max_probs_model.mean(); std = max_probs_model.std() if max_probs_model.size(0) > 1 else torch.tensor(1e-8, device=self.device); weights = torch.exp(-torch.pow(F.relu(mu - preds.max(dim=1)[0]), 2) / (2 * self.n_sigma * std**2 + 1e-8))
        return (weights.detach(), mu.item(), std.item()) if return_stats else weights.detach()




# (先添加 SoftMatch 的日志函数)
def log_softmatch_diagnostics(logger, diagnostics, num_hq_samples):
    if not diagnostics['weights']:
        logger.info("  -> [SoftMatch Diagnostics] No HQ-unsupervised samples trained.")
        return

    weights = torch.cat(diagnostics['weights'])
    confidences = torch.cat(diagnostics['confidences'])
    pseudo_labels = torch.cat(diagnostics['pseudo_labels'])
    true_labels = torch.cat(diagnostics['true_labels'])
    
    avg_mu = np.mean(diagnostics['mus'])
    avg_std = np.mean(diagnostics['stds'])
    
    logger.info(f"  -> [SoftMatch Diagnostics] --- Trained on {len(weights)}/{num_hq_samples} HQ samples ---")
    logger.info(f"    - Avg Weight: {weights.mean().item():.4f} (Min: {weights.min().item():.4f}, Max: {weights.max().item():.4f})")
    logger.info(f"    - Gaussian Center (μ): Avg {avg_mu:.4f} | Gaussian Width (σ): Avg {avg_std:.4f}")

    tiers = {"High (w>0.8)": (0.8, 1.1), "Mid (0.5-0.8)": (0.5, 0.8), "Low (w<0.5)": (0.0, 0.5)}
    for name, (lower, upper) in tiers.items():
        mask = (weights >= lower) & (weights < upper)
        num_in_tier = mask.sum().item()
        if num_in_tier == 0:
            continue
        
        pl_acc = (pseudo_labels[mask] == true_labels[mask]).float().mean().item() * 100
        avg_conf = confidences[mask].mean().item()
        avg_w = weights[mask].mean().item()
        logger.info(f"      - {name}: {num_in_tier:<5} samples | PL Acc: {pl_acc:.2f}% | Avg Conf: {avg_conf:.3f} | Avg Weight: {avg_w:.3f}")



# (This is the modified function in Section 4)
def train_unified_loop(args, encoder, classifier, device, 
                       loader_sup, loader_unsup, optimizer, softmatch_manager, 
                       logger, num_classes, knn_pl, model_pl,
                       knn_scores,  # <--- 🚀 添加这个参数
                       dynamic_consistency_weight,
                       rebalance_factor):
    
    ## --- 🚀 MODIFICATION: Use encoder --- ##
    encoder.train(); classifier.train(); scaler = GradScaler()
    
    knn_pl, model_pl = knn_pl.to(device), model_pl.to(device)
    knn_scores = knn_scores.to(device) # <--- 🚀 将其移动到 device

    # (修改) 拆分 SSL 损失以进行详细日志记录
    total_loss_s, total_loss_c_self, total_loss_c_cross, total_loss_consist = 0.0, 0.0, 0.0, 0.0
    num_sup, num_unsup = 0, 0

    loaders = [l for l in [loader_sup, loader_unsup] if l is not None]
    if not loaders:
        logger.warning("No data loaders available for training. Skipping epoch.")
        return
        
    # (确保两个 loader 都存在，否则 drop_last=True 会在迭代器上出错)
    if not loader_sup or not loader_unsup:
        # (Allow training with only one loader if the other is empty)
        if not loader_sup:
            logger.info("Missing reliable loader, training on unreliable set only.")
        if not loader_unsup:
            logger.info("Missing unreliable loader, training on reliable set only.")
    
    # (Handle cases where one loader might be missing)
    if loader_sup and loader_unsup:
        num_batches = max(len(loader_sup), len(loader_unsup))
        iter_sup = iter(itertools.cycle(loader_sup))
        iter_unsup = iter(itertools.cycle(loader_unsup))
    elif loader_sup:
        num_batches = len(loader_sup)
        iter_sup = iter(loader_sup)
        iter_unsup = None
    elif loader_unsup:
        num_batches = len(loader_unsup)
        iter_sup = None
        iter_unsup = iter(loader_unsup)
    else:
        logger.warning("No data loaders available for training. Skipping epoch.")
        return
    
    for i in range(num_batches):
        optimizer.zero_grad()
        
        all_weak_imgs, all_strong_imgs, all_labels = [], [], []
        
        # --- 1. 可靠集 (Supervised Loss) ---
        loss_s = torch.tensor(0.0, device=device)
        
        if iter_sup:
            # (假设 SSLReadyDataset 返回 4 个值: w, s, label, idx)
            sup_w, sup_s, sup_l, sup_idx = next(iter_sup) 
            sup_w, sup_s, sup_l = sup_w.to(device), sup_s.to(device), sup_l.to(device)
            all_weak_imgs.append(sup_w); all_strong_imgs.append(sup_s)
            all_labels.append(sup_l)
            
            s_labels = F.one_hot(sup_l, num_classes).float() * (1 - args.lsr) + args.lsr / num_classes
            B_s = sup_w.size(0) # 可靠集的 Batch Size
            
            # (L_S 的 Mixup，在可靠集内部进行)
            perm_w = torch.randperm(B_s)
            lam_w = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            mix_w = lam_w * sup_w + (1 - lam_w) * sup_w[perm_w, :]
            mix_l_w = lam_w * s_labels + (1 - lam_w) * s_labels[perm_w, :]

            perm_s = torch.randperm(B_s)
            lam_s = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            mix_s = lam_s * sup_s + (1 - lam_s) * sup_s[perm_s, :]
            mix_l_s = lam_s * s_labels + (1 - lam_s) * s_labels[perm_s, :]
            
            with autocast('cuda'):
                ## --- 🚀 MODIFICATION: Use encoder --- ##
                logits_w = classifier(encoder(mix_w))
                loss_w = -torch.sum(F.log_softmax(logits_w, 1) * mix_l_w, 1).mean()
                logits_s = classifier(encoder(mix_s))
                loss_s_strong = -torch.sum(F.log_softmax(logits_s, 1) * mix_l_s, 1).mean()
                loss_s = (loss_w + loss_s_strong) * 0.5            
            
            total_loss_s += loss_s.item() * B_s
            num_sup += B_s
        else:
            B_s = 0 # (No reliable samples this batch)

        # --- 2. 不可靠集 (SSL 损失) ---
        loss_c_self = torch.tensor(0.0, device=device)
        loss_c_cross = torch.tensor(0.0, device=device)

        if iter_unsup:
            # (SSLReadyDataset 返回 4 个值: w, s, label=-1, idx)
            unsup_w, unsup_s, _, unsup_idx = next(iter_unsup) 
            unsup_w, unsup_s, unsup_idx = unsup_w.to(device), unsup_s.to(device), unsup_idx.to(device)
            all_weak_imgs.append(unsup_w); all_strong_imgs.append(unsup_s)
            all_labels.append(knn_pl[unsup_idx]) # (用于 L_Consist)
            B_u = unsup_w.size(0) # 不可靠集的 Batch Size

            # 2.1) 获取去偏见目标 T_i 和 权重 final_w
            # 2.1) 获取预测并计算几何平均
            with torch.no_grad(), autocast('cuda'):
                p_w = F.softmax(classifier(encoder(unsup_w)), dim=1)
                # p_s = F.softmax(classifier(encoder(unsup_s)), dim=1) # 🚀 OPTIMIZATION: Removed redundant forward pass
                T_i_calibrated = (p_w * rebalance_factor)
                T_i_calibrated = T_i_calibrated / (T_i_calibrated.sum(dim=1, keepdim=True) + 1e-8)

                # 2. Get Hard Pseudo-Labels (Argmax)
                pseudo_labels = torch.argmax(T_i_calibrated, dim=1)

                # 3. Convert to One-Hot
                # Ensure num_classes is available in your scope
                target_one_hot = F.one_hot(pseudo_labels, num_classes).float()

                # 4. Apply Label Smoothing
                # Formula: y_ls = (1 - epsilon) * y_hard + epsilon / K
                target = target_one_hot * (1 - args.lsr) + args.lsr / num_classes
                ######################################
                # 5. IMPORTANT: Detach
                target = target.detach()
                
            
                # (SoftMatch 权重)
                softmatch_weights = softmatch_manager(p_w, unsup_idx)
                consensus_mask = (model_pl[unsup_idx] == knn_pl[unsup_idx])
                base_consistency_weights = torch.full_like(softmatch_weights, 1.0)
                base_consistency_weights[consensus_mask] = 1.0
                final_weights = base_consistency_weights * softmatch_weights # [B_u]

            # 2.2) 损失 A: Unreliable x Unreliable (动态 Mixup)
            with autocast('cuda'):
                perm = torch.randperm(B_u, device=device)
                lam = softmatch_weights # (使用 SoftMatch 权重作为 lambda)
                lam_img = lam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                lam_tgt = lam.unsqueeze(-1)
                # mix_unsup_w = lam_img * unsup_w + (1 - lam_img) * unsup_w[perm]
                mix_unsup_s = lam_img * unsup_s + (1 - lam_img) * unsup_s[perm]
                mix_target_self = lam_tgt * target + (1 - lam_tgt) * target[perm]
                
                ## --- 🚀 MODIFICATION: Use encoder --- ##
                # log_p_w_mix = F.log_softmax(classifier(encoder(mix_unsup_w)), dim=1)
                log_p_s_mix = F.log_softmax(classifier(encoder(mix_unsup_s)), dim=1)
                loss_c_per_sample_s = -torch.sum(mix_target_self.detach() * log_p_s_mix, dim=1)
                loss_c_self = (loss_c_per_sample_s * final_weights).mean()
                
            total_loss_c_self += loss_c_self.item() * B_u
            
            num_unsup += B_u
            
            # (合并两个 SSL 损失)
            loss_c = loss_c_self + loss_c_cross
        else:
            loss_c = torch.tensor(0.0, device=device) # (No unreliable samples)
        total_loss = (loss_s + 
                      dynamic_consistency_weight * loss_c)
        if total_loss > 0 and not torch.isnan(total_loss):
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # --- Epoch 结束后的日志 ---
    avg_loss_s = total_loss_s / num_sup if num_sup > 0 else 0
    # (修改) 拆分 SSL 日志
    avg_loss_c_self = total_loss_c_self / num_unsup if num_unsup > 0 else 0
    avg_loss_c_cross = total_loss_c_cross / num_unsup if num_unsup > 0 else 0
    avg_loss_consist = total_loss_consist / (num_sup + num_unsup) if (num_sup + num_unsup) > 0 else 0
    
    if num_batches > 0: 
        logger.info(f"  -> [Train Loss] Sup={avg_loss_s:.4f}, SSL-Self={avg_loss_c_self:.4f}, SSL-Cross={avg_loss_c_cross:.4f}, Reg={avg_loss_consist:.4f}.")





@torch.no_grad()
def evaluate(encoder, classifier, loader, device):
    encoder.eval(); classifier.eval(); correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast('cuda'): outputs = classifier(encoder(images))
        _, predicted = torch.max(outputs, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0
def compute_adaptive_weights(cosine_sim, neighbors, input_labels, mode, args, device):
    """
    通用相似度权重计算函数 (修正版: 严格区分线性与指数逻辑)
    """
    # =========================================================
    # Mode 1: Cosine (纯线性 Linear)
    # =========================================================
    if mode == 'cosine':
        # ❌ 原错误写法: weights = torch.softmax(cosine_sim, dim=1)
        # ✅ 修正写法: 直接使用余弦值，做线性归一化
        weights = cosine_sim
        
        # 线性归一化 (L1 Normalize): w_i / sum(w)
        # 这样保留了原始的“距离感”，而不是强制拉开差距
        # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights

    # =========================================================
    # Mode 2: Exp (固定锐化 Sharpening)
    # =========================================================
    elif mode == 'exp':
        # 还原 PALS 原文逻辑: weights ~ exp(sim / 0.1)
        # 虽然 softmax(sim/0.1) 数学上等价于 normalized exp，但显式写出来更清晰
        
        scaled_sim = cosine_sim / 0.1
        # 为了数值稳定性，通常减去最大值再 exp (Softmax 的内部实现)
        # 这里直接用 exp，如果数值过大可能会溢出，但 cosine 最大是 1，/0.1=10，exp(10) ~ 22026，不会溢出
        weights = torch.exp(scaled_sim)
        
        # 线性归一化
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights

    # =========================================================
    # Mode 3: TDAS (基于熵/拓扑的自适应温度)
    # =========================================================
    elif mode == 'tdas':
        P = input_labels.float().to(device)
        context_neighbors = neighbors[:, 1:] # [N, K]
        neighbor_candidates = F.embedding(context_neighbors, P)
        local_mean_vector = neighbor_candidates.mean(dim=1)
        
        # 计算局部熵
        p_local = local_mean_vector / (local_mean_vector.sum(dim=1, keepdim=True) + 1e-8)
        local_entropy = -torch.sum(p_local * torch.log(p_local + 1e-8), dim=1, keepdim=True)
        
        max_possible_entropy = np.log(args.num_classes)
        norm_entropy = local_entropy / max_possible_entropy
        
        # 自适应温度
        # 熵高(混淆) -> 温度低(锐化) -> 逼近 Exp 模式
        # 熵低(干净) -> 温度高(平滑) -> 逼近 Linear 模式
        base_tau = 0.5
        gamma = 3.0 
        
        adaptive_tau = base_tau * torch.exp(-gamma * norm_entropy)
        adaptive_tau = torch.clamp(adaptive_tau, min=0.01)
        adaptive_tau = adaptive_tau.to(cosine_sim.device)

        # 应用温度 + Softmax (这里必须用 Softmax/Exp 来实现温度缩放的效果)
        scaled_sim = cosine_sim / adaptive_tau
        softmax_weights = torch.softmax(scaled_sim, dim=1)
        
        # Jaccard 惩罚 (线性乘法)
        P_self = P 
        P_neighbors = F.embedding(neighbors, P)
        intersection = (P_self.unsqueeze(1) * P_neighbors).sum(dim=2)
        union = P_self.sum(dim=1).unsqueeze(1) + P_neighbors.sum(dim=2) - intersection
        jaccard_sim = intersection / (union + 1e-6)
        
        beta = 0.5
        penalty_factor = 1.0 - beta * jaccard_sim
        penalty_factor = penalty_factor.to(cosine_sim.device)
        
        final_weights = softmax_weights * penalty_factor
        weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights
    
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")

import torch
import torch.nn.functional as F
import numpy as np


def reliable_pseudolabel_selection_weighted(logger, args, device, trainloader, features, epoch, model_preds=None):
    
    # ==============================================================================
    # 🛠️ Helper 1: 显存安全的高精度 KNN (PyTorch Native)
    # ==============================================================================
    def knn_search_pytorch_chunked(feats, k, num_heads=1, chunk_size=4096):
        original_matmul_precision = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False 
            N, D_dim = feats.shape
            
            # 1. 确保特征在 GPU 上
            feats = feats.to(device)
            if not feats.is_contiguous(): feats = feats.contiguous()
            
            if num_heads > 1:
                if D_dim % num_heads != 0:
                    if logger: logger.warning(f"Dim {D_dim} not divisible by {num_heads}, using 1 head.")
                    num_heads = 1
                head_dim = D_dim // num_heads
                # [N, H, D/H] -> Normalize each head -> Flatten back to [N, D]
                # Dot product of these "concat-normalized" vectors == Sum of cosine sims of heads
                feats_ready = F.normalize(feats.view(N, num_heads, head_dim), p=2, dim=2).reshape(N, D_dim) 
            else:
                feats_ready = F.normalize(feats, p=2, dim=1)

            final_sims, final_indices = [], []
            if logger: logger.info(f"   >>> [Selection] Running PyTorch Chunked KNN (Heads={num_heads}, Chunk={chunk_size})...")

            with torch.no_grad():
                database_t = feats_ready.t()
                for i in range(0, N, chunk_size):
                    end_idx = min(i + chunk_size, N)
                    
                    # Exact matrix multiplication
                    sim_matrix = torch.mm(feats_ready[i:end_idx], database_t)
                    
                    # Top-K
                    batch_sims, batch_indices = torch.topk(sim_matrix, k=min(k+1, N), dim=1, largest=True, sorted=True)
                    final_sims.append(batch_sims); final_indices.append(batch_indices)
            
            # Return Double Precision for Compatibility
            merged_sims = torch.cat(final_sims, dim=0).double() / float(num_heads)
            merged_indices = torch.cat(final_indices, dim=0).long()
            return merged_sims, merged_indices
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_precision

    # ==============================================================================
    # 🛠️ Helper 2: 权重计算 (FP64)
    # ==============================================================================
    def get_adaptive_affinity_matrix(raw_D, neighbors_indices, current_soft_labels):
        raw_D = raw_D.double()
        current_soft_labels = current_soft_labels.double()
        
        if raw_D.shape[1] > neighbors_indices.shape[1]:
            raw_D_neighbors = raw_D[:, 1:]
        else:
            raw_D_neighbors = raw_D

        neighbor_labels = F.embedding(neighbors_indices, current_soft_labels)
        
        att_temp = 0.5 
        spatial_weights = F.softmax(raw_D_neighbors / att_temp, dim=1).unsqueeze(-1)
        local_mean = (neighbor_labels * spatial_weights).sum(dim=1)
        
        local_mean = local_mean / (local_mean.sum(dim=1, keepdim=True) + 1e-12)
        local_entropy = -torch.sum(local_mean * torch.log(local_mean + 1e-12), dim=1)
        
        max_entropy = np.log(args.num_classes)
        norm_entropy = local_entropy / max_entropy
        
        base_tau = 0.07
        tau_dynamic = base_tau + (torch.pow(norm_entropy, 2) * 0.5) 
        tau_dynamic = tau_dynamic.unsqueeze(1) 
        
        scaled_sim = raw_D / tau_dynamic
        max_val, _ = scaled_sim.max(dim=1, keepdim=True)
        weights = torch.exp(scaled_sim - max_val.detach())
        return weights

    def get_affinity_matrix(mode_name, raw_D):
        raw_D = raw_D.double()
        if mode_name == 'exp':
            return torch.exp(raw_D / 0.1)
        elif mode_name == 'D' or mode_name == 'linear':
            return raw_D
        elif mode_name == 'rank':
            K_curr = raw_D.shape[1]
            ranks = torch.arange(K_curr, device=device).unsqueeze(0).expand(raw_D.shape[0], K_curr).double()
            sigma = K_curr / 4.0 
            return torch.exp(-ranks / sigma)
        else:
            return raw_D

    def get_dynamic_weight_matrix(mode, raw_D, neighbors_idx, ref_soft):
        if mode == 'daes':
            return get_adaptive_affinity_matrix(raw_D, neighbors_idx[:, 1:], ref_soft)
        
        # --- 新增的部分 ---
        elif mode == 'topology_entropy':
             # 确保传入正确的 num_classes
            return get_topology_guided_affinity(raw_D, neighbors_idx, ref_soft, args.num_classes, device)
        # ----------------
        
        else:
            raw_w = get_affinity_matrix(mode, raw_D)
            return raw_w / (raw_w.sum(dim=1, keepdim=True) + 1e-12)
    # ==============================================================================
    # 🛠️ Helper 3: 单步传播 (FP64)
    # ==============================================================================
    def _run_step(input_soft, weight_tensor, target_neighbors, is_iter1, tag_name):
        input_soft = input_soft.double()
        weight_tensor = weight_tensor.double()
        
        knn_idx = target_neighbors.view(N, args.k_val+1, 1).expand(N, args.k_val+1, args.num_classes)
        knn_input = input_soft.expand(N, -1, -1)
        
        score = torch.sum(torch.mul(torch.gather(knn_input, 1, knn_idx), weight_tensor.view(N, -1, 1),), 1)

        if is_iter1:
            _pl = torch.max(score, -1)[1]
            output_soft = torch.zeros((len(_pl), args.num_classes), device=device, dtype=torch.float64).scatter_(1, _pl.view(-1,1), 1.0)
        else:
            output_soft = score / (score.sum(1).unsqueeze(-1) + 1e-12)
            _pl = torch.max(score, -1)[1]
        
        return output_soft, _pl

    # ==============================================================================
    # 🚀 PART 1: 准备阶段 (FAISS Multi-Head Search)
    # ==============================================================================
    # 使用 Helper 1 进行搜索
    # 使用 Helper 1 进行搜索
    D_mh, neighbors_mh = knn_search_pytorch_chunked(features, args.k_val, num_heads=4)
    
    N = features.shape[0]

    # 2. 标签准备 (FP64)
    labels = torch.tensor(trainloader.dataset.soft_labels, device=device, dtype=torch.float64)
    clean_labels = torch.tensor(trainloader.dataset.clean_labels, device=device, dtype=torch.long)
    soft_labels = labels.clone()
    
    if hasattr(trainloader.dataset, 'weights'):
        prior = torch.tensor(trainloader.dataset.weights, device=device, dtype=torch.float64)
    else:
        prior = torch.ones_like(soft_labels)

    # 3. 模型预测修正
    if epoch > args.start_correct:
        prob, pred = torch.max(model_preds, 1)
        conf_th = args.conf_th_h - (args.conf_th_h - args.conf_th_l) * ((epoch - args.start_correct)/(args.epoch - args.start_correct))
        
        mask = prob > conf_th
        if mask.sum() > 0:
            soft_labels[mask] = 0.0
            soft_labels[mask, pred[mask]] = 1.0
            labels[mask] = 0.0
            labels[mask, pred[mask]] = 1.0

    initial_input = (soft_labels * prior)

    # ==============================================================================
    # 🚀 PART 2: 主执行流程 (动态模式 + FP64)
    # ==============================================================================
    if logger: logger.info(f"   >>> [Main Execution] {args.sim_mode_1} -> {args.sim_mode_2}")

    # Iter 1
    w_iter1 = get_dynamic_weight_matrix(args.sim_mode_1, D_mh, neighbors_mh, initial_input)
    soft_labels_iter1, pseudo_labels_iter1 = _run_step(initial_input, w_iter1, neighbors_mh, True, f"Main-{args.sim_mode_1}")

    # Iter 2
    w_iter2 = get_dynamic_weight_matrix(args.sim_mode_2, D_mh, neighbors_mh, soft_labels_iter1)
    soft_labels_iter2, pseudo_labels_iter2 = _run_step(soft_labels_iter1, w_iter2, neighbors_mh, False, f"Main-{args.sim_mode_2}")

    # ==============================================================================
    # 🚀 PART 3: 最终筛选 (GPU + FP64)
    # ==============================================================================
    final_soft = soft_labels_iter2 
    final_pl = pseudo_labels_iter2
    
    prob_temp = torch.clamp(final_soft, min=1e-6, max=1-1e-6)
    discrepancy_measure = -torch.log(prob_temp)

    max_idx = final_soft.max(dim=1)[1]
    agreement_bool = (labels.gather(1, max_idx.unsqueeze(1)).squeeze(1) == 1.0)
    
    valid_indices = max_idx[agreement_bool]
    if valid_indices.numel() > 0:
        agreement_counts = torch.bincount(valid_indices, minlength=args.num_classes).double()
    else:
        agreement_counts = torch.zeros(args.num_classes, device=device, dtype=torch.float64)

    if args.delta == 0.5: limit_per_class = torch.median(agreement_counts)
    elif args.delta == 1.0: limit_per_class = torch.max(agreement_counts)
    elif args.delta == 0.0: limit_per_class = torch.min(agreement_counts)
    else: limit_per_class = torch.quantile(agreement_counts, args.delta)

    final_selection_mask = torch.zeros(N, device=device, dtype=torch.float64)
    selected_examples_labels = torch.full((N, args.num_classes), float('inf'), device=device, dtype=torch.float64)

    for i in range(args.num_classes):
        idx_class = (labels[:, i] == 1.0)
        samples_per_class = idx_class.sum()
        
        if samples_per_class == 0: continue
            
        discrepancy_class = discrepancy_measure[idx_class, i]
        k_corrected = min(limit_per_class, samples_per_class).long()
        if k_corrected < 1: continue
            
        val, top_rel_idx = torch.topk(discrepancy_class, k=int(k_corrected.item()), largest=False, sorted=False)
        
        full_indices = idx_class.nonzero().squeeze(1)
        selected_global_indices = full_indices[top_rel_idx]
        
        final_selection_mask[selected_global_indices] = 1.0
        selected_examples_labels[selected_global_indices, i] = val

    # ==============================================================================
    # 5. 返回结果
    # ==============================================================================
    _, selected_labels_val = torch.min(selected_examples_labels, 1)
    
    if logger:
        n_sel = final_selection_mask.sum().item()
        acc_check = (final_pl[final_selection_mask.bool()] == clean_labels[final_selection_mask.bool()]).float().mean().item()
        logger.info(f"   >>> [Selection] Selected: {int(n_sel)} | Method: MH-FAISS (CPU) + FP64 Calc | Rel Acc: {acc_check*100:.2f}%")

    return (
        final_selection_mask.float(), 
        selected_labels_val, 
        final_pl, 
        torch.max(model_preds, 1)[1], 
        final_soft.float()
    )





def run_single_experiment(args):
    start_time = time.time(); set_seed(args.seed)
    log_dir = os.path.join(args.out, args.exp_name, f"seed_{args.seed}")
    logger = setup_logger(log_dir)
    # logger = setup_logger(log_dir, to_console=True)
    logger.info(f"--- Starting Dynamic Strategy Run with Seed: {args.seed} ---"); logger.info(f"Settings: {vars(args)}")
    
    # --- (新增) W&B 初始化 (来自 PALS_v1) ---
    # wandb.init(project='Hybrid-SSL-PALS', config=args, name=f"{args.exp_name}_seed{args.seed}", mode="online") # 设为 "online" 或 "disabled"
    wandb.init(project='Hybrid-SSL-PALS', config=args, name=f"{args.exp_name}_seed{args.seed}", mode="disabled") # 设为 "online" 或 "disabled"

    device = torch.device(f"cuda:{args.cuda_dev}" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 数据加载 ---
    
    # --- (修改) 更新 num_classes_map ---
    num_classes_map = {
        'CIFAR10': 10, 'CIFAR100': 100, 'CIFAR100H': 100, 'CUB200': 200,
        'Treeversity': 6,  # (来自 PALS_v1 命令)
        'Benthic': 8,      # (来自 PALS_v1 命令)
        'Plankton': 10     # (来自 PALS_v1 命令)
    }
    if args.dataset not in num_classes_map:
        raise ValueError(f"Dataset {args.dataset} not recognized in num_classes_map.")
    num_classes = num_classes_map[args.dataset]
    
    args.num_classes = num_classes
    args.seed_dataset = args.seed
    
    weak_t, strong_t, test_t = get_pals_transforms(args.dataset)
    crowdsource_datasets = [
        'Treeversity', 'Benthic', 'Plankton', 
    ]  
    # --- (重大修改) 替换数据加载逻辑 ---
    
    # 1. 模拟数据集 (CIFAR / CUB)
    if args.dataset in ['CIFAR10', 'CIFAR100', 'CIFAR100H']:
        is_h = 'H' in args.dataset
        BaseClass = CIFAR100Partial if '100' in args.dataset else CIFAR10Partial
        base_train_ds = BaseClass(args, train=True, download=True, transform=None)
        
        if hasattr(base_train_ds, 'partial_noise'):
            logger.info(f"Generating simulated NPLL noise for {args.dataset} (pr={args.pr}, nr={args.nr})")
            if '100' in args.dataset:
                base_train_ds.partial_noise(args.pr, args.nr, heirarchical=is_h)
            else:
                base_train_ds.partial_noise(args.pr, args.nr)
                
        TestClass = datasets.CIFAR100 if '100' in args.dataset else datasets.CIFAR10
        test_ds = TestClass(root=args.train_root, train=False, download=True, transform=test_t)

    elif args.dataset == 'CUB200':
        BaseClass = CUB200Partial
        base_train_ds = BaseClass(args, train=True, transform=None)
        if hasattr(base_train_ds, 'partial_noise'):
             logger.info(f"Generating simulated NPLL noise for CUB200 (pr={args.pr}, nr={args.nr})")
             base_train_ds.partial_noise(args.pr, args.nr)
        test_ds = BaseClass(args, train=False, transform=test_t)
        
# ... (CUB200 的加载逻辑不变) ...
  
    # 2. 真实众包数据集 (使用 Crowdsource 类)
    # elif args.dataset in ['Treeversity', 'Benthic', 'Plankton','Pig']:
    elif args.dataset in crowdsource_datasets:
        logger.info(f"Loading crowdsource dataset: {args.dataset} (lpi={args.lpi})")
        
        # --- (已修复) ---
        # 根据截图，所有三个数据集都有 5 个 fold。
        # 我们统一使用 PALS_v1 中 slice=1 的默认切分方式。
        # train_split, test_split = ['fold1','fold4','fold5'], ['fold3']
        train_split, test_split = ['fold1','fold2','fold4','fold5'], ['fold3']
        logger.info(f"Using splits: Train={train_split}, Test={test_split}")
        # --- (修复结束) ---
        
        # (重要) 确保 train_root 指向特定数据集的文件夹, e.g., './data/Treeversity#6'
        dataset_root = os.path.join(args.train_root, args.dataset)
        if args.dataset == 'Treeversity':
             dataset_root = os.path.join(args.train_root, 'Treeversity#6') # (PALS_v1 的特殊路径)
        
        logger.info(f"Loading from root: {dataset_root}")
        
        # 复制一份 args, 仅用于 Crowdsource 类
        crowd_args = copy.deepcopy(args)
        crowd_args.train_root = dataset_root
        
        base_train_ds = Crowdsource(
            crowd_args,
            splits=train_split,
            transform=None # Transform 在 SSLReadyDataset 中应用
        )
        test_ds = Crowdsource(
            crowd_args,
            splits=test_split,
            transform=test_t
        )
    else:
        raise ValueError(f"Dataset {args.dataset} loader not implemented.")
    # --- (修改结束) ---

    # (从这里开始，你的代码逻辑保持不变，它会自动处理)
    original_pl = torch.from_numpy(base_train_ds.soft_labels)
    true_labels = torch.from_numpy(base_train_ds.clean_labels)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # (在 Section 5: Main Experiment Logic)
# ...
# --- 2. 模型、优化器和调度器设置 ---
    
    ## --- 🚀 MODIFICATION START: Removed SimSiam, use encoder directly --- ##
    encoder, feature_dim = get_base_encoder(args.network, args.dataset)
    encoder = encoder.to(device) # No more simsiam_model
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    # --- (已更正) 仅对众包数据集使用分层学习率 ---
    # if args.dataset in ['Treeversity', 'Benthic', 'Plankton']:
    if args.dataset in crowdsource_datasets:
        logger.info(f"Using differential LR for crowdsource dataset {args.dataset} (Encoder LR / 100)")
        # (来自 PALS_v1 的设置，encoder LR x 0.01)
        optimizer = optim.SGD([
            {'params': encoder.parameters(), 'lr': args.lr / 100.0}, 
            # {'params': encoder.parameters(), 'lr': args.lr}, 
            {'params': classifier.parameters(), 'lr': args.lr},
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        # (CIFAR 和 CUB-200 均使用标准优化器)
        logger.info(f"Using standard optimizer settings for {args.dataset}")
        params_to_optimize = list(encoder.parameters()) + list(classifier.parameters())
        optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    ## --- 🚀 MODIFICATION END --- ##
    
    # --- (调度器逻辑不变) ---
    if args.lr_scheduler == 'cosine':
        t_max_for_cosine = args.epochs - args.lr_warmup_epochs if args.lr_warmup_epochs > 0 else args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_for_cosine)
        logger.info(f"Using CosineAnnealingLR scheduler with T_max = {t_max_for_cosine} epochs.")
    
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
        logger.info(f"Using MultiStepLR scheduler with milestones at {args.lr_decay_epochs} and gamma={args.lr_decay_rate}.")    
        
    softmatch_manager = SoftMatchWeightManager(len(base_train_ds), num_classes, device=device)

    best_test_acc = 0.0
    final_epoch_test_acc = 0.0

    ema_model_probs = torch.ones(len(base_train_ds), num_classes, device=device) / num_classes
    ema_alpha = 0.999 
# --- 🚀 混合方案：根据数据集类型估计目标分布 ---
    if hasattr(base_train_ds, 'weights'):
        # 适用于 Crowdsource: 使用数据驱动的估计
        logger.info(" -> [Re-balance] 正在从 'weights' (众包先验) 估计目标分布...")
        estimated_dist_np = base_train_ds.weights.mean(axis=0)
        estimated_dist_np = estimated_dist_np / estimated_dist_np.sum()
        p_target_estimated = torch.tensor(estimated_dist_np, device=device, dtype=torch.float).unsqueeze(0)
        logger.info(f" -> [Re-balance] 使用“估计的”目标分布: \n{estimated_dist_np}")
    else:
        # 适用于 CUB/CIFAR: 退回到“均匀分布”假设
        logger.info(" -> [Re-balance] 'weights' 属性未找到。退回到“均匀分布”假设。")
        p_target_estimated = (torch.ones(1, num_classes, device=device) / num_classes)
        logger.info(f" -> [Re-balance] 使用“均匀的”目标分布。")
    # --- 3. 主训练循环 ---
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f"======== Epoch {epoch+1}/{args.epochs} ========")
        
        # --- (已修复) 3. Warmup 逻辑 ---
        if epoch < args.lr_warmup_epochs:
            # (标准的线性 Warmup 逻辑)
            # 计算 warmup 比例 (从 1/N 增长到 N/N)
            lr_scale = (epoch + 1) / args.lr_warmup_epochs
            
            # (遍历所有参数组，包括你设置了分层学习率的组)
            for param_group in optimizer.param_groups:
                # param_group['initial_lr'] 是 PyTorch 优化器自动保存的初始学习率
                param_group['lr'] = param_group['initial_lr'] * lr_scale
            
            # (打印日志，确认 LR 正在变化)
            current_lr = optimizer.param_groups[-1]['lr'] # (通常看最后一个组，即分类器)
            logger.info(f" -> Warmup Epoch {epoch+1}/{args.lr_warmup_epochs}, LR set to {current_lr:.6f}")
            
        elif epoch == args.lr_warmup_epochs:
             logger.info(f" -> Warmup finished. LR reset to initial values.")
             # (在 warmup 结束后，确保 LR 恢复到初始值，以防 scheduler 不更新)
             for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr']


        
        ## --- 🚀 MODIFICATION START: Reverted to original selection logic --- ##
        # --- 3.1 数据划分 ---
        
        ## --- 🚀 MODIFICATION START: Call Selection Function --- ##
        logger.info(" -> [Selection] Getting features for all train data...")
        feature_loader = DataLoader(FeatureExtractionDataset(base_train_ds, test_t), batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
        features, predictions = get_features(encoder, classifier, feature_loader, device)

        # 创建一个模拟的 loader，仅用于传递 dataset 对象
        class MockTrainloader:
            def __init__(self, dataset):
                self.dataset = dataset
        mock_trainloader = MockTrainloader(base_train_ds)
        
        logger.info(f" -> [Selection] Running reliable_pseudolabel_selection_weighted (k={args.k_val}, delta={args.delta})...")
        
        # 调用 *修改后* 的函数，获取所有必需的组件
        selected_examples_mask, selected_labels_all, knn_pl, model_pl, knn_scores = reliable_pseudolabel_selection_weighted(
            logger,
            args, 
            device, 
            mock_trainloader, 
            features, 
            epoch + 1,        # 传递当前 epoch (用于日志)
            model_preds=predictions
        )
        
        # --- [Diagnostic Info] ---
        num_selected = selected_examples_mask.sum().item()
        logger.info(f" -> [Selection] Function call complete. Total samples selected: {num_selected}")

        # 将掩码 (mask) 转换为脚本所需的格式 (list of pairs)
        # 优化: 在 GPU 上进行索引操作，最后统一转 CPU
        rel_indices_gpu = torch.where(selected_examples_mask.bool())[0]
        
        # 'selected_labels_all' 包含 *所有* 样本基于差异度量的标签
        rel_labels_gpu = selected_labels_all[rel_indices_gpu]
        
        rel_indices = rel_indices_gpu.cpu().tolist()
        rel_labels = rel_labels_gpu.cpu().tolist()
        # reliable_pairs = list(zip(rel_indices, rel_labels)) # (这个变量现在已定义，但未使用)
        
        if num_selected > 0:
            rel_true_labels = true_labels.to(device)[rel_indices_gpu] # 使用张量索引 (Ensure true_labels on device)
            selection_acc = (rel_labels_gpu == rel_true_labels).float().mean().item()
            logger.info(f" -> [Selection] Reliable Set Accuracy (vs True Labels): {selection_acc * 100:.2f}%")
            wandb.log({'Reliable Set Accuracy': selection_acc * 100,
                       'Reliable Set Size': num_selected}, step=epoch+1) # (使用 epoch+1 避免 wandb 警告)
        else:
             logger.info(" -> [Selection] No reliable samples selected.")
             wandb.log({'Reliable Set Accuracy': 0,
                       'Reliable Set Size': 0}, step=epoch+1) # (使用 epoch+1)
        ## --- 🚀 MODIFICATION END --- ##


        # 这部分代码现在可以正常工作了
        rel_indices_set = set(rel_indices)

        num_unreliable = len(base_train_ds) - len(rel_indices_set)
        unreliable_ratio = num_unreliable / len(base_train_ds)
        unreliable_indices = list(set(range(len(base_train_ds))) - rel_indices_set)
        logger.info(f" -> Data Split: Reliable({len(rel_indices_set)}), Unreliable({len(unreliable_indices)})")
        
        # 这个日志函数现在也可以工作了
        if args.detailed_log:
            log_unified_partition_diagnostics(logger, true_labels, device, 
                                              rel_indices_set, unreliable_indices, 
                                              model_pl, knn_pl, predictions)
        
        # --- 3.2 动态一致性权重 ---
        reliable_set_len = len(rel_indices_set)
        consensus_proportion = 0.0 
        
        if reliable_set_len > 0:
            rel_indices_tensor = torch.tensor(list(rel_indices_set), dtype=torch.long, device=device)
            # (将 PL 移动到 device 进行比较)
            model_pred_rel = model_pl.to(device)[rel_indices_tensor]
            knn_pred_rel = knn_pl.to(device)[rel_indices_tensor]
            consensus_count_rel = (model_pred_rel == knn_pred_rel).sum().item()
            consensus_proportion = consensus_count_rel / reliable_set_len
        
        consensus_power = 2.0 
        # consensus_power = 1.0 #/consensus_power1
        dynamic_consistency_weight = unreliable_ratio * (consensus_proportion ** consensus_power) * args.consistency_weight
        logger.info(f" -> Dynamic Weight: UnreliableRatio={unreliable_ratio:.2f} * ReliableConsensus={consensus_proportion:.2f} (power {consensus_power}) -> Consistency Weight={dynamic_consistency_weight:.3f}")

        # --- 3.3 全局再平衡因子 ---
        # (保持不变)
        logger.info(" -> [Re-balance] Calculating distribution rebalance factor...")
        p_model_biased = ema_model_probs.mean(dim=0, keepdim=True)
        # p_target = (torch.ones(1, num_classes, device=device) / num_classes)
        p_target = p_target_estimated
        rebalance_factor = (p_target / (p_model_biased + 1e-8)).detach()
        
        if (epoch + 1) % 10 == 0:
             max_bias, max_class = p_model_biased.max(dim=1)
             min_bias, min_class = p_model_biased.min(dim=1)
             logger.info(f"     [Re-balance Stats] Model Bias (EMA): Max={max_bias.item():.4f} (Class {max_class.item()}), Min={min_bias.item():.4f} (Class {min_class.item()})")
             logger.info(f"     [Re-balance Stats] Rebalance Factor: Max={rebalance_factor.max().item():.2f}, Min={rebalance_factor.min().item():.2f}")
        
        
        # --- 3.4 创建 DataLoaders ---
        loader_sup, loader_unsup = None, None
            
        if rel_indices:
            ds_sup = SSLReadyDataset(base_train_ds, list(rel_indices), list(rel_labels), weak_t, strong_t)
            loader_sup = DataLoader(ds_sup, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
        if unreliable_indices:
            ds_unsup = SSLReadyDataset(base_train_ds, unreliable_indices, [-1] * len(unreliable_indices), weak_t, strong_t)
            # loader_unsup = DataLoader(ds_unsup, batch_size=args.unreliable_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            loader_unsup = DataLoader(ds_unsup, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        # --- 3.5 调用训练循环 ---
        # (现在我们传递 knn_scores)
        train_unified_loop(args, encoder, classifier, device, 
                           loader_sup, loader_unsup, optimizer, softmatch_manager, 
                           logger, num_classes, 
                           knn_pl, model_pl,
                           knn_scores, # <--- 🚀 传递新参数
                           dynamic_consistency_weight,
                           rebalance_factor
                           )

        if epoch >= args.lr_warmup_epochs and scheduler is not None:
             scheduler.step()
        
        ## --- 🚀 MODIFICATION: Use encoder --- ##
        test_acc = evaluate(encoder, classifier, test_loader, device)
        
        final_epoch_test_acc = test_acc 
        if test_acc > best_test_acc: 
            best_test_acc = test_acc
            
        ema_model_probs = ema_alpha * ema_model_probs + (1 - ema_alpha) * predictions
        
        logger.info(f"Epoch {epoch+1} Summary: Test Acc={test_acc:.2f}% | Best Acc={best_test_acc:.2f}% | LR={optimizer.param_groups[0]['lr']:.6f} | Time: {time.time() - epoch_start_time:.2f}s\n")
        
        # --- (新增) W&B 日志 (来自 PALS_v1) ---
        wandb.log({
            # 'Test Loss': 0, # evaluate 函数不返回 loss
            'Test Accuracy': test_acc,
            'Best Test Accuracy': best_test_acc,
            'Learning Rate': optimizer.param_groups[0]['lr']
        }, step=epoch + 1)

    # ... (循环结束) ...
    
    duration = time.time() - start_time
    logger.info(f"--- Run Finished (Duration: {duration/60:.2f} min). Best Acc: {best_test_acc:.2f}%, Final Epoch Acc: {final_epoch_test_acc:.2f}% ---")
    
    # --- (新增) W&B 结束 (来自 PALS_v1) ---
    wandb.finish()
    
    return best_test_acc, final_epoch_test_acc, duration


@torch.no_grad()
def log_unified_partition_diagnostics(logger, true_labels, device, 
                                      rel_indices_set, unreliable_indices, 
                                      model_pl, knn_pl, predictions):
    """
    为统一化的两阶段数据划分（可靠集 vs 不可靠集）设计的诊断函数。
    不可靠集内部会进一步分析“共识”与“非共识”样本的情况。
    """
    logger.info("--- Epoch Partition Diagnostics (Unified) ---")
    true_labels = true_labels.to(device)
    
    # --- 🚀 修复：将所有传入的伪标签张量移动到 GPU (device) ---
    model_pl, knn_pl, predictions = model_pl.to(device), knn_pl.to(device), predictions.to(device)
    # --- 修复结束 ---

    # 1. 可靠集诊断 (与之前相同)
    if rel_indices_set:
        rel_indices = torch.tensor(list(rel_indices_set), device=device, dtype=torch.long) # 确保 long
        # 可靠集的标签来自KNN提纯后的结果，我们用它和真实标签比较
        rel_acc = (knn_pl[rel_indices] == true_labels[rel_indices]).float().mean().item() * 100
        logger.info(f"  -> [Reliable Set]        Size: {len(rel_indices_set):<5} | Accuracy (vs KNN PL): {rel_acc:.2f}%")

    # 2. 不可靠集诊断 (新的核心逻辑)
    if unreliable_indices:
        unrel_indices_tensor = torch.tensor(unreliable_indices, device=device, dtype=torch.long) # 确保 long
        
        # 在不可靠集内部，动态划分出共识集与非共识集
        consensus_mask = (model_pl[unrel_indices_tensor] == knn_pl[unrel_indices_tensor])
        consensus_indices = unrel_indices_tensor[consensus_mask]
        non_consensus_indices = unrel_indices_tensor[~consensus_mask]

        # 2a. 共识集 (Consensus Set) 诊断
        if len(consensus_indices) > 0:
            con_pl_acc = (knn_pl[consensus_indices] == true_labels[consensus_indices]).float().mean().item() * 100
            con_model_conf = predictions[consensus_indices].max(dim=1)[0].mean().item()
            logger.info(f"  -> [Unreliable-Consensus]  Size: {len(consensus_indices):<5} | PL Acc (KNN): {con_pl_acc:.2f}% | Avg Model Conf: {con_model_conf:.3f}")

        # 2b. 非共识集 (Non-Consensus Set) 诊断
        if len(non_consensus_indices) > 0:
            non_con_pl_acc_knn = (knn_pl[non_consensus_indices] == true_labels[non_consensus_indices]).float().mean().item() * 100
            non_con_pl_acc_model = (model_pl[non_consensus_indices] == true_labels[non_consensus_indices]).float().mean().item() * 100
            logger.info(f"  -> [Unreliable-NonConsens] Size: {len(non_consensus_indices):<5} | PL Acc (KNN): {non_con_pl_acc_knn:.2f}% | PL Acc (Model): {non_con_pl_acc_model:.2f}%")

    # 3. 全局意见不合诊断 (与之前相同，依然很有用)
    disagreement_mask = (model_pl != knn_pl)
    num_disagree = disagreement_mask.sum().item()
    if num_disagree > 0:
        indices = torch.where(disagreement_mask)[0]
        model_acc_on_disagree = (model_pl[indices] == true_labels[indices]).float().mean().item() * 100
        knn_acc_on_disagree = (knn_pl[indices] == true_labels[indices]).float().mean().item() * 100
        logger.info(f"  -> [Global Disagreement]   Samples: {num_disagree:<5} ({num_disagree/len(model_pl):.2%}) | Model Acc: {model_acc_on_disagree:.2f}% | KNN Acc: {knn_acc_on_disagree:.2f}%")

# ==============================================================================
#                      MAIN (MODIFIED FOR DUAL STATS)
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    
    # (确保 wandb 已登录)
    # try:
    #     wandb.login()
    # except:
    #     print("Wandb login failed. Set wandb mode to 'disabled'.")
    #     wandb.init(mode="disabled")
    # wandb.init(mode="disabled")
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




"""
python trept_unsup_mixup_crowd_pals.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Benthic/lpi3_delta0.75 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.75 \
    --network R50 
python trept_unsup_mixup_crowd_pals.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Benthic/lpi3_delta1.0_model_predict_123 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 
python trept_unsup_mixup_crowd_pals.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Benthic/lpi3_delta1.0_model_predict_123_all_trainset \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50    
python trept_unsup_mixup_crowd_pals.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Benthic/exp_D/lpi3_delta1.0_model_predict_123_all_trainset_rebalance_allsr_fcw1cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 1.0 --consistency_weight 0.9 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 1
python trept_unsup_mixup_crowd_pals.py \
    --dataset Plankton \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Plankton/exp_D/lpi3_delta1.0_model_predict_1234567_all_trainset_rebalance_allsr_fcw1cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 1.0 --consistency_weight 0.9 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 1   
python trept_unsup_mixup_crowd_pals.py \
    --dataset Pig \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Pig/exp_D/lpi3_delta1.0_model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.9 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 0     
python trept_unsup_mixup_crowd_pals.py \
    --dataset MiceBone \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/MiceBone/exp_D/lpi3_delta1.0_model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.9 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 0   
python trept_unsup_mixup_crowd_pals.py \
    --dataset QualityMRI    \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/QualityMRI/exp_D/lpi3_delta1.0_model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.9 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 0  
python trept_unsup_mixup_crowd_pals.py \
    --dataset Turkey    \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Turkey/exp_D/lpi3_delta1.0_model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.9 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 0    
Synthetic     
python trept_unsup_mixup_crowd_pals.py \
    --dataset Synthetic    \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Synthetic/exp_D/lpi3_delta1.0_model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.9 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R18  --cuda_dev 1 
python trept_unsup_mixup_crowd_pals.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Treeversity/exp_D/lpi3_delta1.0_model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.9 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.9 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 1     

python trept_unsup_mixup_crowd_pals.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Treeversity/exp_D/lpi10_delta1.0/model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.5 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.5 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 1    
python trept_unsup_mixup_crowd_pals.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Treeversity/exp_D/lpi10_delta1.0/model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.5_cuda0 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.5 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 0        
python trept_unsup_mixup_crowd_pals.py \
    --dataset Benthic \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Benthic/exp_D/lpi10_delta1.0/model_predict_1234567_all_trainset_rebalance_allsr_fcw0cw0.5_cuda0 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 0.5 \
    --seeds 1 2 3 4 5 6 7 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50  --cuda_dev 1      
python trept_unsup_mixup_crowd_pals.py \
    --dataset Plankton \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Plankton/lpi3_delta1.0_model_predict_123 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 
python trept_unsup_mixup_crowd_pals.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 3 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Treeversity/lpi3_delta1.0_model_predict_123 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50 \
python trept_unsup_mixup_crowd_pals.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Treeversity/lpi10_delta1.0_model_predict_123 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50         

python trept_unsup_mixup_crowd_pals.py \
    --dataset Treeversity \
    --train_root ./data \
    --lpi 10 \
    --epochs 100 \
    --lr_decay_epochs 60 \
    --pr 0.1 --nr 0.0 \
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/Treeversity/lpi10_delta1.0_model_predict_123_rebalance_alllsr \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 1.0 \
    --network R50   


python trept_unsup_mixup_crowd_pals.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CUB200/lpi10_delta0.5_model_predict_123_p0.05n0.2 \
    --batch_size 32 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.5 \
    --network R18       

python trept_unsup_mixup_crowd_pals.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CUB200/delta0.25_model_predict_123_p0.05n0.2_bs64_rand_lam \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250     

python trept_unsup_mixup_crowd_pals.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CUB200/delta0.25_model_predict_123_p0.05n0.2_bs64_rand_lam_no_cross_mixup \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250      

python trept_unsup_mixup_crowd_pals.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CUB200/delta0.25_model_predict_123_p0.05n0.2_bs64_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --lr_decay_epochs  60 120 160 200 \  
    --delta 0.25 \
    --network R18 \
    --epochs  250    --cuda_dev  1  


python trept_unsup_mixup_crowd_pals.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CUB200/consensus_power1/delta0.25_model_predict_123_p0.05n0.2_bs64_rand_lam_no_cross_mixup_noncensus1.0 \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250       
python trept_unsup_mixup_crowd_pals.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CUB200/consensus_power2/exp_D/delta0.25_model_predict_123_p0.05n0.2_bs64_rand_lam_no_cross_mixup_noncensus1.0_all_lsr \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250    
#########################################################################

python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100H \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100H/consensus_power1/delta0.25_model_predict_123_p0.05n0.2_bs256_rand_lam_no_cross_mixup_noncensus1.0 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500        
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100H \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100H/consensus_power2/pw_directD/delta0.25_model_predict_123_p0.05n0.2_bs256_rand_lam_no_cross_mixup_noncensus1.0 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500     
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100H \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100H/consensus_power2/pw_expD/delta0.25_model_predict_123_p0.05n0.2_bs256_rand_lam_no_cross_mixup_noncensus1.0_alllsr \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500      
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power1/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus1.0 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500      


python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power1/random_beta/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500     


python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/random_beta/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500     
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/geometric/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500      

python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/geometric_norebalance/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500        

python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/pw_direct_D/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500      
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/pw_direct_D/delta0.5_model_predict_123_p0.05n0.5_bs256_soft_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs  500      
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.00 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/pw_direct_D/delta0.5_model_predict_123_p0.00n0.0_bs256_soft_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs  500     
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.00 --nr 0.0\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR10/consensus_power2/pw_direct_D/delta0.5_model_predict_123_p0.00n0.0_bs256_soft_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs  500     
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR10/consensus_power2/pw_direct_D/delta0.25_model_predict_123_p0.5n0.3_bs256_soft_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500    
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR10/consensus_power2/pw_exp_D/delta0.25_model_predict_123_p0.5n0.3_bs256_soft_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500     
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR10/consensus_power2/pw_exp_D/delta0.5_model_predict_123_p0.5n0.3_bs256_soft_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs  500     
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR10/consensus_power2/pw_exp_D/delta0.5_model_predict_123_p0.5n0.3_bs256_soft_lam_no_cross_mixup_noncensus0.5_all_lsr \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs  500       
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR10/consensus_power2/pw_exp_D/delta0.25_model_predict_123_p0.5n0.3_bs256_soft_lam_no_cross_mixup_noncensus0.5_all_lsr \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500       

python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/pw_exp_D/delta0.25_model_predict_123_p0.05n0.5_bs256_soft_lam_no_cross_mixup_noncensus0.5_all_lsr \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500  
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/pw_exp_D/delta0.25_model_predict_123_p0.05n0.3_bs256_soft_lam_no_cross_mixup_noncensus0.5_all_lsr \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500     
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/pw_direct_D/delta0.5_model_predict_123_p0.05n0.5_bs256_soft_lam_no_cross_mixup_noncensus0.5_all_lsr \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs  500        
python trept_unsup_mixup_crowd_pals.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals/CIFAR100/consensus_power2/pw_direct_D_no_rebalance/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500      
    
    """

# TPDS
"""python trept_unsup_mixup_crowd_pals_TPDS.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS/CIFAR100/consensus_power2/TPDS/delta0.5_model_predict_123_p0.05n0.5_bs256_soft_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0  --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.5 \
    --network R18 \
    --epochs  500 --cuda_dev 0"""


#standard
"""python trept_unsup_mixup_crowd_pals_TPDS_standard.py \
    --dataset CIFAR100H \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard/CIFAR100H/consensus_power1/delta0.25_model_predict_123_p0.05n0.2_bs256_rand_lam_no_cross_mixup_noncensus1.0 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 
    python trept_unsup_mixup_crowd_pals_TPDS_standard.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard/CIFAR100/consensus_power1/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus1.0 \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 cosine --sim_mode_2 exp"""
    
    
"""python trept_unsup_mixup_crowd_pals_TPDS_standard.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard/CIFAR100/consensus_power2/exp_exp/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus1.0_rectify \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 0 --sim_mode_1 exp --sim_mode_2 exp"""    

#exp D
"""python trept_unsup_mixup_crowd_pals_TPDS_standard.py \
    --dataset CIFAR100 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.5\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard/CIFAR100/consensus_power2/exp_D/delta0.25_model_predict_123_p0.05n0.5_bs256_rand_lam_no_cross_mixup_noncensus1.0_rectify \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 exp --sim_mode_2 D"""        
#exp D
"""python trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta_MH_KNN_high_index_topology.py \
    --dataset CIFAR10 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.5 --nr 0.3\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta_MH_KNN_high_index_topology/CIFAR10/consensus_power2/exp_daes/delta0.25_model_predict_123_p0.5n0.3_bs256_rand_lam_no_cross_mixup_noncensus1.0_rectify \
    --batch_size 256 \
    --lr 0.1 \
    --wd 1e-3 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler cosine \
    --delta 0.25 \
    --network R18 \
    --epochs  500 --cuda_dev 1 --sim_mode_1 topology_entropy --sim_mode_2 daes --num_worker 2        """      


"""python trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta/CUB200/exp_exp/delta0.25_model_predict_123_p0.05n0.2_bs64_rand_lam_no_cross_mixup_noncensus0.5 \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.25 \
    --network R18 \
    --epochs  250 \
    --cuda_dev 0 --sim_mode_1 exp --sim_mode_2 exp     


python trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta_MH_KNN_high_index.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta_MH_KNN_high_index/CUB200/exp_daes/delta0.5_model_predict_123_p0.05n0.2_bs64 \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.5 \
    --network R18 \
    --epochs  250 \
    --cuda_dev 1 --sim_mode_1 exp --sim_mode_2 daes --num_worker 1  
python trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta_MH_KNN_high_index_topology.py \
    --dataset CUB200 \
    --train_root ./data \
    --lpi 10 \
    --pr 0.05 --nr 0.2\
    --out ./out_ultimate \
    --exp_name PALS_softMatch/trept_unsup_mixup_crowd_pals_TPDS_standard_rank_calibrated_differ_exp_delta_MH_KNN_high_index_topology/CUB200/topo_daes/delta0.5_model_predict_123_p0.05n0.2_bs64 \
    --batch_size 64 \
    --lr 0.05 \
    --wd 5e-4 \
    --feature_consistency_weight 0.0 --consistency_weight 1.0 \
    --seeds 1 2 3 \
    --lsr 0.50 \
    --detailed_log \
    --lr_scheduler step \
    --delta 0.5 \
    --network R18 \
    --epochs  250 \
    --cuda_dev 1 --sim_mode_1 topology_entropy --sim_mode_2 daes --num_worker 1          
"""      
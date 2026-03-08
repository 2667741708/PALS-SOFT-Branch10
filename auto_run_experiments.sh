#!/bin/bash

LOG_FILE="experiment_scheduler.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

wait_for_gpu() {
    local gpu_id=$1
    log "Waiting for GPU $gpu_id to become free..."
    while true; do
        # 提取显存使用量 (MiB)
        MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
        # 提取 GPU 利用率 (%)
        UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
        
        # 如果显存使用小于 1000MB 且 GPU 利用率小于 10%，认为空闲
        if [ "$MEM_USED" -lt 1000 ] && [ "$UTIL" -lt 10 ]; then
            log "GPU $gpu_id is now free (Mem: ${MEM_USED}MiB, Util: ${UTIL}%)."
            break
        fi
        sleep 60
    done
}

run_experiment_1() {
    log "Starting Branch 2: Label-Suppression (No-Fuse) on GPU 1"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate torch_cuda128_whm
    
    python "结合投影_众包_自节点标签级权重抑制.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "结合投影_众包_自节点标签级权重抑制/CIFAR10/pr0.5nr0.3e100topdaes_topdaes_hl15_kh1_nofuse" \
    --pr 0.5 --nr 0.3 --epochs 100 --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 --knn_heads 1 --seeds 1 2 3 --cuda_dev 1 >> branch2_nofuse.log 2>&1
    log "Branch 2 completed."
}

run_experiment_2() {
    log "Starting Branch 3: Label-Suppression (Weighted-Sum) on GPU 0"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate torch_cuda128_whm
    
    python "结合投影_众包_自节点_加权和.py" \
    --dataset CIFAR10 \
    --out topology_daes \
    --exp_name "结合投影_众包_自节点_加权和/CIFAR10/pr0.5nr0.3e100topdaes_topdaes_hl15_kh1_fuse" \
    --pr 0.5 --nr 0.3 --epochs 100 --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
    --lr 0.1 --wd 1e-3 --history_len 15 --k_val 15 --lsr 0.0 --knn_heads 1 --seeds 1 2 3 --enable_knn1_model_fuse --fusion_mode weighted_sum --cuda_dev 0 >> branch3_fuse.log 2>&1
    log "Branch 3 completed."
}

log "Scheduler started."

# 假设当前卡 1 上运行的旧实验（因为缺少 torchvision 报错或被中断）已经停止，我们在此处排队。
# 也可以同时监测卡0和卡1。由于有两个剩余实验，我们将它们分别塞入 0 号和 1 号卡的等待队列中。

# 后台并行等待：如果卡1空出来，跑 Branch 2
(wait_for_gpu 1 && run_experiment_1) &

# 后台并行等待：如果卡0空出来，跑 Branch 3
(wait_for_gpu 0 && run_experiment_2) &

log "Jobs have been dispatched to wait for GPUs. Check experiment_scheduler.log for status."
wait
log "All scheduled experiments finished."

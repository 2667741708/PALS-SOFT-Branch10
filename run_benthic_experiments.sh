#!/bin/bash
# ==============================================================================
# Benthic 实验并行调度脚本 (6支 x 3 seeds = 18次实验)
# GPU 0: B0 (baseline) → B1 (knn_heads=4) → B2 (delta=0.75)
# GPU 1: B3 (delta=0.5) → B4 (k_val=10) → B5 (model_fuse)
# 生成时间: 2026-03-04
# Python 环境: torch_rtx5080_pals (PyTorch 2.11+cu128)
# ==============================================================================

PYTHON="/home/c201/miniconda3/envs/torch_rtx5080_pals/bin/python"
SCRIPT_DIR="/home/c201/公共/whm/PALS-SOFT/单流拓扑共识"
SCRIPT="结合投影_众包_自节点.py"
OUT="./topology_daes"
BASE_ARGS="--train_root ./data --dataset Benthic --lpi 10 --out ${OUT} \
    --network R50 --epochs 100 --batch_size 32 \
    --lr 0.05 --wd 5e-4 \
    --sim_mode_1 topology_daes --sim_mode_2 topology_daes \
    --delta 1.0 --lsr 0.0 \
    --lr_scheduler step --lr_decay_epochs 60 --lr_decay_rate 0.2 \
    --seeds 1 2 3 \
    --detailed_log \
    --history_len 100 --knn_heads 1"

EXP_ROOT="结合投影_众包_自节点/Benthic/LPI10"

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Benthic 实验组"
echo "[INFO] Python: ${PYTHON}"
echo "============================================================"

# ---------- GPU 0 任务组 (B0 → B1 → B2 顺序执行) ----------
run_gpu0() {
    cd "${SCRIPT_DIR}"

    # B0: Baseline (参考线)
    echo "[GPU0][$(date '+%H:%M:%S')] >> 开始 B0 Baseline"
    ${PYTHON} "${SCRIPT}" \
        ${BASE_ARGS} \
        --k_val 5 \
        --cuda_dev 0 \
        --exp_name "${EXP_ROOT}/B0_baseline_del1.0_k5_hl100_h1"
    echo "[GPU0][$(date '+%H:%M:%S')] << 完成 B0 (exit=$?)"

    # B1: 多头KNN (knn_heads: 1→4)
    echo "[GPU0][$(date '+%H:%M:%S')] >> 开始 B1 knn_heads=4"
    ${PYTHON} "${SCRIPT}" \
        ${BASE_ARGS} \
        --k_val 5 --knn_heads 4 \
        --cuda_dev 0 \
        --exp_name "${EXP_ROOT}/B1_knn_heads4_del1.0_k5_hl100"
    echo "[GPU0][$(date '+%H:%M:%S')] << 完成 B1 (exit=$?)"

    # B2: 更严格门控 (delta: 1.0→0.75)
    echo "[GPU0][$(date '+%H:%M:%S')] >> 开始 B2 delta=0.75"
    ${PYTHON} "${SCRIPT}" \
        ${BASE_ARGS} \
        --k_val 5 --delta 0.75 \
        --cuda_dev 0 \
        --exp_name "${EXP_ROOT}/B2_delta0.75_k5_hl100_h1"
    echo "[GPU0][$(date '+%H:%M:%S')] << 完成 B2 (exit=$?)"
}

# ---------- GPU 1 任务组 (B3 → B4 → B5 顺序执行) ----------
run_gpu1() {
    cd "${SCRIPT_DIR}"

    # B3: 宽松门控 (delta: 1.0→0.5)
    echo "[GPU1][$(date '+%H:%M:%S')] >> 开始 B3 delta=0.5"
    ${PYTHON} "${SCRIPT}" \
        ${BASE_ARGS} \
        --k_val 5 --delta 0.5 \
        --cuda_dev 1 \
        --exp_name "${EXP_ROOT}/B3_delta0.5_k5_hl100_h1"
    echo "[GPU1][$(date '+%H:%M:%S')] << 完成 B3 (exit=$?)"

    # B4: 扩大KNN邻域 (k_val: 5→10)
    echo "[GPU1][$(date '+%H:%M:%S')] >> 开始 B4 k_val=10"
    ${PYTHON} "${SCRIPT}" \
        ${BASE_ARGS} \
        --k_val 10 \
        --cuda_dev 1 \
        --exp_name "${EXP_ROOT}/B4_k10_del1.0_hl100_h1"
    echo "[GPU1][$(date '+%H:%M:%S')] << 完成 B4 (exit=$?)"

    # B5: 激活模型几何融合 (enable_knn1_model_fuse)
    echo "[GPU1][$(date '+%H:%M:%S')] >> 开始 B5 model_fuse"
    ${PYTHON} "${SCRIPT}" \
        ${BASE_ARGS} \
        --k_val 5 \
        --enable_knn1_model_fuse \
        --cuda_dev 1 \
        --exp_name "${EXP_ROOT}/B5_model_fuse_del1.0_k5_hl100_h1"
    echo "[GPU1][$(date '+%H:%M:%S')] << 完成 B5 (exit=$?)"
}

# ---------- 并行启动两个GPU任务组 ----------
run_gpu0 2>&1 | tee /tmp/benthic_gpu0.log &
PID_GPU0=$!

run_gpu1 2>&1 | tee /tmp/benthic_gpu1.log &
PID_GPU1=$!

echo "[Main][$(date '+%H:%M:%S')] GPU0 PID=${PID_GPU0}, GPU1 PID=${PID_GPU1}"
echo "[Main] 等待所有实验完成..."

wait $PID_GPU0
EXITCODE_GPU0=$?
echo "[Main][$(date '+%H:%M:%S')] GPU0 任务组结束 (exit=${EXITCODE_GPU0})"

wait $PID_GPU1
EXITCODE_GPU1=$?
echo "[Main][$(date '+%H:%M:%S')] GPU1 任务组结束 (exit=${EXITCODE_GPU1})"

# ---------- 汇总结果 ----------
echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部实验完成！汇总结果："
echo "============================================================"

OUT_BASE="${SCRIPT_DIR}/topology_daes/${EXP_ROOT}"
for branch in \
    "B0_baseline_del1.0_k5_hl100_h1" \
    "B1_knn_heads4_del1.0_k5_hl100" \
    "B2_delta0.75_k5_hl100_h1" \
    "B3_delta0.5_k5_hl100_h1" \
    "B4_k10_del1.0_hl100_h1" \
    "B5_model_fuse_del1.0_k5_hl100_h1"; do
    LOG="${OUT_BASE}/${branch}/master_log.txt"
    if [ -f "${LOG}" ]; then
        echo "--- ${branch} ---"
        grep "Final Reported" "${LOG}"
    else
        echo "--- ${branch} --- [LOG NOT FOUND: ${LOG}]"
    fi
done

echo "============================================================"
echo "详细日志: /tmp/benthic_gpu0.log  /tmp/benthic_gpu1.log"
echo "============================================================"

#!/bin/bash
# ============================================================
# Project 24: Launch MAP-Elites search on cloud GPU
#
# Strategy: run N parallel workers, each writing to its own
# results directory. Merge archives afterwards.
#
# Usage:
#   bash cloud/run_search.sh              # auto-detect GPUs
#   bash cloud/run_search.sh 4            # force 4 workers
#   HOURS=12 RES=256 bash cloud/run_search.sh  # override defaults
# ============================================================
set -e

cd /workspace/project_lenia_search

# --- Configuration (override via env vars) ---
N_WORKERS="${1:-$(python3 -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
HOURS="${HOURS:-8}"
RES="${RES:-128}"
STEPS="${STEPS:-8000}"
BATCH="${BATCH:-20}"
SEED_ARCHIVE="${SEED_ARCHIVE:-}"   # path to existing archive to resume from

echo "============================================================"
echo "  Project 24: Cloud MAP-Elites Search"
echo "============================================================"
echo "  Workers    : ${N_WORKERS}"
echo "  Hours      : ${HOURS}"
echo "  Resolution : ${RES}x${RES}"
echo "  Steps/eval : ${STEPS}"
echo "  Batch size : ${BATCH}"
echo "  Seed archive: ${SEED_ARCHIVE:-none (fresh start)}"
echo "============================================================"

# --- Launch workers ---
mkdir -p results_cloud

for i in $(seq 0 $((N_WORKERS - 1))); do
    WORKER_DIR="results_cloud/worker_${i}"
    mkdir -p "$WORKER_DIR"

    # Each worker uses a different GPU (round-robin)
    GPU_ID=$((i % $(python3 -c 'import torch; print(max(torch.cuda.device_count(), 1))')))

    echo "Starting worker ${i} on GPU ${GPU_ID} -> ${WORKER_DIR}"

    # Build command
    CMD="python3 run_search.py \
        --hours ${HOURS} \
        --batch_size ${BATCH} \
        --resolution ${RES} \
        --steps ${STEPS} \
        --device cuda \
        --results_dir ${WORKER_DIR} \
        --fitness complexity \
        --init mixed"

    # Add seed archive if provided
    if [ -n "${SEED_ARCHIVE}" ] && [ -f "${SEED_ARCHIVE}" ]; then
        CMD="${CMD} --archive ${SEED_ARCHIVE}"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup ${CMD} \
        > "${WORKER_DIR}/search_log.txt" 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "All ${N_WORKERS} workers launched."
echo "Monitor: tail -f results_cloud/worker_0/search_log.txt"
echo "Check:   python3 cloud/check_progress.py"
echo ""
echo "After completion, merge: python3 cloud/merge_archives.py"

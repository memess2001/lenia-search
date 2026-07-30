#!/bin/bash
# ============================================================
# Upload project code to Vast.ai instance
# Usage: ./cloud/upload.sh <user@host> [port]
#   e.g. ./cloud/upload.sh root@123.45.67.89 22222
# ============================================================
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <user@host> [ssh_port]"
    echo "  e.g. $0 root@ssh5.vast.ai 12345"
    exit 1
fi

HOST="$1"
PORT="${2:-22}"
REMOTE_DIR="/workspace/project_lenia_search"

echo "=== Uploading to ${HOST}:${PORT} ==="

# Create remote directory
ssh -p "$PORT" "$HOST" "mkdir -p ${REMOTE_DIR}"

# Rsync project (exclude large result dirs, venv, git)
rsync -avz --progress \
    -e "ssh -p ${PORT}" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude 'results/' \
    --exclude 'results_localized/' \
    --exclude 'results_phase2_screen/' \
    --exclude '*.gif' \
    --exclude '*.png' \
    ./ "${HOST}:${REMOTE_DIR}/"

echo ""
echo "=== Upload complete! Now run on the instance: ==="
echo "  ssh -p ${PORT} ${HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  bash cloud/setup.sh"
echo "  bash cloud/run_search.sh"

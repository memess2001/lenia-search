#!/bin/bash
# ============================================================
# Project 24: Vast.ai environment setup
# Run once after SSH into the instance
# ============================================================
set -e

echo "=== Project 24: Cloud Environment Setup ==="

# --- System deps ---
apt-get update -qq && apt-get install -y -qq git rsync > /dev/null 2>&1
echo "[1/4] System deps OK"

# --- Python deps (PyTorch should be pre-installed on Vast.ai CUDA images) ---
pip install -q numpy scipy Pillow tqdm
echo "[2/4] Python deps OK"

# --- Verify torch + CUDA ---
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU count: {torch.cuda.device_count()}')
"
echo "[3/4] PyTorch verified"

# --- Verify project code ---
cd /workspace/project_lenia_search
python3 -c "
from src.lenia_core import KERNEL_REGISTRY, GROWTH_REGISTRY
from src.genome import random_genome, LENIA_GENOME
print(f'  Kernels: {len(KERNEL_REGISTRY)}')
print(f'  Growth:  {len(GROWTH_REGISTRY)}')
print(f'  Genome kernel choices: {len(LENIA_GENOME[\"kernel_type\"][\"choices\"])}')
assert len(KERNEL_REGISTRY) == 28
assert len(GROWTH_REGISTRY) == 16
print('  All registries OK')
"
echo "[4/4] Project code verified"

echo ""
echo "=== Setup complete! Ready to run search. ==="

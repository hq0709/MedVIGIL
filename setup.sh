#!/bin/bash
# MedVision: Setup script for Medical VLM Study
# This script sets up the environment, clones LLaVA, and downloads models/data.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "MedVision: Medical VLM Study Setup"
echo "============================================"

# --- Step 1: Create conda environment ---
echo "[1/5] Setting up Python environment..."
if ! conda info --envs | grep -q "medvlm"; then
    conda create -n medvlm python=3.10 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate medvlm

# --- Step 2: Install dependencies ---
echo "[2/5] Installing Python dependencies..."
pip install -r requirements.txt

# --- Step 3: Clone LLaVA repository (needed for model architecture) ---
echo "[3/5] Cloning LLaVA repository..."
if [ ! -d "LLaVA" ]; then
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip install -e ".[train]"
    cd ..
else
    echo "LLaVA already cloned, skipping."
fi

# --- Step 4: Download model weights ---
echo "[4/5] Downloading model weights..."
python download_model.py

# --- Step 5: Download VQA-RAD dataset ---
echo "[5/5] Downloading VQA-RAD dataset..."
python data/download_vqa_rad.py

echo "============================================"
echo "Setup complete!"
echo "Activate environment: conda activate medvlm"
echo "Run experiments:      python experiments/run_all.py"
echo "============================================"

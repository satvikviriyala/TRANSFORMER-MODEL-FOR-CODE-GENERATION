#!/bin/bash

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# Load environment variables from .env file
set -a # automatically export all variables
source .env
set +a

# --- Configuration ---
NUM_GPUS=${WORLD_SIZE:-4} # Default to 4 if WORLD_SIZE not in .env
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-12356}

# --- Paths (Ensure these are correct or passed as args) ---
CONFIG_PATH="configs/model_config.yaml"
# DATA_PATH is read from config inside the script now
OUTPUT_DIR=${PRETRAIN_OUTPUT_DIR:-"checkpoints/pretrained/"}

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Starting DDP Pre-training..."
echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Num GPUs: $NUM_GPUS"
echo "Output Dir: $OUTPUT_DIR"

# Use torchrun (recommended)
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    src/training/pretrain_ddp.py \
    # Add command line args here if pretrain_ddp.py uses argparse
    # --config_path $CONFIG_PATH \
    # --output_dir $OUTPUT_DIR

echo "DDP Pre-training finished."
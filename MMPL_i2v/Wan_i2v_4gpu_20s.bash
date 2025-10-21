#!/bin/bash

# Automatically find the checkpoint with the largest number
CKPT_PATH=pretrained_models/i2v_14B_6k.pt
# Set config and data paths
CONFIG_PATH="configs/self_forcing_df.yaml"
DATA_PATH="i2v_data"
SEED=0

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Data directory not found: $DATA_PATH"
    exit 1
fi

# Create dynamic output folder with step number
BASE_OUTPUT_DIR="outputs_test/long_20s"

echo "✅ output_folder: $BASE_OUTPUT_DIR"

(
  CUDA_VISIBLE_DEVICES=0,1,2,3 python Wan_fps_inference_parallel_4gpu_20s.py \
    --config_path $CONFIG_PATH \
    --output_folder $BASE_OUTPUT_DIR\
    --checkpoint_path $CKPT_PATH \
    --data_path $DATA_PATH \
    --seed $SEED\
    --i2v
) &

# Wait for all background jobs to complete
wait
echo "✅ All inference jobs completed."
echo "✅ Results saved in: $BASE_OUTPUT_DIR"
#!/bin/bash

CKPT_PATH="pretrained_models/t2v_14B_8k.pt"
# Set config and data paths
CONFIG_PATH="configs/self_forcing_df.yaml"
DATA_PATH="prompts/test_prompts.txt"
DURATION=2
SEED=0

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Data file not found: $DATA_PATH"
    exit 1
fi

# Create dynamic output folder with step number
BASE_OUTPUT_DIR="outputs_test"

echo "✅ output_folder: $BASE_OUTPUT_DIR"

(
  CUDA_VISIBLE_DEVICES=0 python Wan_fps_inference_1gpu.py \
    --config_path $CONFIG_PATH \
    --output_folder $BASE_OUTPUT_DIR \
    --checkpoint_path $CKPT_PATH \
    --data_path $DATA_PATH \
    --duration $DURATION \
    --seed $SEED
) &

wait
echo "✅ All inference jobs completed."
echo "✅ Results saved in: $BASE_OUTPUT_DIR"
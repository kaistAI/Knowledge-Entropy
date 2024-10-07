#!/bin/bash

# Assign the single path to a variable
CKPT_PATH="$1"
echo "Processing checkpoint path: $CKPT_PATH"

SCRIPT_DIR=$(dirname "$(realpath "$BASH_SOURCE")")
ROOT_DIR=$(realpath "$SCRIPT_DIR/..")
echo "Root dir is $ROOT_DIR"

# Extract the number after "step"
step_number=$(echo "$CKPT_PATH" | sed -n 's/.*step\([0-9]*\)-unsharded\/.*/\1/p')
echo "Extracted step number: $step_number"

OUTPUT_PATH="checkpoints/pretrained_1B/$step_number-unsharded"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"
# mkdir -p "$OUTPUT_PATH/hf"

# Change to the output directory
cd "$OUTPUT_PATH"

# Download the config.yaml file
wget "${CKPT_PATH}config.yaml"
wget "${CKPT_PATH}model.pt"
wget "${CKPT_PATH}train.pt"

cd "$ROOT_DIR"

# python scripts/convert_olmo_to_hf_new.py --input_dir "${OUTPUT_PATH}" --output_dir "${OUTPUT_PATH}/hf" --tokenizer_json_path tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json

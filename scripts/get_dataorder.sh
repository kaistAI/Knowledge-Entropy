#!/bin/bash

# Define the target file path
FILE_PATH="data/global_indices/1B/global_indices.npy"

# Define the download URL
DOWNLOAD_URL="https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy"

# Check if the file exists
if [ -f "$FILE_PATH" ]; then
    echo "File already exists at: $FILE_PATH. Exit"
else
    mkdir -p "$(dirname "$FILE_PATH")"
    wget -O "$FILE_PATH" "$DOWNLOAD_URL"
    echo "Download complete at $FILE_PATH"    
fi
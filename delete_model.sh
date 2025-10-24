#!/bin/bash
# Script to delete a cached Hugging Face model

if [ $# -eq 0 ]; then
    echo "Usage: ./delete_model.sh <model-name>"
    echo ""
    echo "Examples:"
    echo "  ./delete_model.sh defog/sqlcoder-7b-2"
    echo "  ./delete_model.sh mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
    echo ""
    echo "Available models:"
    for model_dir in ~/.cache/huggingface/hub/models--*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir" | sed 's/models--//' | sed 's/--/\//g')
            size=$(du -sh "$model_dir" | cut -f1)
            echo "  - $model_name ($size)"
        fi
    done
    exit 1
fi

# Convert model name to directory format (e.g., mlx-community/Qwen -> models--mlx-community--Qwen)
model_name="$1"
model_dir_name="models--$(echo "$model_name" | sed 's/\//--/g')"
model_path="$HOME/.cache/huggingface/hub/$model_dir_name"

if [ ! -d "$model_path" ]; then
    echo "Error: Model '$model_name' not found in cache."
    echo "Path checked: $model_path"
    exit 1
fi

size=$(du -sh "$model_path" | cut -f1)
echo "Model: $model_name"
echo "Size: $size"
echo "Path: $model_path"
echo ""
read -p "Are you sure you want to delete this model? (y/N): " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Deleting $model_name..."
    rm -rf "$model_path"
    echo "Model deleted successfully!"
else
    echo "Deletion cancelled."
fi

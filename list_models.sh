#!/bin/bash
# Script to list cached Hugging Face models

echo "Cached Hugging Face Models:"
echo "=========================================="
echo ""

for model_dir in ~/.cache/huggingface/hub/models--*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir" | sed 's/models--//' | sed 's/--/\//g')
        size=$(du -sh "$model_dir" | cut -f1)
        files=$(find "$model_dir/snapshots" -type f 2>/dev/null | wc -l | tr -d ' ')
        last_modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$model_dir" 2>/dev/null)
        printf "Model: %-60s\n" "$model_name"
        printf "  Size: %-10s Files: %-6s Last Modified: %s\n" "$size" "$files" "$last_modified"
        printf "  Path: %s\n\n" "$model_dir"
    fi
done

total_size=$(du -sh ~/.cache/huggingface/hub/ | cut -f1)
echo "=========================================="
echo "Total cache size: $total_size"

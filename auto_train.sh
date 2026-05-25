#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== 1/3 重新 tokenize（加上 EOS 文件分隔符）==="
python3 prepare_data.py --force --jobs 4

echo ""
echo "=== 2/3 備份舊 checkpoint ==="
if [ -d checkpoints ] && [ -n "$(ls -A checkpoints 2>/dev/null)" ]; then
    mv checkpoints "checkpoints_old_$(date +%Y%m%d_%H%M%S)"
    echo "  已搬移至 checkpoints_old_*"
else
    echo "  無舊 checkpoint，跳過"
fi

echo ""
echo "=== 3/3 開始訓練 ==="
python3 train.py
 
if [ $? -eq 0 ]; then
    echo "訓練完成！"
else
    echo "訓練過程中發生錯誤，請檢查輸出訊息。"
fi

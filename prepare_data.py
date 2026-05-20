"""將 input{1..N}.txt 合併轉換為 DataLoaderLite 可讀取的 .npy shard 格式。
使用 chunk 方式處理大檔案，避免 Python int list 吃爆記憶體。"""
import os
import re
import math
import logging

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prepare_data")

data_root = "edu_fineweb10B"
shard_size = 1_000_000
val_ratio = 0.01
CHUNK_SIZE = 10_000_000  # 每次讀取 10MB 文字，控制 token list 大小

os.makedirs(data_root, exist_ok=True)
enc = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)


def find_input_files():
    pattern = re.compile(r"^input(\d+)\.txt$")
    files = []
    for fname in os.listdir("."):
        m = pattern.match(fname)
        if m:
            files.append((int(m.group(1)), fname))
    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError("找不到 input1.txt, input2.txt, ... 任何 inputX.txt")
    logger.info(f"找到 {len(files)} 個輸入檔: {', '.join(f[1] for f in files)}")
    return [f for _, f in files]


input_files = find_input_files()

# 估算總 token 數（取前 1000 行）
logger.info("估算 token 總數...")
total_estimated = 0
for fname in input_files:
    sample_lines = []
    with open(fname, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            sample_lines.append(line)
    sample_text = "".join(sample_lines)
    sample_tokens = enc.encode(sample_text)
    tokens_per_char = len(sample_tokens) / max(len(sample_text), 1)
    file_size = os.path.getsize(fname)
    estimated = int(tokens_per_char * file_size)
    total_estimated += estimated
    logger.info(f"  {fname}: ~{estimated / 1e6:.1f}M tokens")

logger.info(f"估計總 token 數: ~{total_estimated / 1e6:.1f}M")

# 用 chunk 方式處理每個檔案，避免 Python list 累積到爆記憶體
# 每個 chunk 的 token list 用完就轉 numpy，不保留巨量 Python ints
logger.info("開始讀取並 tokenize 所有檔案（chunk 模式）...")
file_arrays = []  # 存每個檔案的 numpy array

for fname in input_files:
    logger.info(f"  讀取 {fname}...")
    chunk_arrays = []
    with open(fname, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            tokens = enc.encode(chunk)
            chunk_arrays.append(np.array(tokens, dtype=np.int32))

    if chunk_arrays:
        combined = np.concatenate(chunk_arrays)
        file_arrays.append(combined)
        logger.info(f"  {fname}: {len(combined):,} tokens")

# 合併所有檔案
tokens = np.concatenate(file_arrays) if file_arrays else np.array([], dtype=np.int32)
del file_arrays  # 釋放各檔案的陣列
logger.info(f"總 token 數: {len(tokens):,}")

# 分割 train/val
val_cutoff = int(len(tokens) * val_ratio)
val_tokens = tokens[:val_cutoff]
train_tokens = tokens[val_cutoff:]
logger.info(f"train tokens: {len(train_tokens):,}, val tokens: {len(val_tokens):,}")


def save_shards(tokens_arr, prefix):
    n_shards = math.ceil(len(tokens_arr) / shard_size)
    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, len(tokens_arr))
        shard = tokens_arr[start:end].astype(np.int32)
        filename = os.path.join(data_root, f"{prefix}_{i:02d}.npy")
        np.save(filename, shard)
        logger.info(f"  saved {filename} — shape {shard.shape}")


logger.info("儲存 val shards...")
save_shards(val_tokens, "val")

logger.info("儲存 train shards...")
save_shards(train_tokens, "train")

n_files = len([f for f in os.listdir(data_root) if f.endswith(".npy")])
logger.info(f"完成！共 {n_files} 個 shard 檔案寫入 {data_root}/")

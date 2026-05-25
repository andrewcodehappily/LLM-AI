"""將 input{1..N}.txt 合併轉換為 DataLoaderLite 可讀取的 .npy shard 格式。

支援增量模式：已 tokenize 過的檔案會自動跳過，只處理新檔案。
用法: python3 prepare_data.py                # 增量模式，單進程
      python3 prepare_data.py --jobs 4        # 增量模式，4 進程 parallel
      python3 prepare_data.py --force --jobs 4  # 全部重來 + 多進程
"""
import os, re, math, logging, time, sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger("prepare_data")

data_root = "edu_fineweb10B"
shard_size = 1_000_000
val_ratio = 0.01
CHUNK_SIZE = 10_000_000
temp_dir = os.path.join(data_root, "_temp")


def find_input_files():
    pattern = re.compile(r"^input(\d+)\.txt$")
    files = []
    for fname in os.listdir("."):
        m = pattern.match(fname)
        if m:
            files.append((int(m.group(1)), fname))
    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError("找不到 input1.txt, input2.txt, ...")
    logger.info(f"找到 {len(files)} 個輸入檔")
    return [f for _, f in files]


def get_existing_token_counts(temp_dir, input_files):
    """回傳 {fname: token_count} 已存在 temp .npy 的檔案。"""
    existing = {}
    if not os.path.exists(temp_dir):
        return existing
    for fname in input_files:
        npy_path = os.path.join(temp_dir, f"{fname}.npy")
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            existing[fname] = len(arr)
            del arr
    return existing


def tokenize_one(fname, enc, chunk_callback=False):
    """單一檔案 tokenize → 存暫存 .npy，回傳 (fname, token_count)。"""
    chunk_arrays = []
    chunk_i = 0
    with open(fname, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            t0 = time.time()
            tokens = enc.encode(chunk)
            chunk_arrays.append(np.array(tokens, dtype=np.int32))
            dt = time.time() - t0
            chunk_i += 1
            if chunk_callback and chunk_i % 10 == 0:
                logger.info(f"    chunk {chunk_i}: {len(tokens):,} tokens in {dt:.2f}s")
                sys.stdout.flush()
    combined = np.concatenate(chunk_arrays)
    # Append document separator (Qwen <|endoftext|> = 151643)
    combined = np.append(combined, enc.eos_token_id)
    out_path = os.path.join(temp_dir, f"{fname}.npy")
    np.save(out_path, combined)
    return fname, len(combined)


def tokenize_worker(fname):
    """給 multiprocessing 用的 worker：獨立載入 tokenizer，處理一個檔案。"""
    # 每個 worker 有自己的 logger
    worker_logger = logging.getLogger(f"worker-{fname}")
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    t0 = time.time()
    _fname, n_tokens = tokenize_one(fname, enc)
    dt = time.time() - t0
    size_mb = os.path.getsize(fname) / (1024 * 1024)
    tok_speed = n_tokens / dt
    return fname, n_tokens, size_mb, dt, tok_speed


def build_shards(prefix, range_start, range_end, input_files, cum_offsets, out_dir):
    """從 temp .npy 讀取 token 區間，組 shard 到 out_dir。"""
    n_shards = math.ceil((range_end - range_start) / shard_size)
    for si in range(n_shards):
        shard_start = range_start + si * shard_size
        shard_end = min(range_start + (si + 1) * shard_size, range_end)

        pieces = []
        for fi, fname in enumerate(input_files):
            f_start = cum_offsets[fi]
            f_end = cum_offsets[fi + 1]
            overlap_start = max(shard_start, f_start)
            overlap_end = min(shard_end, f_end)

            if overlap_start < overlap_end:
                local_start = overlap_start - f_start
                local_end = overlap_end - f_start
                arr = np.load(os.path.join(temp_dir, f"{fname}.npy"))
                pieces.append(arr[local_start:local_end])
                del arr

        shard = np.concatenate(pieces)
        filename = os.path.join(out_dir, f"{prefix}_{si:04d}.npy")
        np.save(filename, shard)
    logger.info(f"  {prefix}: {n_shards} shards done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="強制全部重新 tokenize")
    parser.add_argument("--jobs", type=int, default=1, help="平行 tokenize 的 worker 數（預設 1，M4 Max 建議 2~4）")
    args = parser.parse_args()

    os.makedirs(temp_dir, exist_ok=True)
    input_files = find_input_files()

    # === 載入 tokenizer（只一次） ===
    logger.info("載入 Qwen tokenizer...")
    t0 = time.time()
    enc = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    logger.info(f"tokenizer 載入耗時 {time.time()-t0:.1f}s")

    # === 檢查哪些已存在，只處理新檔案 ===
    existing_counts = get_existing_token_counts(temp_dir, input_files)
    if args.force:
        logger.info("--force 模式：全部重新 tokenize")
        existing_counts = {}

    logger.info(f"開始 tokenize（已有 {len(existing_counts)} 個檔案快取）...")

    # 找出需要 tokenize 的檔案
    new_files = [f for f in input_files if f not in existing_counts]

    file_lengths = []
    if new_files and args.jobs > 1:
        # === 多進程模式 ===
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info(f"多進程模式：{args.jobs} workers 平行處理 {len(new_files)} 個檔案")
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(tokenize_worker, f): f for f in new_files}
            for future in as_completed(futures):
                fname, n_tokens, size_mb, dt, tok_speed = future.result()
                logger.info(f"  {fname} ({size_mb:.0f}MB) -> {n_tokens:,} tokens in {dt:.0f}s ({tok_speed:,.0f} tok/s)")
                file_lengths.append((fname, n_tokens))
        # 保持跟 input_files 順序一致
        file_lengths.sort(key=lambda x: input_files.index(x[0]))
        file_lengths = [n for _, n in file_lengths]
    else:
        # === 單進程模式（原本的順序 tokenize） ===
        for fname in input_files:
            if fname in existing_counts:
                n_tokens = existing_counts[fname]
                logger.info(f"  {fname}: 已存在，跳過 ({n_tokens:,} tokens)")
                file_lengths.append(n_tokens)
                continue

            t0 = time.time()
            size_mb = os.path.getsize(fname) / (1024 * 1024)
            _fname, n_tokens = tokenize_one(fname, enc, chunk_callback=True)
            dt = time.time() - t0
            tok_speed = n_tokens / dt
            logger.info(f"  {fname} ({size_mb:.0f}MB) -> {n_tokens:,} tokens in {dt:.0f}s ({tok_speed:,.0f} tok/s)")
            file_lengths.append(n_tokens)

    total_tokens = sum(file_lengths)
    logger.info(f"總 token 數: {total_tokens:,}")

    # 累計 offset
    cum_offsets = [0]
    for fl in file_lengths:
        cum_offsets.append(cum_offsets[-1] + fl)

    val_cutoff = int(total_tokens * val_ratio)
    logger.info(f"train tokens: {total_tokens - val_cutoff:,}, val tokens: {val_cutoff:,}")

    # === 安全組 shard：先寫到新目錄，全部完成再覆蓋 ===
    new_dir = os.path.join(data_root, "_new")
    os.makedirs(new_dir, exist_ok=True)

    logger.info("儲存 val shards...")
    build_shards("val", 0, val_cutoff, input_files, cum_offsets, new_dir)

    logger.info("儲存 train shards...")
    build_shards("train", val_cutoff, total_tokens, input_files, cum_offsets, new_dir)

    # 全部寫完確認無誤 → 刪舊 shard → 搬新 shard
    old_shards = [f for f in os.listdir(data_root) if f.endswith(".npy") and not f.startswith("_")]
    for f in old_shards:
        os.remove(os.path.join(data_root, f))
    logger.info(f"已刪除 {len(old_shards)} 個舊 shard")

    for f in os.listdir(new_dir):
        os.rename(os.path.join(new_dir, f), os.path.join(data_root, f))
    os.rmdir(new_dir)

    n_files = len([f for f in os.listdir(data_root) if f.endswith(".npy")])
    logger.info(f"完成！共 {n_files} 個 shard 寫入 {data_root}/")
    logger.info(f"（暫存檔保留在 {temp_dir}/，下次加新資料會自動跳過已處理的檔案）")

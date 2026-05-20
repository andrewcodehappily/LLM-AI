"""
下載 opencsg/chinese-fineweb-edu 的部分資料，
每個 parquet 檔案的 text 寫入獨立的 inputX.txt。
"""
import argparse
import logging
import os
import sys

import pyarrow.parquet as pq
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_fineweb_edu")

DATASET = "opencsg/chinese-fineweb-edu"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
# 只看 IndustryCorpus（第一部分，最通用的）
PREFIX = "IndustryCorpus"


def get_parquet_files():
    """透過 HuggingFace API 取得所有 parquet 檔案列表。"""
    r = requests.get(f"https://huggingface.co/api/datasets/{DATASET}")
    r.raise_for_status()
    siblings = r.json().get("siblings", [])
    files = [
        s["rfilename"]
        for s in siblings
        if s["rfilename"].endswith(".parquet") and s["rfilename"].startswith(PREFIX)
    ]
    files.sort()
    return files


def download_and_convert(parquet_path: str, output_txt: str):
    """下載一個 parquet 檔案，把 text 欄位寫入 output_txt。"""
    url = f"{BASE_URL}/{parquet_path}"
    logger.info(f"下載 {parquet_path} ...")

    # 下載到暫存檔案再讀取（避免網路不穩導致記憶體中的資料損毀）
    local_parquet = parquet_path.replace("/", "_")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(local_parquet, "wb") as f:
        for chunk in tqdm(
            resp.iter_content(chunk_size=10_000_000),
            desc=f"下載中",
            total=total // 10_000_000,
            unit="10MB",
        ):
            f.write(chunk)

    # 讀取 parquet 並寫入 text
    logger.info(f"讀取並寫入 {output_txt} ...")
    table = pq.read_table(local_parquet)
    texts = table.column("text")

    with open(output_txt, "w", encoding="utf-8") as f:
        for i in range(table.num_rows):
            text = str(texts[i].as_py())
            f.write(text + "\n")

    # 刪除暫存 parquet
    os.remove(local_parquet)
    logger.info(f"完成 {output_txt} ({table.num_rows} 行, 壓縮前檔案大小不明)")


def main():
    parser = argparse.ArgumentParser(
        description="下載 Chinese FineWeb-Edu 為 inputX.txt"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="要下載的 parquet 檔案數量（預設 10）",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="從第幾個 parquet 開始（用於續傳）",
    )
    args = parser.parse_args()

    all_files = get_parquet_files()
    logger.info(f"共有 {len(all_files)} 個 parquet 檔案")

    selected = all_files[args.start_from : args.start_from + args.num_files]
    logger.info(f"將下載 {len(selected)} 個，從索引 {args.start_from} 開始")

    # 找現有 input 的最大編號
    import re

    existing = []
    for fname in os.listdir("."):
        m = re.match(r"input(\d+)\.txt$", fname)
        if m:
            existing.append(int(m.group(1)))
    next_id = max(existing) + 1 if existing else 3

    for i, parquet_file in enumerate(selected):
        output = f"input{next_id + i}.txt"
        if os.path.exists(output):
            logger.info(f"跳過 {output}，已存在")
            continue
        download_and_convert(parquet_file, output)

    logger.info("全部下載完成！")


if __name__ == "__main__":
    main()

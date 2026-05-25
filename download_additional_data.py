"""下載 MNBVC 補充資料（知乎、ChatGPT、論壇）並輸出為 input*.txt 格式。

用法:
  python3 download_additional_data.py                    # 全部下載
  python3 download_additional_data.py --categories zhihu  # 只下載特定分類
  python3 download_additional_data.py --force             # 重新下載

下載內容:
  - zhihu: 知乎問答 (~1GB) → 對話式
  - chatgpt: ChatGPT 百度知道 (~1.5GB) → 對話式
  - forum: 糗事百科 (~15GB, 前10個parquet) → 論壇討論
"""
import os, sys, json, gzip, time, argparse, logging
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("download_data")

REPO = "liwu/MNBVC"
REPO_TYPE = "dataset"

# 各分類的遠端檔案清單（會動態抓取）
def get_category_files():
    """回傳 {分類: [檔案路徑]}"""
    logger.info("讀取 MNBVC 檔案清單...")
    files = list_repo_files(REPO, repo_type=REPO_TYPE)

    # --- 只挑有用的 ---
    # zhihu: qa/20230196/zhihu/*.jsonl.gz
    zhihu = sorted(f for f in files if f.startswith("qa/20230196/zhihu/") and f.endswith(".jsonl.gz"))

    # chatgpt: qa/chatgpt/20230211/*.jsonl.gz
    chatgpt = sorted(f for f in files if f.startswith("qa/chatgpt/") and f.endswith(".jsonl.gz"))

    # forum: qiushi parquet (只取前10個 ~15GB)
    forum = sorted(f for f in files if f.startswith("forum/qiushi_articles_parquet/") and f.endswith(".parquet"))
    forum = forum[:10]

    logger.info(f"zhihu: {len(zhihu)} 個檔案")
    logger.info(f"chatgpt: {len(chatgpt)} 個檔案")
    logger.info(f"forum: {len(forum)} 個檔案 ({len(forum)*1.5:.0f}GB)")

    return {"zhihu": zhihu, "chatgpt": chatgpt, "forum": forum}


def extract_text_zhihu(filepath):
    """從知乎 jsonl.gz 提取文字（每條: 問題 + 回答）"""
    texts = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            q = data.get("问", "").strip()
            a = data.get("答", "").strip()
            if q and a:
                texts.append(f"問：{q}\n答：{a}")
    return texts


def extract_text_chatgpt(filepath):
    """從 ChatGPT jsonl.gz 提取文字（每條: 問題 + 回答）"""
    texts = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            q = data.get("问", "").strip()
            a = data.get("答", "").strip()
            if q and a:
                texts.append(f"問：{q}\n答：{a}")
    return texts


def extract_text_forum(filepath):
    """從 forum parquet 提取文字（每條: 文本內容）"""
    table = pq.read_table(filepath, columns=["文本", "块类型"])
    texts = []
    for i in range(len(table)):
        t = table["文本"][i].as_py()
        typ = table["块类型"][i].as_py()
        if t and t != "None" and typ == "文本":
            texts.append(t.strip())
    return texts


# 每個分類的處理函數
EXTRACTORS = {
    "zhihu": extract_text_zhihu,
    "chatgpt": extract_text_chatgpt,
    "forum": extract_text_forum,
}


def download_and_extract_one(category, remote_path):
    """下載單一檔案並提取文字，回傳文字列表"""
    local_path = hf_hub_download(REPO, remote_path, repo_type=REPO_TYPE)
    extractor = EXTRACTORS[category]
    texts = extractor(local_path)
    return texts


def write_input_file(texts, filepath):
    """將文字列表寫入 input*.txt（每個 document 換行分隔）"""
    with open(filepath, "w", encoding="utf-8") as f:
        for t in texts:
            # 清理：取代連續換行為單一換行
            t = t.replace("\r\n", "\n").replace("\r", "\n")
            # 移除空行過多的段落
            lines = [l for l in t.split("\n") if l.strip()]
            cleaned = "\n".join(lines)
            f.write(cleaned + "\n\n")
    size_mb = os.path.getsize(filepath) / 1024**2
    return size_mb


def get_next_input_number():
    """找到下一個 input 編號"""
    import re
    pattern = re.compile(r"^input(\d+)\.txt$")
    nums = [0]
    for fname in os.listdir("."):
        m = pattern.match(fname)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1


def process_category(category, remote_files, output_num, force=False):
    """處理一個分類：下載所有檔案 → 寫入一個 input*.txt"""
    output_path = f"input{output_num}.txt"
    temp_path = output_path + ".tmp"

    # 檢查是否已存在且不需重做
    if os.path.exists(output_path) and not force:
        size = os.path.getsize(output_path) / 1024**2
        logger.info(f"{category}: 已存在 {output_path} ({size:.0f}MB)，跳過")
        return output_num + 1

    logger.info(f"{category}: 開始下載 {len(remote_files)} 個檔案...")
    all_texts = []
    total_files = len(remote_files)

    for i, remote_path in enumerate(remote_files):
        t0 = time.time()
        try:
            texts = download_and_extract_one(category, remote_path)
            all_texts.extend(texts)
            dt = time.time() - t0
            logger.info(f"  [{i+1}/{total_files}] {remote_path.split('/')[-1][:50]} → {len(texts):,} 條, {dt:.1f}s")
        except Exception as e:
            logger.warning(f"  [{i+1}/{total_files}] 失敗: {e}")
            continue

    logger.info(f"{category}: 共 {len(all_texts):,} 條，寫入 {output_path}...")
    size_mb = write_input_file(all_texts, temp_path)
    os.rename(temp_path, output_path)
    logger.info(f"{category}: 完成 → {output_path} ({size_mb:.0f}MB)")
    return output_num + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下載 MNBVC 補充訓練資料")
    parser.add_argument("--categories", nargs="+", choices=["zhihu", "chatgpt", "forum", "all"],
                        default=["all"], help="要下載的分類")
    parser.add_argument("--force", action="store_true", help="重新下載已存在的檔案")
    args = parser.parse_args()

    categories_to_download = ["zhihu", "chatgpt", "forum"] if "all" in args.categories else args.categories

    remote_files = get_category_files()
    next_num = get_next_input_number()

    logger.info(f"將從 input{next_num}.txt 開始寫入")
    logger.info(f"預計下載: {categories_to_download}")

    for cat in categories_to_download:
        if cat not in remote_files or not remote_files[cat]:
            logger.warning(f"{cat}: 無檔案可下載，跳過")
            continue
        next_num = process_category(cat, remote_files[cat], next_num, force=args.force)

    # 總計資訊
    total_inputs = sum(1 for f in os.listdir(".") if f.startswith("input") and f.endswith(".txt"))
    total_size = sum(os.path.getsize(f) for f in os.listdir(".") if f.startswith("input") and f.endswith(".txt"))
    logger.info(f"所有下載完成！共 {total_inputs} 個 input 檔案，總計 {total_size/1024**3:.2f}GB")
    logger.info("現在可以執行 python3 prepare_data.py --jobs 4 來 tokenize")

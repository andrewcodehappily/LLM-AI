"""
下載中文 Wikipedia（官方 wikimedia 版本）並存成 input2.txt
"""
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_wiki")

logger.info("下載中文 Wikipedia（wikimedia/wikipedia 20231101.zh）...")
# 改用官方 wikimedia 版本
ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", trust_remote_code=True)

logger.info(f"共 {len(ds)} 篇文章，開始寫入 input2.txt...")
with open("input2.txt", "w", encoding="utf-8") as f:
    for i, example in enumerate(ds):
        text = example.get("text", "")
        f.write(text + "\n")
        if (i + 1) % 10000 == 0:
            logger.info(f"  已處理 {i+1}/{len(ds)} 篇")

logger.info("完成！中文 Wikipedia 已存到 input2.txt")

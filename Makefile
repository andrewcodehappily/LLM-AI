.PHONY: setup download-wiki download-fineweb prepare-data train clean

# ─── 環境 ─────────────────────────────────────────────

setup:
	pip install -e .

# ─── 資料下載 ─────────────────────────────────────────

download-wiki:
	python download_wiki.py

download-fineweb:
	python download_fineweb_edu.py --num-files 10

download-all: download-wiki download-fineweb

# ─── 資料預處理 ────────────────────────────────────────

prepare-data:
	python prepare_data.py

# ─── 訓練 ──────────────────────────────────────────────

train:
	python train.py

# ─── 清理 ──────────────────────────────────────────────

clean-shards:
	rm -rf edu_fineweb10B/

clean-checkpoints:
	rm -rf checkpoints/

clean-logs:
	rm -f logs.txt

clean-inputs:
	rm -f input*.txt input*.txt.bak

clean-all: clean-shards clean-checkpoints clean-logs clean-inputs
	rm -rf *.parquet __pycache__/ .venv/

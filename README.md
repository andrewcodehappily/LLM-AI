# GPT-2 Medium 中文語言模型 (MLX)

使用 Apple MLX 框架實作的 GPT-2 Medium（355M 參數）中文語言模型，基於 [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) 與 [MLX GPT-2](https://github.com/yuchaoran2011/gpt2-mlx)。

## 特點

- **GPT-2 Medium 架構（355M 參數）：** 24 層 Transformer、16 注意力頭、1024 維度
- **Qwen 2.5 Tokenizer（151,680 vocab）：** 中文壓縮效率是 GPT-2 tokenizer 的 3-4 倍，實際 context window 達 ~2000 字
- **中文高品質資料訓練：** Chinese Wikipedia + Chinese FineWeb-Edu，共 32.3 億 tokens
- **Chinchilla 最優訓練：** 355M × 20 = 71 億 tokens 訓練目標，梯度累積 64 步
- **MLX + Apple Silicon 優化：** 支援 `mx.compile` 前向/反向傳播編譯、bf16 混合精度

## 環境需求

- macOS（Apple Silicon，建議 M3/M4 系列）
- Python 3.12+
- MLX

## 安裝

```bash
pip install -r requirements.txt
```

## 資料準備

本專案使用兩個資料來源：

### 1. 中文 Wikipedia（約 7.7 億 tokens）

```bash
python download_wiki.py
```

### 2. Chinese FineWeb-Edu（高品質教育資料）

從 HuggingFace [`opencsg/chinese-fineweb-edu`](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu) 下載：

```bash
python download_fineweb_edu.py --num-files 10
```

### 3. 合併並轉換為 shard

```bash
python prepare_data.py
```

這會將所有 `inputX.txt` 合併，用 Qwen 2.5 tokenizer 編碼，並輸出至 `edu_fineweb10B/` 目錄（每個 shard 100 萬 tokens）。

## 訓練

```bash
python train.py
```

### 超參數

| 參數 | 數值 |
|------|------|
| 模型參數量 | 355M |
| Batch size (B) | 8 |
| 序列長度 (T) | 1024 |
| 總 batch size (tokens) | 524,288 |
| 梯度累積步數 | 64 |
| 最大學習率 | 6e-4 |
| 最小學習率 | 6e-5 |
| Warmup 步數 | 715 |
| 最大步數 | 13,542 |
| 權重衰減 | 0.01 |
| AdamW betas | (0.9, 0.95) |
| 梯度裁剪 | 1.0 |
| 資料類型 | bf16 |

訓練會在每個 checkpoint（每 1000 步）和 validation（每 100 步）時自動儲存，並顯示 validation loss 與生成樣例。

## 專案結構

```
├── gpt2.py              # GPT-2 Medium 模型架構
├── train.py             # 訓練腳本
├── inference.py         # 文字生成
├── dataloader.py        # 資料載入與 batching
├── checkpoint.py        # 檢查點儲存/讀取
├── prepare_data.py      # 資料預處理（tokenize + shard）
├── download_fineweb_edu.py  # Chinese FineWeb-Edu 下載器
├── download_wiki.py     # 中文 Wikipedia 下載器
└── run_prepare.sh       # 一鍵執行資料準備
```

## 授權

MIT

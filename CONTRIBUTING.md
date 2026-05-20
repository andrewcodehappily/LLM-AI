# 貢獻指南

歡迎參與本專案的開發！以下是一些協作規範，讓大家可以順利合作。

## 環境需求

- **macOS（Apple Silicon）**：本專案使用 MLX 框架，僅支援 Apple Silicon Mac
- **Python >= 3.12**
- **建議 64GB+ 記憶體**，訓練時約佔用 20-25GB RAM

## 開發環境設定

```bash
# Clone 專案
git clone https://github.com/andrewcodehappily/LLM-AI.git
cd LLM-AI

# 建議建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -e .
```

## 分支策略

- `main`：穩定版本，保持可訓練狀態
- `feat/*`：新功能開發
- `fix/*`：錯誤修復
- `experiment/*`：實驗性修改（超參數調整、架構變更等）

請勿直接推送到 `main`，所有變更應透過 Pull Request 合併。

## Pull Request 流程

1. 從 `main` 建立你的功能分支
2. 實作你的變更
3. 確認程式碼可以執行：
   ```bash
   # 確認模型可以初始化
   python -c "from gpt2 import GPT, GPTConfig; GPT(GPTConfig())"
   ```
4. 提交 PR 並說明變更內容與動機
5. 等待 review 後合併

## 程式碼風格

- 使用 [Ruff](https://github.com/astral-sh/ruff) 格式化，行長度 88
- 型別提示：函式參數與回傳值請加 type hints
- 註解：中文註解，解釋「為什麼」而不是「做什麼」
- import 順序：標準函式庫 → 第三方套件 → 本地模組

## 常見檔案說明

| 檔案 | 用途 |
|------|------|
| `gpt2.py` | 模型架構 — 修改網路結構時動這裡 |
| `train.py` | 訓練腳本 — 超參數、訓練邏輯 |
| `inference.py` | 生成文字 |
| `dataloader.py` | 資料載入 — shard 讀取 |
| `prepare_data.py` | 資料預處理 |
| `download_*.py` | 資料下載器 |

## 注意事項

- **不要提交大檔案**：`.gitignore` 已排除資料、checkpoint、虛擬環境等
- **不要提交個人超參數實驗**：實驗性的 config 請放在自己的分支
- **訓練前先確認 checkpoint 目錄是空的**，避免從錯誤的 checkpoint 恢復

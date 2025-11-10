# 基於深度學習之氣胸X光影像分割
(SIIM-ACR Pneumothorax Segmentation Challenge)

[![Kaggle](https://img.shields.io/badge/Kaggle-SIIM--ACR%20Pneumothorax-blue.svg)](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

## 📖 專案簡介

本專案旨在解決 [SIIM-ACR Pneumothorax Segmentation Kaggle 競賽](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) 的挑戰：建立一個能自動且精確分割出X光片中氣胸區域的AI模型。

我們利用深度學習影像分割技術，特別是 U-Net 及其變體 (U-Net++)，來輔助醫師進行快速且客觀的診斷。

### 團隊成員
* B1228005 胡樂麒
* B1228010 李怡萱
* B1228011 劉姮廷
* B1228021 邱庭俞
* B1228039 蔡勇濱

*本 Repository 是課程專案的開源實作，詳細方法請參閱 [reports/proposal.pdf](reports/proposal.pdf)。*

---

## 🚀 快速開始

### 1. 環境設置

```bash
# 1. 複製本專案 (請替換成您自己的 repo 連結)
git clone [https://github.com/](https://github.com/)[Your-Username]/SIIM-ACR-Pneumothorax-Segmentation.git
cd SIIM-ACR-Pneumothorax-Segmentation
```
```
# 2. (建議) 建立並啟動虛擬環境
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```
```
# 3. 安裝所需套件
pip install -r requirements.txt
```
### 2\. 資料準備

本專案使用 Kaggle 上的 SIIM-ACR Pneumothorax Segmentation 數據集。

我們推薦使用 Kaggle 官方 API 來下載數據：

1.  **安裝 Kaggle API** (如果您尚未安裝):

    ```bash
    pip install kaggle
    ```

    (您可能需要先在您的 Kaggle 帳戶「Settings」中建立 API Token 並將其放置在 `~/.kaggle/kaggle.json`)

2.  **下載數據集**：
    請在專案根目錄 (與 `src/` 同層) 執行以下指令，將數據下載到 `data/` 資料夾中：

    ```bash
    kaggle datasets download -d jesperdramsch/siim-acr-pneumothorax-segmentation-data -p data/
    ```

3.  **解壓縮數據**：
    下載完成後，您會在 `data/` 中找到一個 `siim-acr-pneumothorax-segmentation-data.zip` 檔案。請將其解壓縮。

    ```bash
    # (macOS/Linux)
    unzip data/siim-acr-pneumothorax-segmentation-data.zip -d data/

    # (Windows - 可能需要手動解壓縮或使用其他工具)
    ```

4.  **確認結構**：
    解壓縮完成後，您的 `data/` 資料夾結構應如下所示。`src/train.py` 腳本將會從此處讀取資料：

    ```
    SIIM-ACR-Pneumothorax-Segmentation/
    ├── data/
    │   ├── train-rle.csv
    │   ├── pneumothorax/
    │   │   ├── dicom-images-train/
    │   │   │   └── ... (dicom 檔案)
    │   │   └── ...
    │   └── siim-acr-pneumothorax-segmentation-data.zip (可選，可刪除)
    ├── src/
    │   └── train.py
    └── README.md
    ```

### 3\. 模型訓練

我們使用 `src/train.py` 腳本來執行 K-fold 交叉驗證訓練。

```bash
# 執行 U-Net++ (預設) 訓練，共 15 個 epochs，batch size 為 16
python src/train.py --model unetpp --epochs 15 --batch_size 16

# 執行 U-Net (baseline) 訓練
python src/train.py --model unet --epochs 15 --batch_size 16
```

您可以使用以下參數自定義訓練過程：

  * `--model`: 要訓練的模型 (`unet` 或 `unetpp`，預設: `unetpp`)
  * `--epochs`: 訓練的 epoch 數量 (預設: `15`)
  * `--batch_size`: 批次大小 (預設: `32`)
  * `--learning_rate`: 學習率 (預設: `1e-4`)
  * `--num_folds`: K-fold 的折數 (預設: `5`)
  * `--data_dir`: 資料來源路徑 (預設: `../data`)
  * `--model_dir`: 模型權重儲存路徑 (預設: `../checkpoints`)

訓練日誌 (Logs) 將儲存在 `lightning_logs/` 中，您可以使用 TensorBoard 查看。
訓練好的模型權重 (`.ckpt`) 將儲存在 `checkpoints/` 中。

-----

## 📁 Repository 結構
```
SIIM-ACR-Pneumothorax-Segmentation/
├── .gitignore               # 忽略 .ckpt, 數據集等
├── README.md                # 專案說明 (您正在閱讀)
├── requirements.txt         # Python 依賴套件
├── data/                    # (用 .gitignore 忽略，存放 Kaggle 數據)
├── checkpoints/             # (用 .gitignore 忽略，存放訓練好的模型權重)
├── lightning_logs/          # (用 .gitignore 忽略，存放 TensorBoard 日誌)
├── notebooks/               # 存放 EDA 和實驗過程的 Jupyter Notebooks
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_UNet_Experiment.ipynb
│   └── 03_UNet++_Experiment.ipynb
├── reports/
│   └── proposal.pdf         # 專案計畫書
└── src/
    ├── __init__.py
    ├── dataset.py           # PyTorch Dataset/DataLoader, RLE 編解碼
    ├── models.py            # U-Net, U-Net++ (PyTorch Lightning Module)
    ├── metrics.py           # Dice / IoU 評估指標
    └── train.py             # K-fold 交叉驗證訓練主腳本
```
-----

## 📊 實驗結果

| 模型 | Encoder | 平均 Val IoU (5-fold) |
| :--- | :--- | :--- |
| U-Net | efficientnet-b0 | (待填入) |
| U-Net++ | efficientnet-b0 | (待填入) |

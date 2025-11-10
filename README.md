# 基於深度學習之氣胸X光影像分割
(SIIM-ACR Pneumothorax Segmentation Challenge)

[![Kaggle](https://img.shields.io/badge/Kaggle-SIIM--ACR%20Pneumothorax-blue.svg)](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)

## 📖 專案簡介

本專案旨在解決 [SIIM-ACR Pneumothorax Segmentation Kaggle 競賽](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) 的挑戰：建立一個能自動且精確分割出X光片中氣胸區域的AI模型。

我們利用深度學習影像分割技術，特別是 U-Net 及其變體 (U-Net++)，來輔助醫師進行快速且客觀的診斷。

### 團隊成員
* B1228005 胡樂麒
* B1228010 李怡萱
* B1228011 劉 廷
* B1228021 邱庭俞
* B1228039 蔡勇濱

*本 Repository 是課程專案的開源實作，詳細方法請參閱 [reports/proposal.pdf](reports/proposal.pdf)。*

---

## 🚀 快速開始

### 1. 環境設置

```bash
# 1. 複製本專案
git clone [https://github.com/](https://github.com/)[Your-Username]/SIIM-ACR-Pneumothorax-Segmentation.git
cd SIIM-ACR-Pneumothorax-Segmentation

# 2. (建議) 建立並啟動虛擬環境
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3. 安裝所需套件
pip install -r requirements.txt

### 2. 資料準備

本專案使用 Kaggle 上的 SIIM-ACR Pneumothorax Segmentation 數據集。由於數據檔案過大，請勿將其上傳至 GitHub。

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
    # (Windows/Linux/MacOS 可能有所不同)
    unzip data/siim-acr-pneumothorax-segmentation-data.zip -d data/
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

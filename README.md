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

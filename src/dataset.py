# src/dataset.py

import os
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# --- RLE 工具函數 (來自您的 .py 檔) ---

def rle2mask(rle, width, height):
    """
    將 RLE 字串轉換為二進位 mask。
    """
    if rle == ' -1' or rle == '-1' or (not isinstance(rle, str) and np.isnan(rle)):
        return np.zeros(width * height, dtype=np.uint8)

    mask = np.zeros(width * height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(height, width).T

def mask2rle(img, width, height):
    """
    將二進位 mask 轉換為 RLE 字串。
    """
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    if lastColor == 255:
        rle.append(str(runStart))
        rle.append(str(runLength))

    return " ".join(rle) if rle else "-1"

# --- 資料增強 (Augmentations) ---

def get_transforms():
    """
    定義訓練和驗證的影像轉換。
    """
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        ToTensorV2()
    ])
    
    return train_transform, val_transform

# --- PyTorch Dataset ---

class CustomDataset(Dataset):
    """
    客製化的 PyTorch Dataset 類別。
    (來自 siim_acr_pneumothorax_test_v2_0.py)
    """
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # 轉換為 3 通道 (模型需要)
        image = np.stack((image, image, image), axis=-1)  # H×W×3
        mask = np.expand_dims(mask, axis=-1)              # H×W×1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 將 mask 轉換為 [0, 1] 範圍並確保維度正確 (C, H, W)
        mask = mask.permute(2, 0, 1) / 255.0
        
        return image, mask

# --- 資料準備主函數 ---

def prepare_data(data_dir, n_splits=5, random_state=42):
    """
    讀取 DICOM 檔案、RLEs，並準備 K-fold 交叉驗證。
    """
    train_rle_path = os.path.join(data_dir, 'train-rle.csv')
    data_df = pd.read_csv(train_rle_path)
    data_df[' EncodedPixels'] = data_df[' EncodedPixels'].str.strip()
    
    # 獲取所有 DICOM 檔案路徑
    dicom_paths = glob(os.path.join(data_dir, 'pneumothorax/dicom-images-train/*/*/*.dcm'))
    
    # 建立 ImageId 到路徑的映射
    image_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in dicom_paths}

    # 預先讀取所有影像和 masks
    all_images = []
    all_masks = []
    all_image_ids = []
    
    print("Loading all images and masks into memory...")
    for img_id, rle_str in tqdm(data_df.values):
        if img_id in image_to_path:
            dcm_path = image_to_path[img_id]
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array
            
            # 調整大小 (來自您的 .py 檔)
            img = cv2.resize(img, (256, 256)) 
            mask = rle2mask(rle_str, 1024, 1024)
            mask = cv2.resize(mask, (256, 256))
            
            all_images.append(img)
            all_masks.append(mask)
            all_image_ids.append(img_id)

    all_images = np.array(all_images)
    all_masks = np.array(all_masks)
    
    print("Data loaded successfully.")

    # 根據 proposal，您使用了 K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(kf.split(all_images))
    
    return all_images, all_masks, folds

# src/models.py

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

# --- 評估指標與損失函數 (來自您的 .py 檔) ---

def iou_metric(pred, target, threshold=0.5, smooth=1e-6):
    """
    計算 IoU (Jaccard Index)。
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

class DiceLoss(nn.Module):
    """
    Dice Loss 損失函數。
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# --- U-Net Lightning Module (來自 unet.py) ---

class LitUnet(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # 使用 segmentation_models_pytorch (如您 .py 檔所示)
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None # Loss 函數中會處理 sigmoid
        )
        self.loss_fn = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        iou = iou_metric(torch.sigmoid(y_hat), y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 使用 AdamW (如您的 proposal 所述)
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

# --- U-Net++ Lightning Module (來自 unet++.py) ---

class LitUnetPlusPlus(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
        self.loss_fn = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        iou = iou_metric(torch.sigmoid(y_hat), y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

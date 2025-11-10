# src/train.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

# 從我們建立的 .py 檔案中導入
from dataset import prepare_data, CustomDataset, get_transforms
from models import LitUnet, LitUnetPlusPlus

# --- 設定超參數 ---
# (您可以透過 argparse 覆蓋這些預設值)
NUM_FOLDS = 5
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DATA_DIR = "../data" # 假設數據在 data/ 目錄下
MODEL_DIR = "../checkpoints" # 假設模型存在 checkpoints/ 目錄下

def main(args):
    
    # --- 1. 準備資料 ---
    all_images, all_masks, folds = prepare_data(
        data_dir=args.data_dir,
        n_splits=args.num_folds,
        random_state=42
    )
    train_transform, val_transform = get_transforms()

    fold_results = []

    # --- 2. K-Fold 交叉驗證迴圈 ---
    for fold in range(args.num_folds):
        print("\\n" + "="*30)
        print(f" STARTING FOLD {fold+1} / {args.num_folds} ")
        print("="*30)

        train_idx, val_idx = folds[fold]

        # 建立 Datasets
        # 注意：我們使用 Subset 來指向預先載入的 numpy 陣列
        full_dataset = CustomDataset(all_images, all_masks)
        
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        
        # 應用不同的 transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        # 建立 DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )

        # --- 3. 初始化模型 ---
        if args.model == 'unet':
            model_fold = LitUnet(learning_rate=args.learning_rate)
        elif args.model == 'unetpp':
            model_fold = LitUnetPlusPlus(learning_rate=args.learning_rate)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        print(f"Training model: {args.model}")

        # --- 4. 設定 Trainer (Callbacks) ---
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.model_dir, f"{args.model}/fold_{fold+1}"),
            filename=f"best-model-{{epoch:02d}}-{{val_iou:.4f}}",
            monitor="val_iou",
            mode="max",
            save_top_k=1,
            verbose=True
        )
        
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_iou",
            patience=5, # (來自您的 .py 檔)
            mode="max",
            verbose=True
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=pl.loggers.TensorBoardLogger("lightning_logs", name=f"{args.model}/fold_{fold+1}"),
            callbacks=[checkpoint_callback, early_stop_callback],
            enable_progress_bar=True
        )

        # --- 5. 開始訓練 ---
        print(f"Starting training for Fold {fold+1}...")
        trainer.fit(model_fold, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # 記錄最佳結果
        best_iou = trainer.callback_metrics.get("val_iou").item()
        best_model_path = checkpoint_callback.best_model_path
        
        fold_results.append({
            "fold": fold + 1, 
            "val_iou": best_iou, 
            "checkpoint": best_model_path
        })
        
        print(f"Fold {fold+1} finished. Best Val IoU: {best_iou:.4f}")
        print(f"Best model saved at: {best_model_path}")
        
        # (可選) 清理 GPU 記憶體
        del model_fold, trainer
        torch.cuda.empty_cache()


    # --- 6. 總結 K-Fold 結果 ---
    print("\\n" + "="*30)
    print(" K-FOLD CROSS-VALIDATION SUMMARY ")
    print("="*30)
    
    final_ious = [res['val_iou'] for res in fold_results]
    mean_iou = np.mean(final_ious)
    std_iou = np.std(final_ious)
    
    for res in fold_results:
        print(f"Fold {res['fold']}: Best Val IoU = {res['val_iou']:.4f}")
        
    print("-" * 30)
    print(f"Average Best Val IoU across {args.num_folds} folds: {mean_iou:.4f} ± {std_iou:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net/U-Net++ for Pneumothorax Segmentation")
    
    parser.add_argument('--model', type=str, default='unetpp', choices=['unet', 'unetpp'],
                        help='Model architecture to train (default: unetpp)')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Directory containing the data (default: ../data)')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR,
                        help='Directory to save checkpoints (default: ../checkpoints)')
    parser.add_argument('--num_folds', type=int, default=NUM_FOLDS,
                        help=f'Number of folds for K-fold CV (default: {NUM_FOLDS})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs per fold (default: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')

    args = parser.parse_args()
    main(args)

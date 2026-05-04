import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

import config
from dataset import get_train_val_loaders, get_test_loader
from utils import calculate_correct, build_model


# def mixup_data(x, y, alpha=1.0):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).to(x.device)
#
#     mixed_x = lam * x + (1 - lam) * x[index]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


def train(model_name='vgg16'):
    train_loader, val_loader = get_train_val_loaders()

    # -------------------- Device --------------------
    if config.DEVICE == 'cuda':
        device = config.DEVICE
        gpu = torch.cuda.get_device_name(device)
        tqdm.write(f"💻 Device: {gpu}")
    else:
        tqdm.write("💻 Device: CPU")

    # -------------------- Model --------------------
    model = build_model(model_name)

    # -------------------- Loss & Optimizer --------------------
    criterion = nn.CrossEntropyLoss()

    is_vit = 'vit' in model_name.lower()
    if is_vit:
        vit_lr = 3e-4
        optimizer = optim.AdamW(model.parameters(), lr=vit_lr, weight_decay=0.05)
        tqdm.write(f"🔧 Optimizer: AdamW | Initial LR: {vit_lr}")
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM,
                              weight_decay=config.WEIGHT_DECAY, nesterov=True)

    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    # -------------------- AMP --------------------
    use_amp = config.DEVICE == 'cuda'
    scaler = GradScaler() if use_amp else None

    # -------------------- Resume --------------------
    best_val_acc = 0.0
    start_epoch = 0
    save_path = Path(config.save_folder) / model_name
    save_path.mkdir(parents=True, exist_ok=True)
    history_path = save_path / f'history_{model_name}.pkl'
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    latest_checkpoint = save_path / f"latest_checkpoint_{model_name}.pth"
    if latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        if history_path.exists():
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            tqdm.write(f"Loaded history: {len(history['train_loss'])} epochs recorded")
        tqdm.write(f"Resumed from epoch {start_epoch + 1}")

    tqdm.write(f"🧠 Model: {model_name}")
    tqdm.write("🚀 Start training...")
    start_time = time.time()

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # -------------------- Train --------------------
        model.train()
        train_loss = train_correct = 0.0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            batch_size = imgs.shape[0]
            train_loss += loss.item() * batch_size
            train_correct += calculate_correct(outputs, labels)

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })

        # -------------------- Validation --------------------
        model.eval()
        val_loss = val_correct = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{config.NUM_EPOCHS}"):
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                batch_size = imgs.shape[0]
                val_loss += loss.item() * batch_size
                val_correct += calculate_correct(outputs, labels)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_acc = val_correct / len(val_loader.dataset)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        scheduler.step()

        tqdm.write(
            f"[epoch {epoch + 1}] "
            f"train loss: {avg_train_loss:.4f} | train acc: {avg_train_acc:.4f} | "
            f"val loss: {avg_val_loss:.4f} | val acc: {avg_val_acc:.4f}"
        )

        # save the best model
        if avg_val_acc >= best_val_acc:
            best_val_acc = avg_val_acc
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc
            }
            torch.save(best_checkpoint, save_path / f"best_model_{model_name}.pth")
            tqdm.write(f'New best model saved with val acc: {best_val_acc:.4f}')

        # always save the latest checkpoint for resuming
        latest = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }
        if scaler is not None:
            latest['scaler'] = scaler.state_dict()
        torch.save(latest, save_path / f"latest_checkpoint_{model_name}.pth")

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 3600
    tqdm.write(f"\n🎉 Training complete!")
    tqdm.write(f"✅ Training completed in {execution_time_minutes:.2f} hours")
    tqdm.write(f"💾 Best model saved with acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train('resnext50')

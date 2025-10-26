#!/usr/bin/env python3
"""
CRNN OCR Training Script for Algerian License Plates
Architecture: CNN (feature extraction) + LSTM (sequence) + CTC (decoding)
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import cv2
import albumentations as A

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============= CONFIGURATION =============
BASE_DIR = '/home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV'
DATA_DIR = os.path.join(BASE_DIR, 'ocr_training_data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Model parameters
IMG_HEIGHT = 64
IMG_WIDTH = 200
NUM_CLASSES = 11  # 0-9 digits + CTC blank
SEQUENCE_LENGTH = 11  # License plate length

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🔧 Using device: {DEVICE}")

# ============= DATA LOADING =============

def load_splits():
    """Load train/val/test splits from dataset_info.json"""
    # For now, we'll scan the directories created by mix_datasets_local.py
    # The script saved paths, we need to recreate them
    
    synthetic_dir = os.path.join(BASE_DIR, 'synthetic_plates')
    real_dir = os.path.join(BASE_DIR, 'recognition')
    
    def scan_plates(directory):
        images, labels = [], []
        for img_path in Path(directory).rglob('*.jpg'):
            label = img_path.stem.split('_')[0].zfill(11)
            if len(label) >= 10 and len(label) <= 11 and label.isdigit():
                images.append(str(img_path))
                labels.append(label)
        return images, labels
    
    synth_imgs, synth_labs = scan_plates(synthetic_dir)
    real_imgs, real_labs = scan_plates(real_dir)
    
    # Mix and split (same as mix_datasets_local.py)
    synth_needed = int(len(real_imgs) * 0.8 / 0.2)
    synth_needed = min(synth_needed, len(synth_imgs))
    
    if synth_needed < len(synth_imgs):
        indices = random.sample(range(len(synth_imgs)), synth_needed)
        synth_imgs = [synth_imgs[i] for i in indices]
        synth_labs = [synth_labs[i] for i in indices]
    
    all_imgs = synth_imgs + real_imgs
    all_labs = synth_labs + real_labs
    all_srcs = ['synthetic'] * len(synth_imgs) + ['real'] * len(real_imgs)
    
    combined = list(zip(all_imgs, all_labs, all_srcs))
    random.shuffle(combined)
    all_imgs, all_labs, all_srcs = zip(*combined)
    
    total = len(all_imgs)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    
    return {
        'train': (list(all_imgs[:train_end]), list(all_labs[:train_end]), list(all_srcs[:train_end])),
        'val': (list(all_imgs[train_end:val_end]), list(all_labs[train_end:val_end]), list(all_srcs[train_end:val_end])),
        'test': (list(all_imgs[val_end:]), list(all_labs[val_end:]), list(all_srcs[val_end:]))
    }


# ============= AUGMENTATION =============

heavy_aug = A.Compose([
    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
    A.Perspective(scale=(0.02, 0.05), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.6),
    A.GaussNoise(p=0.5),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

medium_aug = A.Compose([
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

no_aug = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])


# ============= DATASET =============

class LicensePlateDataset(Dataset):
    def __init__(self, image_paths, labels, sources, augmentation=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.sources = sources
        self.augmentation = augmentation
        self.is_training = is_training
        
        self.char_to_idx = {str(i): i for i in range(10)}
        self.idx_to_char = {i: str(i) for i in range(10)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Augmentation
        if self.is_training and self.augmentation:
            source = self.sources[idx]
            if source == 'real':
                img = heavy_aug(image=img)['image']
            else:
                img = medium_aug(image=img)['image']
        else:
            img = no_aug(image=img)['image']
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        # Encode label
        label = self.labels[idx]
        encoded = [self.char_to_idx[c] for c in label]
        
        return img, torch.LongTensor(encoded), len(encoded)


def collate_fn(batch):
    """Custom collate for variable length sequences"""
    images, labels, label_lengths = zip(*batch)
    
    images = torch.stack(images, 0)
    labels = torch.cat(labels, 0)
    label_lengths = torch.LongTensor(label_lengths)
    
    return images, labels, label_lengths


# ============= CRNN MODEL =============

class CRNN(nn.Module):
    """CRNN: CNN + RNN + CTC for license plate OCR"""
    
    def __init__(self, img_height=64, img_width=200, num_classes=11, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN Backbone (Feature Extractor)
        self.cnn = nn.Sequential(
            # Conv Block 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
            
            # Conv Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
            
            # Conv Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv Block 4: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W/4
            
            # Conv Block 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Conv Block 6: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16, W/4
        )
        
        # Calculate CNN output width
        # Input: (64, 200) -> After maxpool: (4, 50)
        self.cnn_output_height = img_height // 16  # Should be 4
        self.cnn_output_width = img_width // 4     # Should be 50
        
        # LSTM (Sequence Modeling)
        self.rnn_input_size = 512 * self.cnn_output_height
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully Connected (Output)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # (B, 512, H', W')
        
        # Reshape for RNN: (B, C, H, W) -> (B, W, C*H)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)  # (B, W, C, H)
        conv = conv.reshape(b, w, c * h)  # (B, W, C*H)
        
        # RNN
        rnn_out, _ = self.rnn(conv)  # (B, W, hidden*2)
        
        # FC
        output = self.fc(rnn_out)  # (B, W, num_classes)
        
        # For CTC: (W, B, num_classes)
        output = output.permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)
        
        return output


# ============= TRAINING =============

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, label_lengths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)  # (T, B, num_classes)
        
        # CTC expects: (T, B, C), targets, input_lengths, target_lengths
        T, B, _ = outputs.size()
        input_lengths = torch.full((B,), T, dtype=torch.long)
        
        # Loss
        loss = criterion(outputs, labels, input_lengths, label_lengths)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0
    correct_sequences = 0
    total_sequences = 0
    
    with torch.no_grad():
        for images, labels, label_lengths in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            T, B, _ = outputs.size()
            input_lengths = torch.full((B,), T, dtype=torch.long)
            
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            total_loss += loss.item()
            
            # Decode predictions
            _, preds = outputs.max(2)  # (T, B)
            preds = preds.transpose(1, 0)  # (B, T)
            
            # Calculate accuracy
            label_offset = 0
            for i, length in enumerate(label_lengths):
                pred_seq = preds[i].cpu().numpy()
                target_seq = labels[label_offset:label_offset + length].cpu().numpy()
                label_offset += length
                
                # Remove blanks and duplicates (greedy CTC decode)
                pred_decoded = []
                prev = -1
                for p in pred_seq:
                    if p != 10 and p != prev:  # 10 is blank
                        pred_decoded.append(p)
                    prev = p
                
                # Compare
                pred_decoded = pred_decoded[:len(target_seq)]
                
                correct_chars += sum(1 for p, t in zip(pred_decoded, target_seq) if p == t)
                total_chars += len(target_seq)
                
                if len(pred_decoded) == len(target_seq) and all(p == t for p, t in zip(pred_decoded, target_seq)):
                    correct_sequences += 1
                total_sequences += 1
    
    avg_loss = total_loss / len(dataloader)
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    seq_acc = correct_sequences / total_sequences if total_sequences > 0 else 0
    
    return avg_loss, char_acc, seq_acc


def main():
    print("="*60)
    print("🚀 CRNN OCR TRAINING")
    print("="*60)
    
    # Load data
    print("\n📂 Loading datasets...")
    splits = load_splits()
    
    train_dataset = LicensePlateDataset(
        *splits['train'],
        augmentation=True,
        is_training=True
    )
    val_dataset = LicensePlateDataset(
        *splits['val'],
        augmentation=False,
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    
    # Create model
    print("\n🏗️  Building CRNN model...")
    model = CRNN(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_classes=NUM_CLASSES,
        hidden_size=256
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Loss & Optimizer
    criterion = nn.CTCLoss(blank=10, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\n🏋️  Starting training...")
    best_seq_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_char_acc': [], 'val_seq_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, char_acc, seq_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_char_acc'].append(char_acc)
        history['val_seq_acc'].append(seq_acc)
        
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Char Acc:   {char_acc*100:.2f}%")
        print(f"   Seq Acc:    {seq_acc*100:.2f}%")
        
        # Save best model
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_seq_acc': best_seq_acc,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"   ✅ New best model saved! (Seq Acc: {best_seq_acc*100:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save history
    with open(os.path.join(LOGS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Sequence Accuracy: {best_seq_acc*100:.2f}%")
    print(f"Model saved to: {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()

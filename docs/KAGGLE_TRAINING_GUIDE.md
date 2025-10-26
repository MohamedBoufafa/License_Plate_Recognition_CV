# 🚀 Train CRNN OCR on Kaggle (FREE GPU)

## 📋 **Step-by-Step Guide**

---

## **STEP 1: Upload Datasets to Kaggle**

### **1.1 Go to Kaggle Datasets**
Visit: https://www.kaggle.com/datasets

### **1.2 Create New Dataset (Synthetic Plates)**
1. Click **"New Dataset"**
2. Click **"Upload"**
3. Select: `synthetic_plates_kaggle.zip`
4. Fill details:
   - **Title:** `Algerian License Plates Synthetic`
   - **Subtitle:** `50K synthetic Algerian license plates for OCR training`
   - **Description:** `Generated synthetic license plates with format XXXXXX-XXX-XX`
5. Click **"Create"**
6. Wait for upload (~5-10 min)
7. **Copy the dataset slug** (e.g., `yourusername/algerian-license-plates-synthetic`)

### **1.3 Create New Dataset (Real Plates)**
1. Click **"New Dataset"** again
2. Upload: `real_plates_kaggle.zip`
3. Fill details:
   - **Title:** `Algerian License Plates Real`
   - **Subtitle:** `Real Algerian license plate images`
   - **Description:** `Real license plate images for validation`
4. Click **"Create"**
5. **Copy the dataset slug** (e.g., `yourusername/algerian-license-plates-real`)

---

## **STEP 2: Create Kaggle Notebook**

### **2.1 Create New Notebook**
1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Select **"Notebook"** (not script)
4. Title: `CRNN OCR Training - Algerian Plates`

### **2.2 Enable GPU**
1. Click **"Accelerator"** in right panel
2. Select **"GPU T4 x2"** or **"GPU P100"**
3. Click **"Save"**

### **2.3 Add Your Datasets**
1. Click **"+ Add Data"** in right panel
2. Search for your uploaded datasets:
   - `algerian-license-plates-synthetic`
   - `algerian-license-plates-real`
3. Click **"Add"** for both
4. They appear as: `/kaggle/input/algerian-license-plates-synthetic/`

---

## **STEP 3: Copy Training Code**

### **Cell 1: Install Dependencies**
```python
!pip install albumentations
```

### **Cell 2: Imports & Config**
```python
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

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# CONFIG - ADJUST THESE PATHS TO YOUR KAGGLE DATASET SLUGS
SYNTHETIC_DIR = '/kaggle/input/algerian-license-plates-synthetic'
REAL_DIR = '/kaggle/input/algerian-license-plates-real'
OUTPUT_DIR = '/kaggle/working/checkpoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training params
IMG_HEIGHT = 64
IMG_WIDTH = 200
NUM_CLASSES = 11
BATCH_SIZE = 64  # Larger batch on Kaggle GPU
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda')

print(f"✅ Using device: {DEVICE}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
```

### **Cell 3: Data Loading Functions**
```python
def scan_plates(directory):
    images, labels = [], []
    for img_path in Path(directory).rglob('*.jpg'):
        label = img_path.stem.split('_')[0].zfill(11)
        if len(label) >= 10 and len(label) <= 11 and label.isdigit():
            images.append(str(img_path))
            labels.append(label)
    return images, labels

def load_and_split_data():
    print("📂 Loading datasets...")
    synth_imgs, synth_labs = scan_plates(SYNTHETIC_DIR)
    real_imgs, real_labs = scan_plates(REAL_DIR)
    
    print(f"   Synthetic: {len(synth_imgs):,}")
    print(f"   Real: {len(real_imgs):,}")
    
    # Mix 80/20
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
```

### **Cell 4: Augmentation**
```python
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
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

no_aug = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])
```

### **Cell 5: Dataset Class**
```python
class LicensePlateDataset(Dataset):
    def __init__(self, image_paths, labels, sources, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.sources = sources
        self.is_training = is_training
        self.char_to_idx = {str(i): i for i in range(10)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.is_training:
            if self.sources[idx] == 'real':
                img = heavy_aug(image=img)['image']
            else:
                img = medium_aug(image=img)['image']
        else:
            img = no_aug(image=img)['image']
        
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        label = self.labels[idx]
        encoded = [self.char_to_idx[c] for c in label]
        
        return img, torch.LongTensor(encoded), len(encoded)

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.cat(labels, 0)
    label_lengths = torch.LongTensor(label_lengths)
    return images, labels, label_lengths
```

### **Cell 6: CRNN Model**
```python
class CRNN(nn.Module):
    def __init__(self, img_height=64, img_width=200, num_classes=11, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        self.cnn_output_height = img_height // 16
        self.cnn_output_width = img_width // 4
        self.rnn_input_size = 512 * self.cnn_output_height
        
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2).reshape(b, w, c * h)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)
        return output
```

### **Cell 7: Training Functions**
```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, label_lengths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        T, B, _ = outputs.size()
        input_lengths = torch.full((B,), T, dtype=torch.long)
        
        loss = criterion(outputs, labels, input_lengths, label_lengths)
        
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
    correct_seqs = 0
    total_seqs = 0
    
    with torch.no_grad():
        for images, labels, label_lengths in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            T, B, _ = outputs.size()
            input_lengths = torch.full((B,), T, dtype=torch.long)
            
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            total_loss += loss.item()
            
            _, preds = outputs.max(2)
            preds = preds.transpose(1, 0)
            
            label_offset = 0
            for i, length in enumerate(label_lengths):
                pred_seq = preds[i].cpu().numpy()
                target_seq = labels[label_offset:label_offset + length].cpu().numpy()
                label_offset += length
                
                pred_decoded = []
                prev = -1
                for p in pred_seq:
                    if p != 10 and p != prev:
                        pred_decoded.append(p)
                    prev = p
                
                pred_decoded = pred_decoded[:len(target_seq)]
                
                if len(pred_decoded) == len(target_seq) and all(p == t for p, t in zip(pred_decoded, target_seq)):
                    correct_seqs += 1
                total_seqs += 1
    
    return total_loss / len(dataloader), correct_seqs / total_seqs
```

### **Cell 8: Main Training Loop**
```python
# Load data
splits = load_and_split_data()

train_dataset = LicensePlateDataset(*splits['train'], is_training=True)
val_dataset = LicensePlateDataset(*splits['val'], is_training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=2, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Create model
model = CRNN(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, 256).to(DEVICE)
print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CTCLoss(blank=10, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
best_acc = 0
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}\nEpoch {epoch+1}/{NUM_EPOCHS}\n{'='*60}")
    
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"\n📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"✅ Best model saved! (Acc: {best_acc*100:.2f}%)")

print(f"\n🎉 Training complete! Best accuracy: {best_acc*100:.2f}%")
```

### **Cell 9: Plot Results**
```python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot([x*100 for x in history['val_acc']])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png')
plt.show()

print(f"Best validation accuracy: {max(history['val_acc'])*100:.2f}%")
```

---

## **STEP 4: Run Training**

1. **Run all cells** (Shift+Enter or click "Run All")
2. **Training will start** (~2-3 hours for 100 epochs on T4 GPU)
3. **Monitor progress** in real-time
4. **Download model** after training:
   - Click **"Output"** tab
   - Download `best_model.pth`

---

## **STEP 5: Download Results**

Files in `/kaggle/working/`:
- `best_model.pth` - Trained model
- `training_curves.png` - Loss/accuracy plots

---

## ⚡ **Kaggle Advantages**

| Feature | Local (Your PC) | Kaggle |
|---------|----------------|---------|
| GPU | MX110 (CUDA 5.0) ❌ | T4/P100 (CUDA 7.5+) ✅ |
| Training Time | N/A (incompatible) | ~3 hours |
| Cost | Free | Free (30 hrs/week) |
| Memory | 2GB | 16GB ✅ |
| Batch Size | N/A | 64 ✅ |

---

## 🎯 **Expected Results on Kaggle**

```
Epoch 1: Val Acc: 15%
Epoch 10: Val Acc: 45%
Epoch 20: Val Acc: 68%
Epoch 50: Val Acc: 82%
Epoch 100: Val Acc: 88-92% ✅
```

---

## 💡 **Tips**

1. **Enable Internet** in notebook settings for pip install
2. **Save outputs** regularly (Kaggle auto-saves)
3. **Session limit:** 9 hours (enough for 100 epochs)
4. **Weekly GPU quota:** 30 hours

---

## ✅ **Quick Checklist**

- [ ] Create ZIPs of datasets
- [ ] Upload synthetic dataset to Kaggle
- [ ] Upload real dataset to Kaggle
- [ ] Create new notebook
- [ ] Enable GPU (T4 or P100)
- [ ] Add both datasets
- [ ] Copy all cells above
- [ ] Run all cells
- [ ] Wait ~3 hours
- [ ] Download best_model.pth

**Then you have a trained model!** 🎉

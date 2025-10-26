# 🏠 LOCAL DATA MIXING - Quick Implementation

## Run this Python script locally

Save as `mix_datasets.py` and run:

```python
import os, random, numpy as np
from pathlib import Path
import cv2, torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# PATHS
BASE = '/home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV'
SYNTHETIC = os.path.join(BASE, 'synthetic_plates')
REAL = os.path.join(BASE, 'recognition')

# SCAN
def scan(dir):
    imgs, labs = [], []
    for p in Path(dir).rglob('*.jpg'):
        if len(p.stem) == 11 and p.stem.isdigit():
            imgs.append(str(p))
            labs.append(p.stem)
    return imgs, labs

synth_imgs, synth_labs = scan(SYNTHETIC)
real_imgs, real_labs = scan(REAL)
print(f"Synthetic: {len(synth_imgs)}, Real: {len(real_imgs)}")

# MIX (80/20)
needed = int(len(real_imgs) * 0.8 / 0.2)
synth_sample = random.sample(synth_imgs, min(needed, len(synth_imgs)))
all_imgs = synth_sample + real_imgs
all_labs = [synth_labs[synth_imgs.index(i)] for i in synth_sample] + real_labs
all_srcs = ['synth']*len(synth_sample) + ['real']*len(real_imgs)

# SHUFFLE
combined = list(zip(all_imgs, all_labs, all_srcs))
random.shuffle(combined)
all_imgs, all_labs, all_srcs = zip(*combined)

# SPLIT
total = len(all_imgs)
train_end = int(total * 0.7)
val_end = train_end + int(total * 0.15)

train_data = (list(all_imgs[:train_end]), list(all_labs[:train_end]), list(all_srcs[:train_end]))
val_data = (list(all_imgs[train_end:val_end]), list(all_labs[train_end:val_end]), list(all_srcs[train_end:val_end]))
test_data = (list(all_imgs[val_end:]), list(all_labs[val_end:]), list(all_srcs[val_end:]))

print(f"Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")

# AUGMENTATION
heavy_aug = A.Compose([
    A.Rotate(limit=15, p=0.7),
    A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
    A.GaussianBlur(blur_limit=7, p=0.6),
    A.Resize(64, 200)
])

medium_aug = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.6),
    A.Resize(64, 200)
])

no_aug = A.Compose([A.Resize(64, 200)])

print("✅ Data prepared! Create PyTorch Dataset next.")
```

## Next: Create Dataset class and use in training

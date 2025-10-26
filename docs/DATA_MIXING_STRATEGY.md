# 📊 DATA MIXING & AUGMENTATION STRATEGY

## 🎯 Overview

Now that you have:
- ✅ **50,000 synthetic plates** (generated)
- ✅ **~1,500 real plates** (from `recognition/` folder)

**Next step:** Combine them intelligently for OCR training!

---

## 🔀 MIXING STRATEGY (3 Phases)

### **Phase 1: Warm-up (Epochs 1-10)**
```
Synthetic: 95%
Real:       5%

Purpose: Learn basic digit shapes from clean synthetic data
Augmentation: Light (just resize + slight blur)
```

### **Phase 2: Mixed Training (Epochs 11-50) ⭐ START HERE**
```
Synthetic: 80%
Real:      20%

Purpose: Learn real-world variations while maintaining volume
Augmentation: HEAVY (rotation, perspective, lighting, noise)
```

### **Phase 3: Fine-tuning (Epochs 51-100)**
```
Synthetic: 50%
Real:      50%

Purpose: Perfect performance on real plates
Augmentation: Medium (focus on real-world conditions)
```

---

## 📐 WHY MIX DATASETS?

### **Synthetic Plates (Pros & Cons)**

✅ **Advantages:**
- Perfect labels (no errors)
- Unlimited quantity
- Controlled variations
- No overfitting to camera angles

❌ **Disadvantages:**
- May lack real-world "messiness"
- Font might not match exactly
- Missing real lighting conditions

### **Real Plates (Pros & Cons)**

✅ **Advantages:**
- True real-world conditions
- Actual camera quality
- Real lighting, angles, dirt
- Model learns target distribution

❌ **Disadvantages:**
- Limited quantity (~1,500)
- Possible label errors
- Imbalanced (may have duplicates/similar)
- Risk of overfitting

### **SOLUTION: Combine Both!**

```
80% Synthetic + 20% Real = BEST Results

Why?
• Volume from synthetic (prevents underf itting)
• Reality from real (prevents synthetic bias)
• Heavy augmentation bridges the gap
```

---

## 🎨 AUGMENTATION PIPELINE

### **For REAL Images (Heavy Augmentation)**

Apply ALL of these to compensate for small dataset:

```python
import albumentations as A

heavy_aug = A.Compose([
    # 1. GEOMETRIC (simulate camera angles)
    A.Rotate(limit=15, p=0.7),              # ±15° rotation
    A.Perspective(scale=0.05, p=0.5),        # Viewing angle
    A.ShiftScaleRotate(p=0.6),              # Combined transform
    
    # 2. LIGHTING (day/night, shadows)
    A.RandomBrightnessContrast(
        brightness_limit=0.3,                # ±30%
        contrast_limit=0.3, 
        p=0.8
    ),
    A.RandomGamma(gamma_limit=(70, 130), p=0.5),
    
    # 3. BLUR (motion, focus issues)
    A.OneOf([
        A.GaussianBlur(blur_limit=7, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.6),
    
    # 4. NOISE (sensor noise, compression)
    A.GaussNoise(var_limit=50, p=0.5),
    A.ImageCompression(quality_lower=70, p=0.4),
    
    # 5. FINAL RESIZE
    A.Resize(height=64, width=200),
])
```

### **For SYNTHETIC Images (Medium Augmentation)**

Apply lighter augmentation (already have some variations):

```python
medium_aug = A.Compose([
    A.Rotate(limit=10, p=0.5),               # Lighter rotation
    A.Perspective(scale=0.03, p=0.3),        # Slight perspective
    A.RandomBrightnessContrast(
        brightness_limit=0.2, 
        contrast_limit=0.2,
        p=0.6
    ),
    A.GaussianBlur(blur_limit=3, p=0.3),     # Slight blur
    A.Resize(height=64, width=200),
])
```

### **For VALIDATION/TEST (No Augmentation)**

```python
no_aug = A.Compose([
    A.Resize(height=64, width=200),  # Just resize!
])
```

---

## 📂 DIRECTORY STRUCTURE

### **Before Mixing:**
```
project/
├── synthetic_plates/          ← Generated
│   ├── train/ (40,000)
│   ├── validation/ (5,000)
│   └── test/ (5,000)
│
└── recognition/               ← Real
    ├── train/ (~1,000)
    ├── validation/ (~250)
    └── test/ (~250)
```

### **After Mixing (for training):**
```
ocr_training_data/
├── train/                     ← Mixed!
│   ├── synthetic: 32,000 (80%)
│   └── real: 8,000 (20%)
│   Total: 40,000
│
├── validation/                ← Mixed!
│   ├── synthetic: 4,000
│   └── real: 1,000
│   Total: 5,000
│
└── test/                      ← Mostly REAL!
    ├── synthetic: 500
    └── real: 4,500            ← Bias toward real for true evaluation
    Total: 5,000
```

---

## 🔧 IMPLEMENTATION STEPS

### **Step 1: Scan Both Datasets**

```python
def scan_dataset(directory):
    images, labels = [], []
    for img_path in Path(directory).glob('**/*.jpg'):
        label = img_path.stem  # Filename without extension
        if len(label) == 11 and label.isdigit():
            images.append(str(img_path))
            labels.append(label)
    return images, labels

synth_imgs, synth_labs = scan_dataset('synthetic_plates/')
real_imgs, real_labs = scan_dataset('recognition/')
```

### **Step 2: Calculate Mix Ratio**

```python
SYNTHETIC_RATIO = 0.80  # 80% synthetic

total_real = len(real_imgs)
total_synth_needed = int(total_real * SYNTHETIC_RATIO / (1 - SYNTHETIC_RATIO))

# Sample synthetic to match ratio
synth_sample = random.sample(synth_imgs, min(total_synth_needed, len(synth_imgs)))
```

### **Step 3: Combine and Shuffle**

```python
all_images = synth_sample + real_imgs
all_labels = synth_labels + real_labs
all_sources = ['synthetic'] * len(synth_sample) + ['real'] * len(real_imgs)

# Shuffle together
combined = list(zip(all_images, all_labels, all_sources))
random.shuffle(combined)
```

### **Step 4: Split Train/Val/Test**

```python
# 70% train, 15% val, 15% test
train_size = int(len(combined) * 0.70)
val_size = int(len(combined) * 0.15)

train_data = combined[:train_size]
val_data = combined[train_size:train_size+val_size]
test_data = combined[train_size+val_size:]
```

### **Step 5: Create PyTorch Dataset**

```python
class LicensePlateDataset(Dataset):
    def __init__(self, image_paths, labels, sources, augmentation=None):
        self.images = image_paths
        self.labels = labels
        self.sources = sources
        self.augmentation = augmentation
        
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation based on source
        if self.augmentation:
            if self.sources[idx] == 'real':
                aug = heavy_augmentation(image=img)['image']
            else:
                aug = medium_augmentation(image=img)['image']
        else:
            aug = no_augmentation(image=img)['image']
        
        # Convert to tensor
        img_tensor = torch.from_numpy(aug).permute(2, 0, 1).float() / 255.0
        
        # Encode label
        label_encoded = [int(c) for c in self.labels[idx]]
        
        return img_tensor, torch.LongTensor(label_encoded)
```

### **Step 6: Create DataLoaders**

```python
train_dataset = LicensePlateDataset(train_imgs, train_labs, train_srcs, augmentation=True)
val_dataset = LicensePlateDataset(val_imgs, val_labs, val_srcs, augmentation=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

---

## 📊 EXPECTED RESULTS

### **Training Dataset Distribution:**

| Source | Count | Percentage |
|--------|-------|------------|
| Synthetic | 32,000 | 80% |
| Real | 8,000 | 20% |
| **TOTAL** | **40,000** | **100%** |

### **Augmentation Application:**

| Source | Augmentation | Purpose |
|--------|--------------|---------|
| Real images | **Heavy** | Maximize variety from limited data |
| Synthetic images | **Medium** | Add realism, already have variations |
| Validation/Test | **None** | True performance evaluation |

---

## ✅ BENEFITS OF THIS APPROACH

1. **Volume:** 40,000 training images (vs 1,500 real only)
2. **Variety:** Heavy augmentation on real images
3. **Quality:** Synthetic provides clean learning signal
4. **Balance:** 80/20 ratio prevents synthetic bias
5. **Generalization:** Mix ensures model works on both

---

## 🎯 TRAINING PHASES TIMELINE

### **Phase 1 (Epochs 1-10): Warm-up**
- **Data:** 95% synthetic, 5% real
- **Goal:** Learn digit shapes
- **Expected Acc:** 70-80% on real validation

### **Phase 2 (Epochs 11-50): Main Training**
- **Data:** 80% synthetic, 20% real
- **Goal:** Learn real-world variations
- **Expected Acc:** 85-90% on real validation

### **Phase 3 (Epochs 51-100): Fine-tuning**
- **Data:** 50% synthetic, 50% real
- **Goal:** Perfect real-world accuracy
- **Expected Acc:** 90-95% on real test

---

## 💡 PRO TIPS

### **1. Keep Real Test Set Separate!**
```
NEVER train on your real test set!
Use only for final evaluation.
```

### **2. Monitor Both Sources**
```python
# During training, track separately:
- Loss on synthetic images
- Loss on real images
- Accuracy on real validation

If real >> synthetic → Add more real data augmentation
If synthetic >> real → Reduce synthetic ratio
```

###  **3. Visualize Augmented Samples**
```python
# Always check augmentation output:
- Are plates still readable?
- Too much blur/rotation?
- Adjust augmentation intensity
```

### **4. Progressive Strategy**
```
Week 1: Phase 1 (95% synthetic) → Learn basics
Week 2: Phase 2 (80% synthetic) → Learn variations
Week 3: Phase 3 (50% synthetic) → Perfect real performance
```

---

## 🚀 NEXT STEPS

1. ✅ **Synthetic plates generated** (50,000)
2. ⏭️ **Create mixing script** (see implementation above)
3. ⏭️ **Set up augmentation pipeline** (Albumentations)
4. ⏭️ **Create DataLoaders**
5. ⏭️ **Start OCR training** (CRNN + CTC Loss)

---

## 📝 QUICK REFERENCE

```python
# RECOMMENDED CONFIGURATION
SYNTHETIC_RATIO = 0.80          # 80% synthetic
BATCH_SIZE = 64                  # Batch size
TARGET_HEIGHT = 64               # Image height
TARGET_WIDTH = 200               # Image width (for 11 chars)

# SPLIT RATIOS
TRAIN_RATIO = 0.70               # 70% train
VAL_RATIO = 0.15                 # 15% validation
TEST_RATIO = 0.15                # 15% test

# AUGMENTATION
REAL_AUG = 'heavy'               # Heavy for real images
SYNTHETIC_AUG = 'medium'         # Medium for synthetic
VAL_AUG = 'none'                 # None for validation/test
```

---

**Ready to mix datasets and start training!** 🚀

See `ocr_data_preparation.ipynb` for implementation code.

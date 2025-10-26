# 🏠 RUN DATA MIXING LOCALLY

## ✅ Quick Start

### **Step 1: Install Dependencies**
```bash
pip install torch torchvision albumentations opencv-python matplotlib
```

### **Step 2: Run the Script**
```bash
cd /home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV
python3 mix_datasets_local.py
```

### **Step 3: Check Output**
```bash
ls ocr_training_data/
# Should see: dataset_info.json and sample_augmented.png
```

---

## 📊 What the Script Does

1. **Scans datasets**
   - Synthetic: `synthetic_plates/` (50,000 plates)
   - Real: `recognition/` (~1,500 plates)

2. **Mixes with 80/20 ratio**
   - 80% synthetic
   - 20% real
   - Total: ~7,500 images for training

3. **Splits into train/val/test**
   - Train: 70% (~5,250)
   - Val: 15% (~1,125)
   - Test: 15% (~1,125)

4. **Creates augmentation pipelines**
   - Heavy for real images
   - Medium for synthetic
   - None for validation/test

5. **Saves visualization**
   - Creates `sample_augmented.png` with 12 examples
   - Shows augmentation quality

6. **Saves dataset info**
   - `dataset_info.json` with all metadata

---

## 📂 Output Structure

```
ocr_training_data/
├── dataset_info.json          ← Metadata (splits, ratios, etc.)
└── sample_augmented.png       ← Visual verification
```

---

## 🎯 Expected Output

```
===========================================================
📊 LOCAL DATA MIXING FOR OCR TRAINING
===========================================================

🔍 STEP 1: Scanning datasets...

🔍 Scanning: synthetic_plates
   Found: 50,000 plates

🔍 Scanning: recognition
   Found: 1,458 plates

🔀 STEP 2: Mixing datasets (80/20)...

📊 Mixing datasets:
   Real:      1,458
   Synthetic: 5,832
   Total:     7,290
   Ratio:     80% synth, 20% real

✂️ STEP 3: Splitting into train/val/test...

✂️ Dataset splits:
   Train: 5,103 (synth: 4,082, real: 1,021)
   Val  : 1,093 (synth: 875, real: 218)
   Test : 1,094 (synth: 875, real: 219)

🎨 STEP 4: Creating augmentation pipelines...
   ✅ Heavy (real): Rotation, Perspective, Blur, Noise
   ✅ Medium (synth): Light rotation, brightness
   ✅ None (val/test): Just resize

👀 STEP 5: Visualizing samples...

✅ Saved sample visualization

💾 STEP 6: Saving dataset info...
✅ Dataset info saved

===========================================================
✅ DATA PREPARATION COMPLETE!
===========================================================

Dataset ready for training:
  📁 Location: ocr_training_data
  🖼️  Image size: 64×200
  📊 Train: 5,103 images
  📊 Val:   1,093 images
  📊 Test:  1,094 images

🚀 Next step: Build CRNN model and start training!
===========================================================
```

---

## 🔍 Verify Results

### **Check dataset info:**
```bash
cat ocr_training_data/dataset_info.json
```

### **View sample images:**
```bash
xdg-open ocr_training_data/sample_augmented.png
# or
eog ocr_training_data/sample_augmented.png
```

### **Python check:**
```python
import json

with open('ocr_training_data/dataset_info.json') as f:
    info = json.load(f)

print(f"Train: {info['splits']['train']['total']}")
print(f"Val: {info['splits']['val']['total']}")
print(f"Test: {info['splits']['test']['total']}")
```

---

## 🎨 What Augmentation Looks Like

**sample_augmented.png** shows 12 examples:
- **Blue titles** = Synthetic plates (medium aug)
- **Red titles** = Real plates (heavy aug)

**You should see:**
- ✅ Rotated plates
- ✅ Different brightness/contrast
- ✅ Blurred images
- ✅ Some noise

---

## ⚙️ Customize Settings

Edit the script to change:

```python
# Line 18-23: Paths
BASE_DIR = '/your/custom/path'

# Line 25: Mixing ratio
SYNTHETIC_RATIO = 0.80  # Change to 0.90 for 90% synthetic

# Line 26-27: Image size
TARGET_HEIGHT = 64
TARGET_WIDTH = 200

# Line 28: Batch size
BATCH_SIZE = 32
```

---

## 🐛 Troubleshooting

### **Error: No synthetic plates found**
```bash
# Check if synthetic_plates exists
ls -la synthetic_plates/

# Should have: train/, validation/, test/ folders
```

### **Error: No real plates found**
```bash
# Check if recognition exists
ls -la recognition/

# Should have .jpg files with 11-digit names
```

### **Error: Module not found**
```bash
# Install missing packages
pip install torch torchvision albumentations opencv-python matplotlib
```

### **Script runs but no images shown**
```bash
# Matplotlib might not display
# Check saved file instead:
ls -lh ocr_training_data/sample_augmented.png
```

---

## 🚀 Next Steps

After running this script:

1. ✅ **Data is ready** - Mixed and augmented
2. ⏭️ **Build CRNN model** - See `OCR_IMPLEMENTATION_PLAN.md`
3. ⏭️ **Create training script** - Use PyTorch
4. ⏭️ **Train on local GPU** - Or upload to Kaggle

---

## 💡 Pro Tips

1. **Run on small sample first:**
   ```python
   # In the script, limit datasets for testing:
   synth_imgs = synth_imgs[:1000]  # Test with 1k synthetic
   real_imgs = real_imgs[:200]      # Test with 200 real
   ```

2. **Check augmentation quality:**
   - Open `sample_augmented.png`
   - Make sure text is still readable
   - If too blurry → reduce augmentation intensity

3. **Save splits for later:**
   ```python
   # Add to script after split_dataset():
   with open('splits.json', 'w') as f:
       json.dump({
           'train': train_imgs,
           'val': val_imgs,
           'test': test_imgs
       }, f)
   ```

---

## ✅ Success Checklist

- [ ] Script runs without errors
- [ ] Output shows ~5,000 train images
- [ ] `dataset_info.json` created
- [ ] `sample_augmented.png` shows varied images
- [ ] Augmentation looks good (readable text)
- [ ] 80% synthetic, 20% real ratio confirmed

**If all checked → Ready for OCR training!** 🎯

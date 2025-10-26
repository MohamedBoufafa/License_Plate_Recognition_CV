# 🎨 Synthetic Plate Generator - Quick Guide

## ✅ What's Fixed

### 1. **Text Centering Issue - FIXED** ✅
- **Problem:** Text was not properly centered on plates
- **Solution:** Added proper bounding box adjustment to account for font anchor points
- **Result:** Text is now perfectly centered horizontally and vertically

### 2. **Desktop Download - ADDED** ✅
- **Location:** Plates now save to `~/Desktop/synthetic_plates/` by default
- **Easy access:** No need to hunt through project directories
- **ZIP option:** Added optional ZIP archive creation for easy transfer

---

## 🚀 How to Use

### **Step 1: Open Notebook**
```bash
cd /home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV
jupyter notebook synthetic_plate_generator.ipynb
```

### **Step 2: Run All Cells**
- Click `Cell` → `Run All`
- Or press `Shift + Enter` on each cell sequentially

### **Step 3: Find Your Plates**
Generated plates will be at:
```
~/Desktop/synthetic_plates/
├── train/          (40,000 images)
├── validation/     (5,000 images)
└── test/           (5,000 images)
```

---

## ⚙️ Configuration Options

### **Change Output Location**

Edit **Cell 3 (Configuration)**:

**Option A: Save to Desktop (DEFAULT)**
```python
DESKTOP_PATH = os.path.expanduser("~/Desktop")
BASE_DIR = os.path.join(DESKTOP_PATH, 'synthetic_plates')
```

**Option B: Save to Project Directory**
```python
BASE_DIR = 'synthetic_plates'  # Same folder as notebook
```

**Option C: Custom Path**
```python
BASE_DIR = '/path/to/your/custom/folder/synthetic_plates'
```

### **Adjust Dataset Size**

For testing (smaller dataset):
```python
NUM_TRAIN = 1000    # Instead of 40,000
NUM_VAL = 200       # Instead of 5,000
NUM_TEST = 200      # Instead of 5,000
```

For production (larger dataset):
```python
NUM_TRAIN = 100000  # More training data
NUM_VAL = 10000
NUM_TEST = 10000
```

---

## 📦 Create ZIP Archive

After generation, run the ZIP cell to create a compressed archive:

1. **Scroll to bottom** of notebook
2. **Find cell:** "OPTIONAL: Create ZIP Archive"
3. **Uncomment the line:**
   ```python
   zip_file = create_zip_archive()
   ```
4. **Run cell** - Creates `synthetic_plates_YYYYMMDD_HHMMSS.zip`

**ZIP will be saved to Desktop!**

---

## 🔍 Verify Centering

The preview cell (Cell 7) shows:
- **Top row:** Perfect centering (no variations)
- **Bottom row:** With variations (slight random offsets)

Both should look properly centered!

---

## 📊 Expected Output

### **File Structure:**
```
~/Desktop/synthetic_plates/
├── train/
│   ├── 00000211616.jpg
│   ├── 12345678901.jpg
│   └── ... (40,000 total)
├── validation/
│   └── ... (5,000 total)
└── test/
    └── ... (5,000 total)
```

### **Sizes:**
- Each image: ~10-15 KB
- Train folder: ~400-600 MB
- Total dataset: ~500-750 MB
- ZIP archive: ~300-450 MB (compressed)

### **Generation Time:**
- Train (40k): ~15-20 minutes
- Val (5k): ~2-3 minutes
- Test (5k): ~2-3 minutes
- **Total: ~25 minutes**

---

## 🎨 Plate Features

Each generated plate includes:

✅ **Format:** 11 digits grouped as `XXXXXX XXX XX`  
✅ **Backgrounds:** White (70%) & Yellow (30%)  
✅ **Centered text** with black color  
✅ **Border:** Black rectangle (3px width)  
✅ **Realistic variations:**
- Subtle noise texture
- Slight blur
- Brightness adjustments (±15%)
- Contrast adjustments (±15%)
- Small position offsets (±3px horizontal, ±2px vertical)

---

## 🐛 Troubleshooting

### **Problem: Text still looks off-center**
**Solution:** Check if font loaded correctly (Cell 5 output). Try installing:
```bash
sudo apt-get install fonts-dejavu fonts-liberation
```

### **Problem: "Permission denied" on Desktop**
**Solution:** Change output to project directory:
```python
BASE_DIR = 'synthetic_plates'
```

### **Problem: Notebook crashes during generation**
**Solution:** Reduce dataset size in Configuration:
```python
NUM_TRAIN = 5000  # Start smaller
```

### **Problem: Can't find Desktop folder**
**Solution:** Check path with:
```python
print(os.path.expanduser("~/Desktop"))
```

---

## ✅ Next Steps

After generation:

1. **Verify quality:** Check samples in last cell
2. **Test on real images:** Compare with your `recognition/` dataset
3. **Upload to Kaggle:**
   - Compress to ZIP
   - Upload as Kaggle dataset
   - Use in OCR training notebook
4. **Combine datasets:** Merge synthetic with real plates

---

## 📌 Tips

💡 **Start small:** Generate 1,000 plates first to test  
💡 **Check samples:** Always review the preview images  
💡 **Save to Desktop:** Easier to find and transfer  
💡 **Create ZIP:** For uploading to Kaggle/cloud storage  
💡 **Adjust variations:** Edit helper functions for more/less realism  

---

**Happy generating! 🚀**

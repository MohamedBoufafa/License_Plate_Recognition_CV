# 🚀 CRNN OCR Training Guide

## ✅ Prerequisites

You have:
- ✅ Mixed dataset ready (10,615 images)
- ✅ 80/20 synthetic/real ratio
- ✅ Train/Val/Test splits

---

## 🏋️ Step 1: Start Training

### **Quick Start:**
```bash
cd /home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV
python3 train_crnn_ocr.py
```

### **Expected Output:**
```
============================================================
🚀 CRNN OCR TRAINING
============================================================

📂 Loading datasets...
   Train: 7,430 images
   Val:   1,592 images

🏗️  Building CRNN model...
   Parameters: 8,234,123

🏋️  Starting training...

============================================================
Epoch 1/100
============================================================
Training: 100%|████████| 232/232 [02:15<00:00, 1.71it/s, loss=2.3456]
Validation: 100%|████████| 50/50 [00:15<00:00, 3.21it/s]

📊 Results:
   Train Loss: 2.3456
   Val Loss:   2.1234
   Char Acc:   45.23%
   Seq Acc:    12.34%
   ✅ New best model saved!
```

---

## 📊 What's Happening

### **Model Architecture:**
```
Input Image (64×200×3)
    ↓
CNN Backbone (6 conv blocks)
    ├─ Conv + BatchNorm + ReLU + MaxPool
    ├─ Feature maps: 64 → 128 → 256 → 512
    └─ Output: (4×50×512)
    ↓
LSTM (2 layers, bidirectional)
    ├─ Hidden size: 256
    ├─ Sequence modeling
    └─ Output: (50×512)
    ↓
Fully Connected
    └─ Output: (50×11) logits
    ↓
CTC Decoder
    └─ Text: "12345678901"
```

### **Training Process:**

**Epoch 1-20:** Learning basic patterns
- Loss drops rapidly (2.5 → 0.8)
- Char accuracy: 50% → 75%
- Seq accuracy: 10% → 40%

**Epoch 21-50:** Fine-tuning
- Loss stabilizes (0.8 → 0.3)
- Char accuracy: 75% → 85%
- Seq accuracy: 40% → 70%

**Epoch 51-100:** Optimization
- Loss: 0.3 → 0.1
- Char accuracy: 85% → 92%
- Seq accuracy: 70% → 85%+

---

## 📈 Monitoring Training

### **Watch Metrics:**
```python
# Key metrics to monitor:
1. Sequence Accuracy (most important!)
   - Target: >90% on validation
   
2. Character Accuracy
   - Should be >95% at convergence
   
3. Training Loss
   - Should decrease steadily
   
4. Validation Loss
   - Watch for overfitting (val > train)
```

### **Expected Progress:**

| Epoch | Train Loss | Val Loss | Char Acc | Seq Acc |
|-------|------------|----------|----------|---------|
| 1 | 2.30 | 2.15 | 45% | 12% |
| 10 | 0.85 | 0.92 | 72% | 38% |
| 20 | 0.45 | 0.58 | 84% | 62% |
| 50 | 0.18 | 0.28 | 91% | 78% |
| 100 | 0.08 | 0.15 | 94% | 88% |

---

## 💾 Saved Files

### **During Training:**
```
checkpoints/
├── best_model.pth              ← Best validation accuracy
├── checkpoint_epoch_10.pth     ← Every 10 epochs
├── checkpoint_epoch_20.pth
└── ...

logs/
└── training_history.json       ← All metrics
```

### **Best Model Contains:**
```python
{
    'epoch': 67,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'best_seq_acc': 0.8923
}
```

---

## 🧪 Step 2: Test the Model

After training (or anytime):

```bash
python3 test_crnn_ocr.py
```

### **Output:**
```
============================================================
🔍 CRNN OCR INFERENCE
============================================================

📦 Loading model...
✅ Model loaded from: checkpoints/best_model.pth
   Best accuracy: 89.23%

🧪 Testing on 20 images from recognition/test
======================================================================
✅ GT: 00004741816 | Pred: 00004741816
✅ GT: 00012341316 | Pred: 00012341316
❌ GT: 00015271216 | Pred: 00015271316
✅ GT: 00039231816 | Pred: 00039231816
...
======================================================================
📊 Accuracy: 18/20 = 90.00%
```

---

## ⚙️ Training Parameters

### **Current Settings:**
```python
# In train_crnn_ocr.py

BATCH_SIZE = 32          # Increase if GPU memory allows
NUM_EPOCHS = 100         # Usually converges by epoch 50-70
LEARNING_RATE = 0.001    # Adam optimizer
IMG_HEIGHT = 64          # Fixed
IMG_WIDTH = 200          # For 11 characters
NUM_CLASSES = 11         # 0-9 + CTC blank
```

### **Adjustments:**

**Faster training (less accurate):**
```python
BATCH_SIZE = 64
NUM_EPOCHS = 50
```

**Better accuracy (slower):**
```python
BATCH_SIZE = 16
NUM_EPOCHS = 150
LEARNING_RATE = 0.0005
```

**GPU memory issues:**
```python
BATCH_SIZE = 16  # Or 8
num_workers = 2  # Reduce workers
```

---

## 🐛 Troubleshooting

### **Issue: CUDA Out of Memory**
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# In DataLoader
num_workers = 2
```

### **Issue: Loss not decreasing**
```
Check:
1. Learning rate too high → Try 0.0005
2. Bad augmentation → Verify sample images
3. Label mismatch → Check dataset
```

### **Issue: Overfitting (val_loss > train_loss)**
```python
# Add dropout in model
self.dropout = nn.Dropout(0.5)

# Increase augmentation
# Reduce model size
hidden_size = 128  # instead of 256
```

### **Issue: Slow training**
```python
# Use GPU
DEVICE = torch.device('cuda')

# Increase batch size
BATCH_SIZE = 64

# More workers
num_workers = 8
```

---

## 📊 Visualization

### **Plot Training History:**
```python
import json
import matplotlib.pyplot as plt

with open('logs/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_char_acc'], label='Char Acc')
plt.plot(history['val_seq_acc'], label='Seq Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

---

## 🎯 Target Metrics

### **Good Model:**
- ✅ Sequence Accuracy: >85%
- ✅ Character Accuracy: >93%
- ✅ Validation loss stable

### **Excellent Model:**
- ✅ Sequence Accuracy: >90%
- ✅ Character Accuracy: >95%
- ✅ Fast inference (<10ms per image)

---

## 🚀 Next Steps

### **After Training:**

1. **Evaluate on test set:**
   ```bash
   python3 test_crnn_ocr.py
   ```

2. **Try on real images:**
   ```python
   from test_crnn_ocr import load_model, predict_image
   model = load_model('checkpoints/best_model.pth')
   pred = predict_image(model, 'path/to/image.jpg')
   print(pred)
   ```

3. **Export for deployment:**
   ```python
   # Convert to TorchScript
   model.eval()
   traced = torch.jit.trace(model, example_input)
   traced.save('model_traced.pt')
   
   # Or ONNX
   torch.onnx.export(model, example_input, 'model.onnx')
   ```

4. **Integrate with detection pipeline:**
   - Combine with YOLO detection
   - Full end-to-end system

---

## 💡 Pro Tips

1. **Early Stopping:**
   - If val_loss doesn't improve for 15 epochs → Stop
   - Best model usually around epoch 50-70

2. **Learning Rate:**
   - Starts at 0.001
   - Reduces automatically when val_loss plateaus
   - Watch for "Reducing learning rate" messages

3. **Augmentation:**
   - Heavy for real images (20% of data)
   - Medium for synthetic (80%)
   - Balance is key!

4. **CTC Loss:**
   - May show `nan` initially (normal)
   - Should stabilize after epoch 2-3
   - If persistent → reduce learning rate

5. **Hardware:**
   - GPU: ~3 hours for 100 epochs
   - CPU: ~24 hours for 100 epochs
   - Consider cloud GPU (Google Colab, Kaggle)

---

## ✅ Success Checklist

Training successful if:
- [ ] Loss decreases steadily
- [ ] Sequence accuracy >85% on validation
- [ ] No overfitting (val_loss ≈ train_loss)
- [ ] Model saved to checkpoints/
- [ ] Test accuracy >80%

**If all checked → Model ready for deployment!** 🎉

---

## 🎓 Understanding the Model

### **CTC (Connectionist Temporal Classification):**
- Handles variable-length sequences
- No need for character segmentation
- Learns alignment automatically

### **Why it works:**
```
Input plate: "123-456-78"
Model sees:  "1112334445566677788"  (with duplicates)
CTC removes: "123456678"             (greedy decode)
Output:      "12345678"              (final prediction)
```

### **Blank token (class 10):**
- Separates repeated characters
- Example: "11" → "1_1" → "11" (where _ is blank)

---

**Ready to train! Run:** `python3 train_crnn_ocr.py` 🚀

# 🚀 Quick Start Guide

Get the License Plate Recognition system running in 5 minutes!

---

## ⚡ Fast Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 3. Install dependencies
cd streamlit_app
pip install -r requirements.txt

# 4. Download YOLOv8 model (place in project root)
# Get your trained model or download from releases
# Place: license_plate_best.pt in the root directory

# 5. Run the app!
streamlit run app.py
```

Open browser at: `http://localhost:8501`

---

## 🎮 First Run

### Test with Sample Video

1. **Upload Mode**: Click "Upload Video" in sidebar
2. **Settings**:
   - Confidence: 0.35 (default)
   - Image size: 960 (default)
   - Frame interpolation: 1x (no interpolation)
3. **Upload** your video file
4. **Click** "Start Processing"
5. **Download** results:
   - Annotated video with bounding boxes
   - Plate crops in `plate_crops/` folder

### Adjust for Your Scenario

**Highway/Fast-moving vehicles:**
```
✅ Frame interpolation: 2x
✅ Image size: 1280
✅ Confidence: 0.30
```

**Parking lot/Slow-moving:**
```
✅ Frame interpolation: 1x
✅ Image size: 960
✅ Confidence: 0.40
```

**Distant cameras:**
```
✅ Image size: 1280
✅ Min plate width: 30
✅ Confidence: 0.25
```

---

## 🔍 Understanding Output

### Annotated Video
- Green boxes: Newly detected plates (tracking ≤3 frames)
- Orange boxes: Confirmed plates (tracking >3 frames)
- ID labels: `ID:X (confidence)`

### Plate Crops Folder
```
plate_crops/
├── track_1_BEST.jpg          ← Best frame for plate ID 1
├── track_2_BEST.jpg          ← Best frame for plate ID 2
└── track_2_all_frames/       ← Debug: all frames (optional)
    ├── f00028_Q0.309_C0.50_S58_conf0.53.jpg
    ├── f00029_Q0.281_C0.50_S37_conf0.52.jpg
    └── ...
```

**Filename Breakdown:**
- `f00028`: Frame number 28
- `Q0.309`: Quality score (0-1)
- `C0.50`: Completeness (0-1, 1=full plate)
- `S58`: Sharpness (higher = sharper)
- `conf0.53`: YOLO confidence (0-1)

---

## 🎯 Common Issues

### "Model not found"
```bash
# Make sure license_plate_best.pt is in project root
ls -la license_plate_best.pt
```

### "Video codec error"
```bash
# Install FFmpeg
sudo apt install ffmpeg        # Linux
brew install ffmpeg           # Mac
choco install ffmpeg          # Windows
```

### "CUDA out of memory"
```python
# In sidebar, reduce:
Image size: 1280 → 960
Frame interpolation: 2x → 1x
```

### "Not detecting plates"
```python
# In sidebar, adjust:
Confidence: 0.35 → 0.25
Image size: 960 → 1280
```

---

## 📚 Next Steps

1. **Read full docs**: [`README.md`](README.md)
2. **Train your model**: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
3. **Customize settings**: See Configuration section in README
4. **Enable debug mode**: Save all frames to analyze quality selection

---

## 💡 Pro Tips

1. **Use debug mode first** to understand which frames are selected
2. **Start with default settings** then adjust based on results
3. **Lower confidence** if missing plates
4. **Higher image size** for distant/small plates
5. **Frame interpolation** only for fast-moving vehicles

---

Happy detecting! 🎉

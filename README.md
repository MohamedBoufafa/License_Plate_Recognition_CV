# 🚗 License Plate Recognition System (ALPR)

> Automatic License Plate Recognition using YOLOv8 detection + EasyOCR with advanced tracking and quality-based frame selection

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24.0-red.svg)

---

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Quality Scoring System](#-quality-scoring-system)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## ✨ Features

### 🎯 Detection & Tracking
- **YOLOv8-based plate detection** with multi-scale inference
- **Advanced IOU tracking** with unique plate ID assignment
- **Frame interpolation** (1x/2x/3x/4x) to catch fast-moving plates
- **Quality-based frame selection** - automatically saves best frame per plate

### 📊 Quality Scoring System
Our intelligent quality scorer prioritizes:
- **Size (50%)** - Larger plates = closer = more readable text
- **Completeness (20%)** - Full text visible, not cut off
- **Sharpness (15%)** - Clear text for OCR
- **Brightness (10%)** - Well-lit plates
- **Edge Density (5%)** - Character boundaries

**Result:** System selects the most readable frame for each plate, not just the sharpest!

### 🎨 User Interface
- **Streamlit web interface** - No coding required
- **Real-time processing** with progress tracking
- **Video upload** or **RTSP stream** support
- **Adjustable parameters** - confidence, image size, filters
- **Debug mode** - Save all frames for comparison

### 🔍 Advanced Filtering
- Aspect ratio filtering (1.2-7.0) for various plate types
- Minimum size filters (width/height)
- Confidence threshold adjustment
- Border detection (incomplete plates)

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Video Input    │
│  (MP4/RTSP)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Frame           │◄─────┤ Frame            │
│ Interpolation   │      │ Interpolation    │
│ (Optional 2-4x) │      │ (Optical Flow)   │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ YOLOv8 Plate    │◄─────┤ Multi-scale      │
│ Detection       │      │ (960px + 1280px) │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ IOU Tracker     │◄─────┤ Track Management │
│ (Unique IDs)    │      │ (Lost frames: 30)│
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Quality Scorer  │◄─────┤ • Size (50%)     │
│ (Per Frame)     │      │ • Complete (20%) │
│                 │      │ • Sharp (15%)    │
│                 │      │ • Bright (10%)   │
│                 │      │ • Edges (5%)     │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐
│ Best Frame      │
│ Selection       │
│ (Per Track)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save Crops      │
│ • track_X_BEST  │
│ • Debug frames  │
└─────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for real-time processing)
- FFmpeg (for video encoding)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/license-plate-recognition.git
cd license-plate-recognition
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
cd streamlit_app
pip install -r requirements.txt
```

### Step 4: Download YOLOv8 Model
Place your trained YOLOv8 model (`license_plate_best.pt`) in the project root:
```bash
# Option 1: Train your own model (see IMPLEMENTATION_PLAN.md)
# Option 2: Download pre-trained model
wget https://your-model-link.com/license_plate_best.pt
```

---

## 🎮 Usage

### Run Streamlit App
```bash
cd streamlit_app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Upload Video Mode
1. Select **"Upload Video"** from sidebar
2. Upload MP4/MOV/AVI file
3. Adjust settings:
   - **Confidence threshold**: 0.25-0.50 (lower = more detections)
   - **Image size**: 640/960/1280 (higher = better for small/distant plates)
   - **Frame interpolation**: 1x-4x (higher = catches fast-moving plates)
4. Click **"Start Processing"**
5. Download annotated video + plate crops

### RTSP Stream Mode
1. Select **"RTSP Stream"** from sidebar
2. Enter RTSP URL: `rtsp://username:password@ip:port/stream`
3. View live detections with tracking

### Webcam Mode
```bash
# Run from command line
python -c "from plate_detector import process_webcam; process_webcam(confidence_threshold=0.35)"
```

---

## ⚙️ Configuration

### Sidebar Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Confidence threshold** | 0.15-0.75 | 0.35 | Lower = more sensitive detection |
| **Image size** | 640/960/1280 | 960 | Higher = better for distant plates |
| **Frame interpolation** | 1x-4x | 1x | Creates intermediate frames |
| **Min plate width** | 20-100 | 40 | Minimum pixels width |
| **Min plate height** | 10-50 | 18 | Minimum pixels height |

### When to Adjust

**Missing plates?**
- ⬇️ Lower confidence to 0.25
- ⬆️ Increase image size to 1280
- ⬆️ Enable frame interpolation (2x)

**Too many false positives?**
- ⬆️ Raise confidence to 0.45
- ⬆️ Increase min plate size

**Fast-moving vehicles?**
- ⬆️ Enable 2x-3x frame interpolation
- ⬆️ Use image size 1280

---

## 🎯 Quality Scoring System

Our quality scorer ensures the **most readable frame** is selected for each tracked plate.

### Scoring Formula
```python
quality = (
    area * 0.50 +           # Size (larger = closer = more readable)
    completeness * 0.20 +   # Full text visible (not cut off)
    sharpness * 0.15 +      # Text clarity
    brightness * 0.10 +     # Lighting quality
    edges * 0.05            # Character boundaries
) * confidence_factor (0.8-1.0)
```

### Completeness Detection
Checks if plate appears complete vs partial:
- ✅ Centered plates with clear borders → C=1.0
- ⚠️ Plates near frame edge → C=0.5-0.9
- ❌ Plates cut by frame border → C=0.3-0.5

### Debug Mode
Enable "🔬 Debug: Save ALL frames" to see:
```
track_2_all_frames/
├─ f00028_Q0.309_C0.50_S58_conf0.53.jpg   ← Larger, complete text
├─ f00036_Q0.337_C0.50_S50_conf0.63.jpg   ← SELECTED (highest Q)
├─ f00052_Q0.331_C0.90_S102_conf0.35.jpg  ← Sharper but smaller
└─ f00056_Q0.306_C0.35_S109_conf0.28.jpg  ← Very sharp but tiny
```

**Filename format:** `f{frame}_Q{quality}_C{completeness}_S{sharpness}_conf{confidence}.jpg`

---

## 📁 Project Structure

```
license-plate-recognition/
├── streamlit_app/
│   ├── app.py                    # Streamlit UI
│   ├── plate_detector.py         # Core detection + tracking logic
│   ├── requirements.txt          # Python dependencies
│   ├── uploads/                  # Uploaded videos (gitignored)
│   ├── output/                   # Processed videos (gitignored)
│   └── plate_crops/              # Best frames per plate (gitignored)
│       ├── track_1_BEST.jpg
│       ├── track_2_BEST.jpg
│       └── track_2_all_frames/   # Debug mode
├── license_plate_best.pt         # YOLOv8 model (gitignored)
├── IMPLEMENTATION_PLAN.md        # Dataset + training guide
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Troubleshooting

### Issue: No plates detected
**Solutions:**
1. Lower confidence threshold to 0.25
2. Increase image size to 1280
3. Check if model file exists
4. Verify video codec (try re-encoding with FFmpeg)

### Issue: Wrong frame selected as "best"
**Solutions:**
1. Enable debug mode to see all frames
2. Check if larger plates have lower quality scores
3. Verify completeness scores (C value in filename)

### Issue: Video output codec errors
**Solutions:**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # Mac
```

### Issue: CUDA out of memory
**Solutions:**
1. Reduce image size from 1280 → 960
2. Disable frame interpolation
3. Process smaller video segments

---

## 📊 Performance

| Resolution | FPS (GPU) | FPS (CPU) | Detection Quality |
|------------|-----------|-----------|-------------------|
| 640px      | 30-45     | 5-8       | Good (close plates) |
| 960px      | 20-30     | 3-5       | Better (balanced) |
| 1280px     | 10-15     | 1-2       | Best (distant plates) |

*Tested on: NVIDIA RTX 3060, Intel i7-11700K*

---

## 🎓 Training Your Own Model

See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for:
- Dataset recommendations (UFPR-ALPR, CCPD)
- YOLOv8 training guide
- Data augmentation strategies
- Evaluation metrics

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** - Object detection framework
- **EasyOCR** - Optical character recognition
- **Streamlit** - Web interface framework
- **UFPR-ALPR Dataset** - Training data

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🎯 Future Enhancements

- [ ] OCR integration for automatic text recognition
- [ ] Multi-language plate support
- [ ] Database storage for plate history
- [ ] RESTful API for integration
- [ ] Docker containerization
- [ ] Real-time stream processing optimization
- [ ] Export to CSV/JSON
- [ ] Plate type classification (car/truck/motorcycle)

---

**Made with ❤️ for computer vision and traffic monitoring**

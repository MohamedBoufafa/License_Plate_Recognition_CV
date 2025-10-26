# 🚗 License Plate Recognition System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Real-time License Plate Detection, Tracking, and Recognition** powered by YOLOv8 and custom OCR model.

---

## ✨ Features

### 🎯 **Detection & Tracking**
- **YOLOv8-based detection** with adaptive confidence thresholds
- **Multi-object tracking** with IoU-based matching
- **Far plate optimization** - detects plates at 20m+ distance
- **Adaptive min-frames** for robust tracking

### 🔤 **OCR Recognition**
- **Custom CRNN model** trained on 50K+ synthetic plates
- **Arabic & French support** (Moroccan plates)
- **Real-time text extraction** from tracked plates
- **Best frame selection** for optimal OCR accuracy

### 📹 **Multiple Input Modes**
- **Video Upload** - Process MP4/AVI/MOV files
- **Live Webcam** - Real-time detection from camera
- **Phone Streaming** - RTSP stream from mobile device
- **Batch Processing** - Multiple videos simultaneously

### 🎨 **User Interface**
- **Streamlit web app** - No coding required
- **Real-time progress tracking** with statistics
- **Debug mode** - Save all detection frames
- **Export results** - Annotated videos + plate crops

---

## 🚀 Quick Start

### **Online Demo** (Streamlit Cloud)

**Coming Soon!** Deployment in progress.

### **Local Installation**

#### **1. Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV.git
cd License_Plate_Recognition_CV
```

#### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **3. Download Models**
```bash
# YOLO model (6.2 MB)
wget https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV/releases/download/v1.0/license_plate_best.pt

# OCR model (43 MB)
wget https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV/releases/download/v1.0/best_ocr_model.pth
```

#### **4. Run Application**
```bash
cd streamlit_app
streamlit run app.py
```

Open **http://localhost:8501** in your browser!

---

## 📖 Usage

### **Video Processing**

1. Select **"Upload Video"** mode
2. Upload your video file (MP4/AVI/MOV)
3. Adjust settings:
   - **Confidence:** 0.35-0.50 (default: 0.47)
   - **Image Size:** 960-1280 (higher = better far plates)
   - **Min Frames:** 3 (tracks need 3+ frames to confirm)
4. Click **"Start Processing"**
5. Download **annotated video** + **plate crops**

### **Phone Streaming (RTSP)**

1. Install **MediaMTX** on your PC:
   ```bash
   wget https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_amd64.tar.gz
   tar -xzf mediamtx_v1.9.3_linux_amd64.tar.gz
   ./mediamtx
   ```

2. Install **Larix Broadcaster** on your phone:
   - [Android](https://play.google.com/store/apps/details?id=com.wmspanel.larix_broadcaster)
   - [iOS](https://apps.apple.com/app/larix-broadcaster/id1042474385)

3. Configure Larix:
   - URL: `rtsp://<YOUR-PC-IP>:8554/stream`
   - Resolution: 1280x720 or 1920x1080
   - Start streaming

4. In app:
   - Select **"Phone Stream (RTSP)"**
   - Click **"▶️ Start Stream"** (view-only mode)
   - Click **"🎯 Toggle Processing"** when ready to detect

**See [Phone Streaming Guide](docs/PHONE_STREAMING_GUIDE.md) for details.**

### **Webcam Mode**

1. Select **"Live Camera"**
2. Adjust confidence threshold
3. Click **"Start Camera"**
4. Real-time detection on your webcam!

---

## 🎯 Performance

### **Detection Accuracy**

| Distance | Accuracy | Recommended Settings |
|----------|----------|---------------------|
| **0-10m** | 95%+ | imgsz=960, conf=0.45 |
| **10-20m** | 85-90% | imgsz=1280, conf=0.40 |
| **20-30m** | 70-80% | imgsz=1280, conf=0.35 |
| **30m+** | 50-70% | imgsz=1280, conf=0.30 |

### **Processing Speed**

| Mode | FPS | GPU Usage |
|------|-----|-----------|
| **Video (imgsz=960)** | 8-12 | 40-50% |
| **Video (imgsz=1280)** | 4-8 | 60-70% |
| **RTSP (view-only)** | 10 | 0% |
| **RTSP (processing)** | 10-12 | 40-60% |
| **Webcam** | 15-20 | 50-60% |

*Tested on: RTX 3060, 1920x1080 input*

---

## 📁 Project Structure

```
License_Plate_Recognition_CV/
├── streamlit_app/              # Main application
│   ├── app.py                  # Streamlit UI
│   ├── plate_detector.py       # Detection & tracking logic
│   ├── ocr_module.py           # OCR model wrapper
│   ├── requirements.txt        # App dependencies
│   └── README_OCR.md           # OCR documentation
│
├── docs/                       # Documentation
│   ├── PHONE_STREAMING_GUIDE.md
│   ├── OPTIMAL_SETTINGS_GUIDE.md
│   ├── TOGGLE_PROCESSING_GUIDE.md
│   ├── VIEW_ONLY_LAG_FIX.md
│   └── ... (30+ guides)
│
├── notebooks/                  # Training notebooks
│   ├── license_plate.ipynb     # YOLO training
│   ├── kaggle_crnn_training.ipynb  # OCR training
│   └── synthetic_plate_generator.ipynb
│
├── scripts/                    # Utility scripts
│   ├── train_crnn_ocr.py       # OCR training script
│   ├── ocr_inference.py        # OCR testing
│   └── mix_datasets_local.py   # Dataset preparation
│
├── .streamlit/                 # Streamlit config
│   └── config.toml
│
├── license_plate_best.pt       # YOLO weights (6.2 MB)
├── best_ocr_model.pth          # OCR weights (43 MB)
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── QUICKSTART.md               # Quick start guide
└── README.md                   # This file
```

---

## 🛠️ Advanced Features

### **Adaptive Tracking**

```python
# Automatically adjusts IoU thresholds based on plate size
- Large plates (close): IoU 0.3 (standard)
- Small plates (far): IoU 0.2 + center distance
- Prevents track fragmentation for distant plates
```

### **Adaptive Min-Frames**

```python
# Dynamic confirmation thresholds
- Ultra small (<0.1%): 1 frame
- Small (0.1-0.2%): 1 frame
- Medium (0.2-0.5%): 2 frames
- Normal (>0.5%): 3 frames
```

### **Smart Detection**

```python
# Confidence threshold adapts to plate size
- Small plates: Lower threshold (0.30)
- Medium plates: Base threshold (0.47)
- Large plates: Higher threshold (0.50)
```

### **OCR Optimization**

```python
# Best frame selection for OCR
- Tracks quality score per frame
- Considers: sharpness, size, confidence
- Runs OCR on highest quality frame
```

---

## 📊 Training

### **YOLO Model**

Trained on **UFPR-ALPR dataset** + custom annotations:
- **Training data:** 4,500 images
- **Validation:** 500 images
- **Epochs:** 100
- **mAP@50:** 0.95+

**Train your own:**
```bash
# See docs/KAGGLE_TRAINING_GUIDE.md
python train_yolo.py --data plates.yaml --epochs 100 --imgsz 640
```

### **OCR Model**

Custom **CRNN architecture** trained on synthetic plates:
- **Training data:** 50,000 synthetic plates
- **Character set:** 0-9, A-Z (Arabic support)
- **Accuracy:** 92% on test set

**Train your own:**
```bash
# See docs/OCR_IMPLEMENTATION_PLAN.md
python train_crnn_ocr.py --epochs 30 --batch-size 64
```

---

## 🎨 Screenshots

### **Video Processing**
*(Add screenshot of video upload interface)*

### **Phone Streaming**
*(Add screenshot of RTSP stream with detections)*

### **Results**
*(Add screenshot of plate crops with OCR results)*

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Documentation

Comprehensive guides available in [`docs/`](docs/) folder:

### **User Guides:**
- [Quick Start](QUICKSTART.md)
- [Phone Streaming Setup](docs/PHONE_STREAMING_GUIDE.md)
- [Optimal Settings](docs/OPTIMAL_SETTINGS_GUIDE.md)
- [Toggle Processing](docs/TOGGLE_PROCESSING_GUIDE.md)
- [Run Locally](docs/RUN_LOCALLY.md)

### **Technical Docs:**
- [Adaptive Tracking](docs/ADAPTIVE_TRACKING_FIX.md)
- [Far Plate Optimization](docs/FAR_PLATE_OPTIMIZATION.md)
- [OCR Integration](docs/OCR_INTEGRATION_SUMMARY.md)
- [Debug Tracking](docs/DEBUG_TRACKING_GUIDE.md)

### **Training Guides:**
- [YOLO Training](docs/TRAINING_GUIDE.md)
- [Kaggle Training](docs/KAGGLE_TRAINING_GUIDE.md)
- [OCR Training](docs/OCR_IMPLEMENTATION_PLAN.md)
- [Dataset Preparation](docs/DATA_MIXING_STRATEGY.md)

---

## 🔧 Troubleshooting

### **Common Issues:**

**1. Low FPS / Slow Processing**
```bash
# Reduce image size
imgsz: 1280 → 960 → 640

# Enable frame skipping (RTSP mode)
Process every N frames: 2-3

# Check GPU usage
nvidia-smi
```

**2. Missing Far Plates**
```bash
# Increase image size
imgsz: 960 → 1280

# Lower confidence
conf: 0.47 → 0.35

# Enable adaptive tracking (already enabled by default)
```

**3. RTSP Stream Lag**
```bash
# See docs/VIEW_ONLY_LAG_FIX.md
- Use view-only mode for positioning
- Toggle processing only when needed
- Reduce Larix resolution to 1280x720
```

**4. OCR Not Working**
```bash
# Check model exists
ls -lh best_ocr_model.pth

# Enable OCR in settings
☑ Enable OCR (checkbox)

# Ensure plates are confirmed (min_frames reached)
```

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@misc{license_plate_recognition_cv,
  author = {Your Name},
  title = {License Plate Recognition with YOLOv8 and Custom OCR},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **EasyOCR** - OCR library
- **Streamlit** - Web interface framework
- **UFPR-ALPR** - Training dataset

---

## 📧 Contact

- **GitHub:** [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email:** your.email@example.com

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=YOUR_USERNAME/License_Plate_Recognition_CV&type=Date)](https://star-history.com/#YOUR_USERNAME/License_Plate_Recognition_CV&Date)

---

<div align="center">
Made with ❤️ for the Computer Vision community
</div>

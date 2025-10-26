# 🚀 Deployment Guide

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [GitHub Setup](#github-setup)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Post-Deployment](#post-deployment)
5. [Troubleshooting](#troubleshooting)

---

## ✅ Pre-Deployment Checklist

Before deploying, ensure:

- [ ] **Models uploaded to GitHub Releases** (not in repo)
- [ ] **Documentation organized** in `docs/` folder
- [ ] **`.gitignore` configured** (no large files)
- [ ] **`requirements.txt` up to date**
- [ ] **`packages.txt` created** (system dependencies)
- [ ] **`.streamlit/config.toml` configured**
- [ ] **README.md updated** with correct URLs
- [ ] **License file included**
- [ ] **Test locally** before pushing

---

## 🐙 GitHub Setup

### **Step 1: Create GitHub Repository**

1. Go to [https://github.com/new](https://github.com/new)
2. Repository name: `License_Plate_Recognition_CV`
3. Description: `Real-time License Plate Detection, Tracking, and OCR with YOLOv8`
4. Visibility: **Public** (required for Streamlit Cloud free tier)
5. **Do NOT** initialize with README (we already have one)
6. Click **"Create repository"**

---

### **Step 2: Upload Models to GitHub Releases**

**Important:** Don't commit large model files to Git!

```bash
# 1. Go to your repo on GitHub
# 2. Click "Releases" → "Create a new release"
# 3. Tag version: v1.0
# 4. Release title: "Initial Release - Models v1.0"
# 5. Upload files:
#    - license_plate_best.pt (6.2 MB)
#    - best_ocr_model.pth (43 MB)
# 6. Click "Publish release"
```

**Get download URLs:**
```
https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV/releases/download/v1.0/license_plate_best.pt
https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV/releases/download/v1.0/best_ocr_model.pth
```

---

### **Step 3: Update Model Paths in Code**

**Edit `streamlit_app/plate_detector.py`:**

```python
# Line ~20-23 (MODEL_PATH)
# OLD:
MODEL_PATH = os.environ.get("LPR_MODEL_PATH", "../license_plate_best.pt")

# NEW:
MODEL_PATH = os.environ.get("LPR_MODEL_PATH", "license_plate_best.pt")

# Add download function:
import urllib.request

def download_model_if_needed():
    """Download models from GitHub Releases if not present"""
    models = {
        "license_plate_best.pt": "https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV/releases/download/v1.0/license_plate_best.pt",
        "best_ocr_model.pth": "https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV/releases/download/v1.0/best_ocr_model.pth"
    }
    
    for model_file, url in models.items():
        if not os.path.exists(model_file):
            print(f"Downloading {model_file}...")
            urllib.request.urlretrieve(url, model_file)
            print(f"✅ Downloaded {model_file}")

# Call at startup
download_model_if_needed()
```

**Edit `streamlit_app/app.py` (for OCR model path):**

```python
# Around line 390
# OLD:
ocr_model_path="../best_ocr_model.pth"

# NEW:
ocr_model_path="best_ocr_model.pth"
```

---

### **Step 4: Push to GitHub**

```bash
cd /home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV

# Initialize git (if not already)
git init

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/License_Plate_Recognition_CV.git

# Stage all files
git add .

# Commit
git commit -m "Initial commit: Complete LPR system with YOLOv8 and OCR"

# Push
git push -u origin main
```

**If `main` branch doesn't exist:**
```bash
git branch -M main
git push -u origin main
```

---

## ☁️ Streamlit Cloud Deployment

### **Step 1: Sign Up for Streamlit Cloud**

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **"Sign up"**
3. Sign in with **GitHub** account
4. Authorize Streamlit Cloud access

---

### **Step 2: Deploy App**

1. Click **"New app"**
2. **Repository:** Select `YOUR_USERNAME/License_Plate_Recognition_CV`
3. **Branch:** `main`
4. **Main file path:** `streamlit_app/app.py`
5. **App URL:** Choose custom subdomain (e.g., `lpr-demo`)
6. Click **"Deploy!"**

---

### **Step 3: Configure Advanced Settings** (Optional)

Click **"Advanced settings"** before deploying:

**Python version:**
```
3.11
```

**Secrets (if needed):**
```toml
# Add any API keys or sensitive data
# Example:
[secrets]
API_KEY = "your_secret_key"
```

---

### **Step 4: Wait for Deployment**

Deployment takes 5-10 minutes:

```
Building...
  ├── Installing system packages (packages.txt)
  ├── Installing Python packages (requirements.txt)
  ├── Downloading models from GitHub Releases
  └── Starting app...

✅ Your app is live!
```

**App URL:** `https://lpr-demo.streamlit.app`

---

## 🎯 Post-Deployment

### **Test Your App**

1. **Upload Video Mode:**
   - Upload a test video
   - Verify detection works
   - Check if annotated video downloads

2. **Webcam Mode:**
   - Click "Start Camera"
   - Verify camera access (may not work on Streamlit Cloud)

3. **RTSP Mode:**
   - ⚠️ **Note:** RTSP streaming won't work on Streamlit Cloud (requires local setup)
   - Disable or hide this mode for cloud deployment

---

### **Update README with Live Demo Link**

```bash
# Edit README.md
## 🚀 Quick Start

### **Online Demo** (Streamlit Cloud)

**Try it now:** [https://lpr-demo.streamlit.app](https://lpr-demo.streamlit.app)

*Note: RTSP streaming requires local installation*
```

**Commit and push:**
```bash
git add README.md
git commit -m "Add Streamlit Cloud demo link"
git push
```

---

### **Monitor App Performance**

In Streamlit Cloud dashboard:

1. **Logs** - View real-time logs
2. **Metrics** - CPU/Memory usage
3. **Errors** - Track errors and crashes
4. **Usage** - Visitor statistics

---

## 🐛 Troubleshooting

### **Issue 1: Model Download Fails**

**Error:**
```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

**Solution:**
```python
# Add to plate_detector.py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

---

### **Issue 2: Out of Memory**

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce batch size / image size
# Edit app.py - change default imgsz
imgsz_rtsp = st.sidebar.select_slider(
    "YOLO imgsz", options=[640, 768, 960], value=640  # Lower default
)
```

---

### **Issue 3: Slow Processing**

**Streamlit Cloud uses CPU, not GPU!**

**Optimize for CPU:**

```python
# Force CPU inference
device = "cpu"  # Instead of "cuda"

# Use smaller models
# Consider FP16 or quantized models for faster inference
```

---

### **Issue 4: Package Installation Fails**

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch==2.4.1
```

**Solution:**
```bash
# Update requirements.txt
# Use CPU-only PyTorch for cloud:

torch==2.4.1+cpu
torchvision==0.19.1+cpu

# Add PyTorch index:
--extra-index-url https://download.pytorch.org/whl/cpu
```

---

### **Issue 5: File Upload Size Limit**

**Error:**
```
Uploaded file exceeds 200MB limit
```

**Solution:**
```toml
# Already configured in .streamlit/config.toml
[server]
maxUploadSize = 500  # MB
```

If still limited, Streamlit Cloud max is 200MB. For larger files:
- Use URL input instead of upload
- Process video locally

---

## 📦 Cloud-Specific Optimizations

### **Disable RTSP Mode for Cloud**

**Edit `streamlit_app/app.py`:**

```python
# Around line 40
app_mode = st.sidebar.selectbox(
    "Choose the mode",
    ["Upload Video", "Live Camera"]  # Remove "Phone Stream (RTSP)"
)

# Or add a note:
if app_mode == "Phone Stream (RTSP)":
    st.warning("⚠️ RTSP streaming requires local installation. Please use 'Upload Video' mode.")
```

---

### **Optimize Model Loading**

```python
# Cache model to avoid reloading
import streamlit as st

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)

model = load_yolo_model()
```

---

### **Add Usage Limits**

```python
# Limit video length for cloud deployment
MAX_VIDEO_DURATION = 60  # seconds

if video_duration > MAX_VIDEO_DURATION:
    st.error(f"Video too long! Please upload videos under {MAX_VIDEO_DURATION}s.")
```

---

## 🔄 Update Deployment

### **Option 1: Git Push (Automatic)**

Streamlit Cloud auto-deploys on git push:

```bash
# Make changes
git add .
git commit -m "Fix: Update model download logic"
git push

# App redeploys automatically in ~5 min
```

---

### **Option 2: Manual Redeploy**

In Streamlit Cloud dashboard:

1. Go to your app
2. Click **"Reboot app"**
3. Wait for restart

---

## 📊 Performance Expectations

### **Streamlit Cloud (Free Tier)**

| Resource | Limit |
|----------|-------|
| **CPU** | 0.78 cores |
| **RAM** | 800 MB |
| **Storage** | 1 GB |
| **Bandwidth** | Unlimited |
| **Concurrent Users** | ~5-10 |

### **Processing Speed**

| Video Resolution | FPS (CPU) |
|-----------------|-----------|
| 1920x1080 | 1-2 FPS |
| 1280x720 | 2-4 FPS |
| 640x480 | 4-8 FPS |

**⚠️ Cloud is MUCH slower than local GPU!**

---

## 💰 Upgrade Options

For production use:

**Streamlit Cloud Pro:**
- $20/month per user
- 4 cores
- 16 GB RAM
- Priority support

**Alternatives:**
- **AWS EC2** + GPU instance
- **Google Cloud Run**
- **Heroku**
- **Docker + Kubernetes**

---

## ✅ Final Checklist

Before going live:

- [ ] Models uploaded to Releases
- [ ] Code pushed to GitHub
- [ ] App deployed to Streamlit Cloud
- [ ] App tested and working
- [ ] README updated with demo link
- [ ] Documentation organized
- [ ] Usage limits set
- [ ] Error handling added
- [ ] Performance optimized for CPU
- [ ] RTSP mode disabled/noted for cloud

---

## 🎉 Success!

Your app is now live and accessible worldwide! 🌍

**Share your app:**
- Tweet: "Just deployed my License Plate Recognition app! 🚗🔍"
- LinkedIn: Add to projects
- Portfolio: Showcase your work

**Next steps:**
- Add analytics
- Collect user feedback
- Iterate and improve
- Consider monetization

---

## 📚 Additional Resources

- **Streamlit Cloud Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Deployment Tutorial:** https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- **GitHub Actions:** Automate testing/deployment
- **Custom Domains:** Point your own domain to app

---

**Need help?** Open an issue on GitHub or contact support!

🚀 Happy Deploying! ✨

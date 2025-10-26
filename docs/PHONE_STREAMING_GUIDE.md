# 📱 Phone Streaming Guide - License Plate Detection

## 🎯 Overview

Your app supports **live phone streaming** via RTSP (Real-Time Streaming Protocol). This allows you to use your phone as a mobile camera for real-time license plate detection.

---

## 🔧 **Setup Requirements**

### **1. MediaMTX Server (On Your PC)**

MediaMTX is an RTSP server that receives the stream from your phone and makes it available to your app.

#### **Installation:**

**Linux/macOS:**
```bash
# Download MediaMTX
wget https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_amd64.tar.gz

# Extract
tar -xzf mediamtx_v1.9.3_linux_amd64.tar.gz

# Run
./mediamtx
```

**Or using Docker:**
```bash
docker run --rm -it -p 8554:8554 -p 1935:1935 -p 8888:8888 bluenviron/mediamtx
```

**Windows:**
- Download from: https://github.com/bluenviron/mediamtx/releases
- Extract and run `mediamtx.exe`

#### **Verify MediaMTX is Running:**
```
Server will start on:
- RTSP: port 8554
- Web UI: http://localhost:8888

You should see:
"listener opened on :8554 (TCP+UDP)"
```

---

### **2. Larix Broadcaster (On Your Phone)**

Larix is a mobile app that streams video from your phone camera to an RTSP server.

#### **Installation:**

**Android:**
- Google Play: https://play.google.com/store/apps/details?id=com.wmspanel.larix_broadcaster

**iOS:**
- App Store: https://apps.apple.com/app/larix-broadcaster/id1042474385

---

## 📱 **Step-by-Step Setup**

### **Step 1: Find Your PC's IP Address**

**Linux/macOS:**
```bash
# WiFi
ifconfig wlan0 | grep inet

# Or
ip addr show wlan0 | grep inet
```

**Windows:**
```cmd
ipconfig
```

**Look for:** Something like `192.168.1.100` (your local network IP)

---

### **Step 2: Configure Larix on Your Phone**

1. **Open Larix Broadcaster**
2. **Click Settings (⚙️)**
3. **Add New Connection:**
   - **Name:** My PC MediaMTX
   - **URL:** `rtsp://<YOUR-PC-IP>:8554/stream`
     - Example: `rtsp://192.168.1.100:8554/stream`
   - **Mode:** RTSP
   - **Video Codec:** H.264
   - **Resolution:** 1280x720 or 1920x1080 (higher = better detection, more data)
   - **Framerate:** 25-30 FPS
   - **Bitrate:** 2000-5000 kbps

4. **Save Connection**

---

### **Step 3: Start Streaming from Phone**

1. **Open Larix**
2. **Select your connection** (My PC MediaMTX)
3. **Click the RED record button** to start streaming
4. **You should see:**
   - Green "LIVE" indicator
   - Upload speed indicator
   - Connection stable

**Troubleshooting:**
- ❌ **"Connection failed"** → Check PC IP, MediaMTX running, same WiFi
- ❌ **"Network unreachable"** → Firewall blocking port 8554 (see below)
- ❌ **Lag/stuttering** → Reduce resolution or bitrate

---

### **Step 4: Run Your License Plate App**

1. **Start the app:**
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

2. **In the browser (http://localhost:8501):**
   - Select **"Phone Stream (RTSP)"** from sidebar

3. **Configure settings:**
   - **RTSP URL:** `rtsp://127.0.0.1:8554/stream` (default, reads from MediaMTX)
   - **Detection confidence:** 0.35-0.50 (adjust for your needs)
   - **YOLO imgsz:** 1280 (higher = better far plate detection)
   - **Min plate width:** 40px
   - **Min plate height:** 18px

4. **Click "▶️ Start stream"**

5. **You should see:**
   - Live video feed from your phone ✅
   - License plates detected with bounding boxes ✅
   - Track IDs assigned ✅
   - Real-time FPS counter ✅

---

## 🎥 **Phone Camera Tips**

### **For Best Detection:**

1. **Lighting:**
   - ✅ Good daylight or bright artificial light
   - ❌ Avoid backlighting (sun behind plates)
   - ❌ Avoid night without good lighting

2. **Distance:**
   - **Close plates (2-10m):** Easy, high accuracy
   - **Medium plates (10-20m):** Good, use imgsz=1280
   - **Far plates (20m+):** Challenging, may need lower confidence

3. **Angle:**
   - ✅ Front-facing (perpendicular to plate)
   - ⚠️ Angled (up to 45° works)
   - ❌ Side view (hard to read)

4. **Movement:**
   - ✅ Stable camera (tripod/mount ideal)
   - ⚠️ Slow panning (tracking works)
   - ❌ Fast shaking (tracking fails)

5. **Resolution:**
   - **1920x1080 (Full HD):** Best quality, recommended
   - **1280x720 (HD):** Good balance
   - **640x480:** Fast but lower accuracy

---

## 🔥 **Advanced Configuration**

### **Custom RTSP URL:**

If you want to use a different stream path:

**In MediaMTX config (mediamtx.yml):**
```yaml
paths:
  mystream:
    source: publisher
```

**In Larix:**
```
rtsp://192.168.1.100:8554/mystream
```

**In your app:**
```
rtsp://127.0.0.1:8554/mystream
```

---

### **Multiple Phone Streams:**

**Phone 1 (Larix):**
```
rtsp://192.168.1.100:8554/camera1
```

**Phone 2 (Larix):**
```
rtsp://192.168.1.100:8554/camera2
```

**In your app, process each:**
```
rtsp://127.0.0.1:8554/camera1
rtsp://127.0.0.1:8554/camera2
```

---

### **Stream Over Internet (Not Local Network):**

**Option 1: Port Forwarding**
1. Router settings → Forward port 8554 to your PC
2. Use public IP: `rtsp://<PUBLIC-IP>:8554/stream`
3. ⚠️ **Security risk!** Use authentication

**Option 2: Ngrok Tunnel**
```bash
ngrok tcp 8554
```

Use provided URL in Larix.

**Option 3: VPN**
- Connect phone and PC to same VPN
- Use VPN IP address

---

## 🛠️ **Firewall Configuration**

### **Linux (UFW):**
```bash
sudo ufw allow 8554/tcp
sudo ufw allow 8554/udp
sudo ufw reload
```

### **Windows Firewall:**
1. Windows Defender Firewall → Advanced Settings
2. Inbound Rules → New Rule
3. Port → TCP/UDP → 8554
4. Allow the connection

### **macOS:**
```bash
# System Preferences → Security & Privacy → Firewall
# → Firewall Options → Allow MediaMTX
```

---

## 🎯 **Use Cases**

### **1. Parking Lot Monitoring**
- Mount phone on tripod
- Point at parking entrance/exit
- Stream to PC running detection
- Save detected plates

### **2. Traffic Analysis**
- Phone in car dashboard
- Stream while driving
- Real-time plate detection
- Count unique vehicles

### **3. Security Gate**
- Phone at gate entrance
- Stream to office PC
- Auto-detect and log vehicles
- Integration with access control

### **4. Mobile Patrol**
- Phone in patrol vehicle
- Stream to central server
- Real-time plate checking
- Alert on matches

---

## 🐛 **Troubleshooting**

### **Problem: Can't connect from phone to PC**

**Check:**
1. ✅ Phone and PC on **same WiFi network**
2. ✅ MediaMTX running on PC (`./mediamtx`)
3. ✅ Correct PC IP address in Larix
4. ✅ Port 8554 not blocked by firewall
5. ✅ No VPN/proxy interfering

**Test connection:**
```bash
# On PC, check if port is listening:
netstat -tuln | grep 8554

# Should show:
tcp6       0      0 :::8554      :::*      LISTEN
```

---

### **Problem: Stream connects but app shows error**

**Check:**
1. ✅ App RTSP URL: `rtsp://127.0.0.1:8554/stream` (localhost)
2. ✅ Larix is actively streaming (green LIVE indicator)
3. ✅ MediaMTX shows "reader connected" in logs

**Test stream:**
```bash
# Play stream with ffplay (if installed):
ffplay rtsp://127.0.0.1:8554/stream
```

---

### **Problem: High latency/lag**

**Solutions:**
1. **Lower resolution** in Larix (1280x720 instead of 1920x1080)
2. **Lower bitrate** (2000 kbps instead of 5000)
3. **Lower framerate** (15-20 FPS instead of 30)
4. **Reduce YOLO imgsz** in app (960 instead of 1280)
5. **Use 5GHz WiFi** instead of 2.4GHz (faster)

---

### **Problem: Poor detection accuracy**

**Solutions:**
1. **Increase YOLO imgsz** (1280)
2. **Lower confidence threshold** (0.35 - 0.45)
3. **Better lighting** on phone side
4. **Reduce camera movement** (use tripod/mount)
5. **Get closer** to plates (10-15m optimal)
6. **Check phone focus** (tap screen to focus)

---

## 📊 **Performance Benchmarks**

### **Resolution vs Performance:**

| Resolution | FPS (App) | Detection Quality | Bandwidth |
|-----------|-----------|------------------|-----------|
| 640x480   | 15-20 FPS | Fair             | ~500 kbps |
| 1280x720  | 8-12 FPS  | Good             | ~2 Mbps   |
| 1920x1080 | 4-8 FPS   | Excellent        | ~5 Mbps   |

### **Distance vs Accuracy:**

| Distance | Accuracy | Recommended Settings |
|----------|----------|---------------------|
| 2-10m    | 95%+     | imgsz=960, conf=0.45 |
| 10-20m   | 80-90%   | imgsz=1280, conf=0.40 |
| 20-30m   | 60-80%   | imgsz=1280, conf=0.35 |
| 30m+     | 30-60%   | imgsz=1280, conf=0.30, adaptive enabled |

---

## 🎬 **Quick Start Summary**

**5 Minutes to Stream:**

1. **Start MediaMTX on PC:**
   ```bash
   ./mediamtx
   ```

2. **Get PC IP:**
   ```bash
   ip addr | grep inet
   # Example: 192.168.1.100
   ```

3. **Configure Larix on phone:**
   - URL: `rtsp://192.168.1.100:8554/stream`
   - Start streaming

4. **Run app:**
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

5. **In browser:**
   - Select "Phone Stream (RTSP)"
   - Click "▶️ Start stream"

**Done! You're streaming!** 🎉

---

## 📚 **Additional Resources**

- **MediaMTX Docs:** https://github.com/bluenviron/mediamtx
- **Larix Manual:** https://softvelum.com/larix/
- **RTSP Protocol:** https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol
- **YOLOv8 Docs:** https://docs.ultralytics.com/

---

## 🎯 **Alternative Apps to Larix**

### **Android:**
- **IP Webcam** (Free, simpler UI)
- **DroidCam** (USB + WiFi)
- **RTSP Camera** (Dedicated RTSP)

### **iOS:**
- **Larix Broadcaster** (Best for iOS)
- **Camera Live** (Alternative)
- **IP Camera** (Simple)

---

## ✅ **Checklist**

Before streaming, verify:

- [ ] MediaMTX installed and running
- [ ] PC IP address noted
- [ ] Larix installed on phone
- [ ] Phone and PC on same WiFi
- [ ] RTSP URL configured in Larix
- [ ] Firewall allows port 8554
- [ ] Streamlit app running
- [ ] Good lighting on plates
- [ ] Camera stable/mounted

**Happy Streaming!** 📱✨🚗

import os
import time
import tempfile
from dataclasses import dataclass

import streamlit as st

from plate_detector import process_webcam, process_rtsp_stream, get_system_info, detect_video_with_tracking

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="License Plate Detection", layout="wide")

UPLOAD_DIR = "uploads"
CROPS_DIR = "plate_crops"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)


@dataclass
class Flag:
    value: bool = False


def main():
    st.title("License Plate Detection System")

    # ── System info
    sys_info = get_system_info()
    st.sidebar.title("System Information")
    if sys_info["device"] == "cuda":
        st.sidebar.success(f"🚀 GPU: {sys_info['gpu_name']}")
        st.sidebar.info(f"CUDA Version: {sys_info['cuda_version']}")
    else:
        st.sidebar.warning("⚠️ Running on CPU")
        st.sidebar.info(f"YOLO Model Device: {sys_info['model_device']}")

    # ── App mode
    st.sidebar.title("Settings")
    app_mode = st.sidebar.selectbox(
        "Choose the mode",
        ["Upload Video", "Live Camera", "Phone Stream (RTSP)"]
    )

    # Detection Settings
    st.sidebar.markdown("## Detection Settings")
    conf_thresh = st.sidebar.slider(
        "YOLO confidence threshold", 0.0, 1.0, 0.47, 0.01)
    imgsz = st.sidebar.select_slider("YOLO imgsz (higher = better small plates, slower)",
                                     options=[640, 768, 960, 1280], value=1280)
    
    st.sidebar.caption("💡 For far plates: Use 1280 imgsz + lower min size (20x10)")
    
    min_box_w = st.sidebar.number_input(
        "Min plate width (px)", 10, 4096, 20, 5,
        help="Lower values detect smaller/farther plates. Default 20px catches most far plates.")
    min_box_h = st.sidebar.number_input(
        "Min plate height (px)", 10, 4096, 10, 2,
        help="Lower values detect smaller/farther plates. Default 10px catches most far plates.")
    
    # Tracking & Crop Settings
    st.sidebar.markdown("## Tracking & Crops")
    
    min_frames = st.sidebar.slider(
        "⏱️ Min frames to confirm plate",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of consecutive frames a plate must be tracked before confirmation. Higher = fewer false positives, but might miss fast plates."
    )
    st.sidebar.caption(f"🎯 Current: {min_frames} frame{'s' if min_frames > 1 else ''} (plates visible <{min_frames} frames are filtered out)")
    
    save_best_crops = st.sidebar.checkbox(
        "Save best frame of each plate", value=True,
        help="Saves the clearest image of each tracked plate for OCR")
    
    debug_mode = st.sidebar.checkbox(
        "🔬 Debug: Save ALL frames", value=False,
        help="Testing mode - saves every single frame of each track to compare quality")
    
    # OCR Settings
    st.sidebar.markdown("## 🔤 OCR Settings")
    enable_ocr = st.sidebar.checkbox(
        "Enable OCR", value=True,
        help="Run OCR on detected plates to extract numbers")
    
    ocr_model_path = st.sidebar.text_input(
        "OCR Model Path", 
        value="best_ocr_model.pth",
        help="Path to trained OCR model (best_model.pth)")
    
    if enable_ocr:
        st.sidebar.info("📝 OCR will extract plate numbers from detected plates")
    else:
        st.sidebar.warning("OCR disabled - only detection/tracking")
    
    frame_multiplier = st.sidebar.selectbox(
        "⚡ Frame Interpolation",
        options=[1, 2, 3, 4],
        index=0,
        format_func=lambda x: f"{x}x - {'Off (Normal)' if x == 1 else f'{x}x frames ({x}x slower, catches more plates)'}",
        help="Generate extra frames between video frames to catch fast-moving plates"
    )
    
    st.sidebar.markdown("## Detection Settings")
    st.sidebar.caption("💡 If plates are visible but not detected:")
    st.sidebar.info(
        "1️⃣ Lower 'Confidence threshold' to 0.25\n"
        "2️⃣ Multi-scale runs at 1280px for small/distant plates\n"
        "3️⃣ Reduce 'Min plate width/height' if needed"
    )

    # ──────────────────────────────────────────────────────────────────────────
    if app_mode == "Upload Video":
        st.sidebar.markdown("## Upload Video")
        video_file = st.sidebar.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov"])

        if "stop_processing" not in st.session_state:
            st.session_state.stop_processing = False

        if video_file is not None:
            # Save uploaded video (close so OpenCV can read it on Windows)
            suffix = os.path.splitext(video_file.name)[1]
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tfile.write(video_file.read())
            tfile.close()

            st.video(tfile.name)

            if st.sidebar.button("🎯 Detect & Track"):
                st.session_state.stop_processing = False

                progress_bar = st.progress(0)
                status_text = st.empty()

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### Output Preview")
                    render_area = st.empty()
                with col2:
                    st.markdown("### Tracked Plates")
                    crops_display = st.empty()

                if st.button("🛑 Stop Processing"):
                    st.session_state.stop_processing = True
                    st.warning("Stopping tracking...")

                stop_flag = Flag(value=st.session_state.stop_processing)

                try:
                    base_out = os.path.join(
                        UPLOAD_DIR, "output_" + os.path.basename(tfile.name))
                    
                    crops_dir = CROPS_DIR if save_best_crops else None

                    def update_status(msg: str):
                        status_text.text(msg)

                    def update_progress(pct: int):
                        progress_bar.progress(max(0, min(100, int(pct))))

                    if frame_multiplier > 1:
                        st.info(f"🎯 Tracking mode with {frame_multiplier}x frame interpolation: Processing {frame_multiplier}x more frames to catch fast-moving plates. Plates must be tracked for {min_frames}+ frames to be counted.")
                    else:
                        st.info(f"🎯 Tracking mode: Each plate is tracked across frames. Only plates detected in {min_frames}+ frames are processed (filters false detections).")

                    final_video_path = detect_video_with_tracking(
                        input_path=tfile.name,
                        output_path=base_out,
                        confidence_threshold=conf_thresh,
                        imgsz=imgsz,
                        progress_callback=update_progress,
                        status_callback=update_status,
                        should_stop=stop_flag,
                        min_box_w=int(min_box_w),
                        min_box_h=int(min_box_h),
                        save_crops=save_best_crops,
                        crops_dir=crops_dir,
                        debug_save_all_frames=debug_mode,
                        frame_interpolation_multiplier=frame_multiplier,
                        enable_ocr=enable_ocr,
                        ocr_model_path=ocr_model_path if enable_ocr else None,
                        min_frames_to_confirm=min_frames,
                    )

                    st.success("✅ Tracking completed successfully!")
                    
                    # Ensure video file is fully written
                    time.sleep(0.5)
                    
                    with col1:
                        if final_video_path and os.path.exists(final_video_path):
                            # Check file size to ensure it was written
                            file_size = os.path.getsize(final_video_path)
                            if file_size > 0:
                                file_ext = os.path.splitext(final_video_path)[1]
                                st.info(f"📹 Video: {file_size / (1024*1024):.2f} MB ({file_ext.upper()})")
                                
                                # Display video
                                try:
                                    render_area.video(final_video_path)
                                except Exception as e:
                                    st.error(f"Error displaying video: {e}")
                                    st.info("Try downloading the video instead")
                                
                                # Provide download button for the result
                                # Detect if it's MP4 or AVI
                                file_ext = os.path.splitext(final_video_path)[1]
                                mime_type = "video/mp4" if file_ext == ".mp4" else "video/x-msvideo"
                                
                                with open(final_video_path, "rb") as f:
                                    st.download_button(
                                        "⬇️ Download Tracked Video",
                                        data=f.read(),
                                        file_name=f"tracked_{os.path.splitext(os.path.basename(tfile.name))[0]}{file_ext}",
                                        mime=mime_type,
                                    )
                            else:
                                st.error("Video file is empty (0 bytes)")
                        elif final_video_path is None:
                            # Video encoding failed, but processing succeeded
                            st.warning("⚠️ Video encoding failed due to codec issues, but plate detection completed successfully!")
                            st.info("📊 Results are still available below (crops and OCR)")
                        else:
                            st.error(f"Output video file not found at: {final_video_path}")
                    
                    # Display saved crops
                    if save_best_crops and crops_dir and os.path.exists(crops_dir):
                        with col2:
                            # Show debug folders if debug mode
                            if debug_mode:
                                all_items = os.listdir(crops_dir)
                                debug_folders = sorted([d for d in all_items if d.endswith('_all_frames')])
                                best_files = sorted([f for f in all_items if f.endswith('_BEST.jpg')])
                                
                                if debug_folders:
                                    st.markdown(f"### 🔬 Debug Mode: {len(debug_folders)} Tracks")
                                    st.caption("Compare all frames to verify best frame selection")
                                    
                                    # Show best file for each track
                                    for best_file in best_files[:5]:  # Show first 5
                                        track_id = best_file.replace('track_', '').replace('_BEST.jpg', '')
                                        best_path = os.path.join(crops_dir, best_file)
                                        
                                        with st.expander(f"📁 Track #{track_id} - Best Frame"):
                                            st.image(best_path, caption=f"BEST Frame (Track {track_id})", width=300)
                                            
                                            # Show all frames for this track
                                            track_folder = f"track_{track_id}_all_frames"
                                            track_folder_path = os.path.join(crops_dir, track_folder)
                                            if os.path.exists(track_folder_path):
                                                all_frames = sorted(os.listdir(track_folder_path))
                                                st.caption(f"📊 Total frames: {len(all_frames)}")
                                                
                                                # Show sample of frames
                                                cols = st.columns(3)
                                                for idx, frame_file in enumerate(all_frames[:9]):  # Show first 9
                                                    frame_path = os.path.join(track_folder_path, frame_file)
                                                    cols[idx % 3].image(frame_path, caption=frame_file[:20], width=150)
                                                
                                                if len(all_frames) > 9:
                                                    st.info(f"... and {len(all_frames) - 9} more frames in folder: {track_folder}")
                                else:
                                    st.info("Processing... debug folders will appear soon")
                            else:
                                # Normal mode - show best frames only
                                crop_files = sorted([f for f in os.listdir(crops_dir) if f.endswith('_BEST.jpg')])
                                if crop_files:
                                    st.markdown(f"### 🎯 {len(crop_files)} Best Frames")
                                    if enable_ocr:
                                        st.caption("Clearest frame + OCR results")
                                    else:
                                        st.caption("Automatically selected clearest/largest frame of each plate")
                                    
                                    for crop_file in crop_files[:20]:  # Show up to 20
                                        crop_path = os.path.join(crops_dir, crop_file)
                                        track_id = crop_file.replace('track_', '').replace('_BEST.jpg', '')
                                        
                                        # Check if OCR result exists
                                        ocr_txt_path = os.path.join(crops_dir, f"track_{track_id}_OCR.txt")
                                        ocr_text = ""
                                        if os.path.exists(ocr_txt_path):
                                            with open(ocr_txt_path, 'r') as f:
                                                ocr_text = f.read().strip()
                                        
                                        # Display image with OCR result
                                        if ocr_text:
                                            st.image(crop_path, caption=f"Track #{track_id} | 🔤 {ocr_text}", width=200)
                                        else:
                                            st.image(crop_path, caption=f"Track #{track_id}", width=200)
                                else:
                                    st.info(f"No confirmed plates yet (need {min_frames}+ frames)")

                except Exception as e:
                    st.error(f"Error during tracking: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    try:
                        if os.path.exists(tfile.name):
                            os.unlink(tfile.name)
                    except Exception as e:
                        st.warning(
                            f"Could not remove temporary file: {str(e)}")

    # ──────────────────────────────────────────────────────────────────────────
    elif app_mode == "Live Camera":
        st.sidebar.markdown("## Live Camera Settings")
        st.write("Opens a native OpenCV window. Press **Q** to quit.")
        confidence = st.sidebar.slider(
            "Detection Confidence", 0.0, 1.0, 0.5, 0.01)
        if st.sidebar.button("Start Camera"):
            try:
                process_webcam(confidence_threshold=confidence)
            except Exception as e:
                st.error(f"Error accessing camera: {str(e)}")

    # ──────────────────────────────────────────────────────────────────────────
    else:  # Phone Stream (RTSP)
        st.sidebar.markdown("## Phone Stream Settings")
        st.write(
            "On your phone (Larix) set: `rtsp://<YOUR-PC-IP>:8554/stream`\n"
            "MediaMTX must be running on your PC. This app reads `rtsp://127.0.0.1:8554/stream`."
        )

        rtsp_url = st.sidebar.text_input(
            "RTSP URL to read", "rtsp://127.0.0.1:8554/stream")
        conf_rtsp = st.sidebar.slider(
            "Detection confidence", 0.0, 1.0, 0.35, 0.01)
        imgsz_rtsp = st.sidebar.select_slider(
            "YOLO imgsz", options=[640, 768, 960, 1280], value=960)
        min_w_rtsp = st.sidebar.number_input(
            "Min plate width (px)", 10, 4096, 40, 5, key="rtsp_min_w")
        min_h_rtsp = st.sidebar.number_input(
            "Min plate height (px)", 10, 4096, 18, 2, key="rtsp_min_h")
        
        st.sidebar.markdown("### ⚡ Performance")
        frame_skip = st.sidebar.slider(
            "Process every N frames", 1, 5, 2, 1,
            help="Higher = faster stream, lower accuracy. 1 = process all frames (slow), 2-3 = balanced, 4-5 = fast")
        
        st.sidebar.markdown("### 🔤 OCR Settings")
        enable_ocr_rtsp = st.sidebar.checkbox("Enable OCR", value=True, key="rtsp_ocr")
        min_frames_rtsp = st.sidebar.number_input(
            "Min frames to confirm", 1, 10, 3, 1, key="rtsp_min_frames",
            help="Minimum frames a plate must be tracked before OCR runs")

        if "streaming" not in st.session_state:
            st.session_state.streaming = False
        if "processing" not in st.session_state:
            st.session_state.processing = False

        # Stream control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            start_stream_btn = st.button("▶️ Start Stream")
        with col2:
            toggle_processing_btn = st.button("🎯 Toggle Processing")
        with col3:
            stop_btn = st.button("⏹️ Stop")

        if start_stream_btn:
            st.session_state.streaming = True
            st.session_state.processing = False  # Start without processing
        if toggle_processing_btn:
            st.session_state.processing = not st.session_state.processing
        if stop_btn:
            st.session_state.streaming = False
            st.session_state.processing = False

        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        status_placeholder = st.empty()

        # Show current status
        if st.session_state.streaming:
            if st.session_state.processing:
                status_placeholder.success("🎯 Stream Active + Processing Enabled")
            else:
                status_placeholder.info("👁️ Stream Active (View Only - Click 'Toggle Processing' to enable detection)")

        if st.session_state.streaming:
            try:
                for processed in process_rtsp_stream(
                    rtsp_url=rtsp_url,
                    confidence_threshold=conf_rtsp,
                    imgsz=imgsz_rtsp,
                    min_box_w=int(min_w_rtsp),
                    min_box_h=int(min_h_rtsp),
                    process_every_n_frames=frame_skip,
                    enable_processing=st.session_state.processing,
                    enable_ocr=enable_ocr_rtsp,  # OCR toggle
                    ocr_model_path="best_ocr_model.pth",  # OCR model path
                    min_frames_to_confirm=min_frames_rtsp,  # Min frames for OCR
                ):
                    rgb = processed[:, :, ::-1]  # BGR->RGB
                    frame_placeholder.image(
                        rgb, channels="RGB", width="stretch")
                    if not st.session_state.streaming:
                        break
            except Exception as e:
                st.error(f"Stream error: {e}")
            finally:
                st.session_state.streaming = False
                st.session_state.processing = False
                info_placeholder.info("Stream stopped.")
                status_placeholder.empty()
        else:
            st.info(
                "📱 Set your Larix URL, click **Start Stream** to view, then **Toggle Processing** when ready to detect plates.")


if __name__ == "__main__":
    main()

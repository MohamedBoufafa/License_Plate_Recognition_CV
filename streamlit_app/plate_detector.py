import os
import time
from typing import Callable, Dict, Optional, Tuple, List, Set
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import OCR module
try:
    from ocr_module import get_ocr_model
    OCR_AVAILABLE = True
except ImportError:
    print("[Warning] OCR module not available")
    OCR_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Model download helper

def download_model_if_needed(model_file, url):
    """Download model from GitHub Releases if not present"""
    if not os.path.exists(model_file):
        print(f"📥 Downloading {model_file} from GitHub Releases...")
        print(f"   URL: {url}")
        try:
            import urllib.request
            import ssl
            # Bypass SSL verification if needed
            ssl_context = ssl._create_unverified_context()
            urllib.request.urlretrieve(url, model_file, context=ssl_context)
            print(f"✅ Downloaded {model_file} ({os.path.getsize(model_file) / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"❌ Failed to download {model_file}: {e}")
            print(f"   Please ensure models are uploaded to GitHub Releases:")
            print(f"   https://github.com/MohamedBoufafa/License_Plate_Recognition_CV/releases")
            raise RuntimeError(
                f"Could not download {model_file}. "
                f"Please upload models to GitHub Releases (tag: v1.0) first. "
                f"See DEPLOYMENT_GUIDE.md for instructions."
            )
    else:
        print(f"✅ Model found: {model_file} ({os.path.getsize(model_file) / 1024 / 1024:.1f} MB)")

# Download models from GitHub Releases
print("🔍 Checking for model files...")
YOLO_MODEL_URL = "https://github.com/MohamedBoufafa/License_Plate_Recognition_CV/releases/download/v1.0/license_plate_best.pt"
OCR_MODEL_URL = "https://github.com/MohamedBoufafa/License_Plate_Recognition_CV/releases/download/v1.0/best_ocr_model.pth"

download_model_if_needed("license_plate_best.pt", YOLO_MODEL_URL)
download_model_if_needed("best_ocr_model.pth", OCR_MODEL_URL)
print("✅ All models ready!")

# ──────────────────────────────────────────────────────────────────────────────
# Device & models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch device: {device}")

MODEL_PATH = os.environ.get("LPR_MODEL_PATH", "license_plate_best.pt")
model = YOLO(MODEL_PATH)
if hasattr(model, "model"):
    model.model.to(device)

# Configure NMS for faster processing
model.overrides['max_det'] = 100  # Balance: enough for multi-plate scenes
model.overrides['agnostic_nms'] = True  # Faster NMS
model.overrides['iou'] = 0.5  # NMS IoU threshold (higher = fewer duplicates)
model.overrides['half'] = True  # Use FP16 for faster inference on GPU



def get_system_info() -> Dict[str, str]:
    try:
        gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "N/A"
    except Exception:
        gpu_name = "N/A"
    try:
        model_device = str(next(model.model.parameters()).device)
    except Exception:
        model_device = "unknown"

    info = {
        "device": device,
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda if device == "cuda" else "N/A",
        "model_device": model_device,
    }
    return info

# ──────────────────────────────────────────────────────────────────────────────
# Smart Detection Validation

def is_valid_plate_detection(
    bbox: Tuple[int, int, int, int],
    confidence: float,
    frame_width: int,
    frame_height: int,
    base_confidence_threshold: float = 0.40
) -> bool:
    """
    Smart validation that adjusts confidence threshold based on detection size.
    Allows lower confidence for small/far plates, higher for large/close detections.
    Also validates aspect ratio to filter non-plate shapes.
    
    This solves the problem:
    - Low confidence → catches far plates BUT also random objects
    - High confidence → misses far plates
    
    Solution: Adaptive thresholds + shape validation
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Calculate detection area
    area = width * height
    frame_area = frame_width * frame_height
    relative_area = area / frame_area
    
    # 1. ASPECT RATIO CHECK (most important!)
    # Filter out non-plate shapes IMMEDIATELY
    if height > 0:
        aspect_ratio = width / height
        # License plates: typically 1.5:1 to 6.0:1
        # Too narrow or too wide = not a plate
        if aspect_ratio < 1.2 or aspect_ratio > 7.0:
            return False  # Definitely not a plate shape
    else:
        return False
    
    # 2. SIZE-BASED CONFIDENCE ADJUSTMENT
    # Small detections (far plates): Need lower confidence
    # Large detections (close/random objects): Need higher confidence
    
    if relative_area < 0.003:  # Extremely small (< 0.3% of frame) - VERY FAR PLATES
        # Very far plates can have quite low confidence
        required_confidence = base_confidence_threshold - 0.15  # e.g., 0.47 → 0.32
        
    elif relative_area < 0.008:  # Very small (0.3-0.8%) - FAR PLATES
        # Far plates can have lower confidence (they're naturally weaker)
        required_confidence = base_confidence_threshold - 0.10  # e.g., 0.47 → 0.37
        
    elif relative_area < 0.015:  # Small (0.8-1.5%) - MEDIUM DISTANCE
        # Medium distance plates
        required_confidence = base_confidence_threshold - 0.05  # e.g., 0.47 → 0.42
        
    elif relative_area < 0.05:  # Medium (1.5-5%) - NORMAL RANGE
        # Normal sized plates - use base threshold
        required_confidence = base_confidence_threshold  # e.g., 0.40
        
    elif relative_area < 0.15:  # Large (5-15%) - CLOSE
        # Close plates - require higher confidence (could be false positive)
        required_confidence = base_confidence_threshold + 0.05  # e.g., 0.40 → 0.45
        
    else:  # Very large (>15%) - TOO CLOSE or NOT A PLATE
        # Very large detections are suspicious (might be sign, billboard, etc.)
        required_confidence = base_confidence_threshold + 0.15  # e.g., 0.40 → 0.55
    
    # 3. APPLY ADJUSTED THRESHOLD
    if confidence < required_confidence:
        return False
    
    # 4. MINIMUM SIZE CHECK (regardless of confidence)
    # Too small = noise (very permissive for far plates)
    if width < 12 or height < 6:
        return False
    
    # 5. MAXIMUM SIZE CHECK
    # Too large = probably not a plate
    if relative_area > 0.30:  # More than 30% of frame
        return False
    
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Adaptive Min Frames Helper

def get_adaptive_min_frames(bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int, base_min_frames: int) -> int:
    """
    Get adaptive minimum frames required based on detection size.
    Far plates (small) need fewer frames to confirm because:
    - They're harder to detect consistently
    - Visible for shorter duration
    - Lower tracking persistence
    
    Returns lower min_frames for smaller plates.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    frame_area = frame_width * frame_height
    relative_area = area / frame_area
    
    # Adaptive min_frames based on size
    if relative_area < 0.001:  # Ultra small (very very far) - < 0.1%
        # Ultra far plates: just need 1 frame
        return 1  # Always 1 frame for ultra small
        
    elif relative_area < 0.002:  # Extremely small (very far) - 0.1-0.2%
        # Very far plates: just need 1 frame
        return 1  # Always 1 frame
        
    elif relative_area < 0.005:  # Very small (far) - 0.2-0.5%
        # Far plates: need 1 less frame
        return max(1, base_min_frames - 1)  # e.g., 3 → 2
        
    elif relative_area < 0.015:  # Small (medium distance) - 0.5-1.5%
        # Medium distance: slightly easier
        return max(2, base_min_frames - 1)  # e.g., 3 → 2
        
    else:  # Normal to large
        # Use base min_frames
        return base_min_frames

# ──────────────────────────────────────────────────────────────────────────────
# Image Quality Metrics for Best Frame Selection

def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher value = sharper image (better for OCR)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def calculate_brightness_quality(image: np.ndarray) -> float:
    """
    Calculate brightness quality score (0-1).
    Penalizes too dark or too bright images.
    Optimal range: 50-200 on grayscale
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_brightness = gray.mean()
    
    # Optimal brightness around 120-140
    if 50 <= mean_brightness <= 200:
        # Normalize to 0-1, peak at 130
        if mean_brightness < 130:
            score = (mean_brightness - 50) / 80.0
        else:
            score = (200 - mean_brightness) / 70.0
        return max(0.0, min(1.0, score))
    else:
        # Penalize extreme values
        if mean_brightness < 50:
            return mean_brightness / 50.0 * 0.3
        else:
            return (255 - mean_brightness) / 55.0 * 0.3

def calculate_aspect_ratio_score(bbox: Tuple[int, int, int, int], expected_range=(1.5, 5.5)) -> float:
    """
    Score how close the aspect ratio is to expected plate ratio.
    License plates typically have aspect ratio between 1.5:1 and 5.5:1
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    if height == 0:
        return 0.0
    
    aspect_ratio = width / height
    min_ar, max_ar = expected_range
    
    # Perfect range
    if min_ar <= aspect_ratio <= max_ar:
        # Bonus for being in sweet spot (2.5-4.5)
        if 2.5 <= aspect_ratio <= 4.5:
            return 1.0
        return 0.8
    
    # Penalize how far outside range
    if aspect_ratio < min_ar:
        return max(0.2, aspect_ratio / min_ar)
    else:
        return max(0.2, max_ar / aspect_ratio)

def calculate_edge_density(image: np.ndarray) -> float:
    """
    Calculate edge density - more edges = clearer characters.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
    return edge_ratio

def calculate_completeness_score(bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int, crop: np.ndarray) -> float:
    """
    Check if plate appears complete/full vs partial/cut-off.
    Penalizes plates touching frame borders or with unusual characteristics.
    
    Returns 0.0-1.0 where:
    - 1.0 = definitely complete plate
    - 0.5 = might be partial
    - 0.0 = definitely cut off
    """
    x1, y1, x2, y2 = bbox
    score = 1.0
    
    # Check if plate touches frame borders (likely cut off)
    # Use stricter margin - only penalize if REALLY at edge (not just close)
    border_margin = 2  # Reduced from 5 - less sensitive
    
    # Calculate how much of plate is near edge
    width = x2 - x1
    height = y2 - y1
    
    # Only penalize if plate is AT edge (within 2px) AND it's a significant portion
    edge_penalty = 1.0
    if x1 <= border_margin:
        edge_penalty *= 0.5  # Left edge
    if x2 >= (frame_width - border_margin):
        edge_penalty *= 0.5  # Right edge  
    if y1 <= border_margin:
        edge_penalty *= 0.6  # Top edge (less critical)
    if y2 >= (frame_height - border_margin):
        edge_penalty *= 0.6  # Bottom edge (less critical)
    
    score *= edge_penalty
    
    # Check aspect ratio - full plates have typical ratios
    width = x2 - x1
    height = y2 - y1
    if height > 0:
        ar = width / height
        # Ideal aspect ratio for full plates: 2.0 - 5.0
        if 2.0 <= ar <= 5.0:
            score *= 1.0  # Perfect
        elif 1.5 <= ar < 2.0 or 5.0 < ar <= 6.0:
            score *= 0.9  # Acceptable
        elif 1.2 <= ar < 1.5 or 6.0 < ar <= 7.0:
            score *= 0.7  # Suspicious (might be partial/angled)
        else:
            score *= 0.5  # Very unusual (likely partial)
    
    # Check for strong edges at crop borders (indicates cut-off)
    if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 20:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        h, w = gray.shape
        
        # Check edges at borders
        edge_thickness = 2
        top_edge = gray[:edge_thickness, :].std()
        bottom_edge = gray[-edge_thickness:, :].std()
        left_edge = gray[:, :edge_thickness].std()
        right_edge = gray[:, -edge_thickness:].std()
        
        # High variance at borders = likely cut through plate
        avg_border_var = (top_edge + bottom_edge + left_edge + right_edge) / 4
        if avg_border_var > 50:  # Strong edges at borders
            score *= 0.8
    
    return max(0.0, min(1.0, score))

def calculate_plate_quality_score(crop: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float, frame_width: int = 1920, frame_height: int = 1080) -> float:
    """
    Comprehensive quality score for a license plate frame.
    Combines multiple metrics weighted by importance for OCR readability.
    
    Returns higher scores for frames that are:
    - Complete/full plate (NEW - prevents partial plates winning)
    - Sharper (most important for quality)
    - Better aspect ratio (frontal view)
    - Well-lit
    - Larger size
    - Clear character edges
    """
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 20:
        return 0.0
    
    # Calculate individual metrics
    completeness = calculate_completeness_score(bbox, frame_width, frame_height, crop)
    sharpness = calculate_sharpness(crop)
    brightness_quality = calculate_brightness_quality(crop)
    aspect_ratio_score = calculate_aspect_ratio_score(bbox)
    edge_density = calculate_edge_density(crop)
    
    # Normalize sharpness (typical range 0-2000, good plates 200+)
    sharpness_normalized = min(1.0, sharpness / 500.0)
    
    # Normalize edge density (typical range 0.05-0.3)
    edge_normalized = min(1.0, edge_density / 0.2)
    
    # Calculate area (normalized by typical plate sizes)
    # Larger plates = closer = more readable text!
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_normalized = min(1.0, area / 5000.0)  # Lower threshold = more credit for size
    
    # Weighted combination - Prioritize SIZE and COMPLETENESS for readable text
    # Larger plate with complete text beats small sharp plate with partial text!
    base_quality = (
        area_normalized * 0.50 +           # SIZE IS #1 - larger text = more readable
        completeness * 0.20 +              # COMPLETE text (not cut off) is critical
        sharpness_normalized * 0.15 +      # Sharpness helps but less than size
        brightness_quality * 0.10 +        # Well-lit text
        edge_normalized * 0.05             # Character edges
    )
    
    # Include confidence as factor but NOT dominant
    confidence_factor = 0.8 + (confidence * 0.2)  # Range: 0.8 to 1.0 (smaller impact)
    
    # Completeness already included in base_quality, just apply confidence
    quality_score = base_quality * confidence_factor
    
    return quality_score

# ──────────────────────────────────────────────────────────────────────────────
# Simple IOU Tracker


@dataclass
class Track:
    """Represents a tracked license plate"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frames_tracked: int = 0
    frames_lost: int = 0
    best_confidence: float = 0.0
    best_bbox: Tuple[int, int, int, int] = None
    best_frame_crop: np.ndarray = None
    best_quality_score: float = 0.0  # Comprehensive quality metric
    plate_number: str = ""  # OCR result
    ocr_confidence: float = 0.0  # OCR confidence (if available)


class SimpleTracker:
    """Simple IOU-based tracker for license plates"""
    
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks: List[Track] = []
        self.confirmed_tracks_history: List[Track] = []  # Permanent storage of confirmed tracks
        self.next_track_id = 1
        self.frame_width = 1920  # Will be updated
        self.frame_height = 1080
    
    @staticmethod
    def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_center_distance(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Compute normalized distance between box centers"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Centers
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # Distance
        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        
        # Normalize by box size
        box_size = np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
        
        return dist / max(box_size, 1)
    
    def get_adaptive_iou_threshold(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Get adaptive IoU threshold based on detection size.
        Small plates (far away) need lower IoU threshold due to:
        - Small pixel movement = large relative change
        - Lower detection precision
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = self.frame_width * self.frame_height
        relative_area = area / frame_area
        
        # Adaptive thresholds based on size
        if relative_area < 0.002:  # Extremely small (very far)
            return 0.15  # Very lenient (50% reduction)
        elif relative_area < 0.005:  # Very small (far)
            return 0.20  # Lenient (33% reduction)
        elif relative_area < 0.015:  # Small (medium distance)
            return 0.25  # Slightly lenient
        else:  # Normal to large
            return self.iou_threshold  # Standard 0.3
    
    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float, np.ndarray]], frame_width: int = 1920, frame_height: int = 1080, debug: bool = False, frame_idx: int = 0) -> List[Track]:
        """
        Update tracks with new detections.
        detections: List of (bbox, confidence, crop_image)
        frame_width, frame_height: Frame dimensions for completeness checking
        Returns: List of active tracks
        """
        # Update frame dimensions for adaptive thresholds
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Mark all tracks as potentially lost
        for track in self.tracks:
            track.frames_lost += 1
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for det_idx, (det_bbox, det_conf, det_crop) in enumerate(detections):
            best_score = 0.0
            best_track_idx = -1
            
            # Get adaptive IoU threshold for this detection (based on size)
            adaptive_threshold = self.get_adaptive_iou_threshold(det_bbox)
            
            for track_idx, track in enumerate(self.tracks):
                if track.frames_lost > self.max_lost_frames:
                    continue
                
                # Primary matching: IoU
                iou = self.compute_iou(det_bbox, track.bbox)
                
                # For small plates, also consider center distance as secondary metric
                x1, y1, x2, y2 = det_bbox
                width = x2 - x1
                height = y2 - y1
                area = width * height
                relative_area = area / (frame_width * frame_height)
                
                if relative_area < 0.005:  # Small plate - use hybrid matching
                    # Center distance (normalized)
                    center_dist = self.compute_center_distance(det_bbox, track.bbox)
                    
                    # Hybrid score: IoU + (1 - center_distance)
                    # For small plates, if centers are close, that's a strong signal
                    if center_dist < 1.5:  # Centers within 1.5x plate size
                        score = iou * 0.7 + (1.0 - min(center_dist, 1.0)) * 0.3
                    else:
                        score = iou
                else:
                    score = iou
                
                if score > best_score and score >= adaptive_threshold:
                    best_score = score
                    best_track_idx = track_idx
            
            if best_track_idx >= 0:
                # Update existing track
                track = self.tracks[best_track_idx]
                track.bbox = det_bbox
                track.confidence = det_conf
                track.frames_tracked += 1
                track.frames_lost = 0
                matched_tracks.add(best_track_idx)
                matched_detections.add(det_idx)
                
                # Calculate comprehensive quality score for this frame
                current_quality = calculate_plate_quality_score(det_crop, det_bbox, det_conf, frame_width, frame_height)
                
                if track.best_bbox is None:
                    # First frame for this track
                    track.best_confidence = det_conf
                    track.best_bbox = det_bbox
                    track.best_frame_crop = det_crop.copy()
                    track.best_quality_score = current_quality
                else:
                    # Compare quality scores - update if current is better
                    # No threshold needed - always keep the truly best frame
                    if current_quality > track.best_quality_score:
                        track.best_confidence = det_conf
                        track.best_bbox = det_bbox
                        track.best_frame_crop = det_crop.copy()
                        track.best_quality_score = current_quality
        
        # Create new tracks for unmatched detections
        for det_idx, (det_bbox, det_conf, det_crop) in enumerate(detections):
            if det_idx not in matched_detections:
                initial_quality = calculate_plate_quality_score(det_crop, det_bbox, det_conf, frame_width, frame_height)
                new_track = Track(
                    track_id=self.next_track_id,
                    bbox=det_bbox,
                    confidence=det_conf,
                    frames_tracked=1,
                    frames_lost=0,
                    best_confidence=det_conf,
                    best_bbox=det_bbox,
                    best_frame_crop=det_crop.copy(),
                    best_quality_score=initial_quality
                )
                self.tracks.append(new_track)
                self.next_track_id += 1
        
        # Remove tracks that have been lost too long
        tracks_to_remove = [t for t in self.tracks if t.frames_lost > self.max_lost_frames]
        if tracks_to_remove:
            for track in tracks_to_remove:
                x1, y1, x2, y2 = track.bbox
                area = (x2-x1) * (y2-y1)
                rel_area = area / (self.frame_width * self.frame_height)
                adaptive_min = get_adaptive_min_frames(track.bbox, self.frame_width, self.frame_height, 3)
                is_confirmed = track.frames_tracked >= adaptive_min
                
                # Save confirmed tracks to permanent history before removing
                if is_confirmed and track.track_id not in [t.track_id for t in self.confirmed_tracks_history]:
                    self.confirmed_tracks_history.append(track)
                
                if debug:
                    status = "✅ CONFIRMED" if is_confirmed else "❌ DISCARDED"
                    print(f"Frame {frame_idx}: 🗑️ REMOVED TRACK ID={track.track_id} - "
                          f"{status}, tracked={track.frames_tracked} frames, "
                          f"lost={track.frames_lost} frames, "
                          f"size={x2-x1}x{y2-y1} ({rel_area*100:.3f}%)")
        
        self.tracks = [t for t in self.tracks if t.frames_lost <= self.max_lost_frames]
        
        # Return only active tracks (recently seen)
        return [t for t in self.tracks if t.frames_lost == 0]
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks including recently lost ones"""
        return self.tracks
    
    def get_confirmed_tracks(self, min_frames: int) -> List[Track]:
        """Get tracks that have been seen for at least min_frames"""
        return [t for t in self.tracks if t.frames_tracked >= min_frames]
    
    def get_confirmed_tracks_adaptive(self, base_min_frames: int) -> List[Track]:
        """
        Get tracks that have been confirmed using adaptive min_frames.
        Far/small plates need fewer frames to be considered confirmed.
        Returns all confirmed tracks (active + historical).
        """
        confirmed = []
        confirmed_ids = set()
        
        # Add current active confirmed tracks
        for track in self.tracks:
            # Calculate adaptive min_frames for this track
            adaptive_min = get_adaptive_min_frames(track.bbox, self.frame_width, self.frame_height, base_min_frames)
            if track.frames_tracked >= adaptive_min and track.track_id not in confirmed_ids:
                confirmed.append(track)
                confirmed_ids.add(track.track_id)
        
        # Add historical confirmed tracks (already removed but were confirmed)
        for track in self.confirmed_tracks_history:
            if track.track_id not in confirmed_ids:
                confirmed.append(track)
                confirmed_ids.add(track.track_id)
        
        return confirmed

# ──────────────────────────────────────────────────────────────────────────────
# Video processing with tracking


def detect_video_with_tracking(
    input_path: str,
    output_path: str,
    confidence_threshold: float = 0.47,
    imgsz: int = 1280,  # Higher default for far plate detection
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    should_stop=None,
    min_box_w: int = 20,  # Lower for far plate detection
    min_box_h: int = 10,  # Lower for far plate detection
    save_crops: bool = True,
    crops_dir: Optional[str] = None,
    debug_save_all_frames: bool = False,
    frame_interpolation_multiplier: int = 1,  # 1=off, 2=2x, 3=3x, 4=4x frames
    enable_ocr: bool = True,  # Enable OCR
    ocr_model_path: Optional[str] = None,  # Path to OCR model
    min_frames_to_confirm: int = 3,  # Minimum frames to confirm a plate
    debug_tracking: bool = True,  # Enable detailed tracking debug
) -> str:
    """
    Process video with YOLO detection + tracking + OCR.
    Tracks unique plates and runs OCR on best frames.
    """
    # Initialize OCR if available and enabled
    ocr_model = None
    if enable_ocr and OCR_AVAILABLE:
        try:
            ocr_model = get_ocr_model(ocr_model_path, device)
            if ocr_model:
                if status_callback:
                    status_callback("✅ OCR model loaded")
        except Exception as e:
            print(f"[OCR] Failed to initialize: {e}")
            if status_callback:
                status_callback(f"⚠️ OCR disabled: {e}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Direct MP4 output - try multiple codecs for best compatibility
    final_out = os.path.splitext(output_path)[0] + ".mp4"
    
    # Try codecs in order of preference
    codecs_to_try = ['avc1', 'h264', 'H264', 'X264', 'mp4v']
    out = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(final_out, fourcc, fps, (w, h))
            if out.isOpened():
                if status_callback:
                    status_callback(f"Using codec: {codec}")
                break
            else:
                out.release()
        except:
            continue
    
    if out is None or not out.isOpened():
        raise ValueError(f"Failed to open video writer with any codec. Install FFmpeg: sudo apt install ffmpeg")

    # Setup crop saving
    if save_crops and crops_dir:
        os.makedirs(crops_dir, exist_ok=True)

    processed = 0
    seen_track_ids: Set[int] = set()
    unique_plates_count = 0
    saved_crops: Dict[int, str] = {}  # track_id -> crop_path

    def _stopped() -> bool:
        if should_stop is None:
            return False
        if callable(should_stop):
            try:
                return bool(should_stop())
            except Exception:
                return False
        return bool(getattr(should_stop, "value", False))

    # Initialize tracker
    tracker = SimpleTracker(iou_threshold=0.3, max_lost_frames=30)
    frame_idx = 0
    
    # For frame interpolation
    prev_frame = None
    frames_to_process = []
    
    # Time tracking
    start_time = time.time()
    detection_time = 0.0
    ocr_time = 0.0
    
    while cap.isOpened():
        if _stopped():
            if status_callback:
                status_callback("⏹️ Stopped by user.")
            break

        ret, current_frame = cap.read()
        if not ret:
            break

        # Prepare frames to process (original + interpolated if enabled)
        frames_to_process = []
        
        # Frame interpolation: create multiple blended frames between prev and current
        if frame_interpolation_multiplier > 1 and prev_frame is not None:
            # Generate (multiplier - 1) intermediate frames
            for i in range(1, frame_interpolation_multiplier):
                # Calculate blend weight: i/multiplier
                alpha = i / frame_interpolation_multiplier
                # Linear interpolation between prev and current
                interpolated = cv2.addWeighted(prev_frame, 1.0 - alpha, current_frame, alpha, 0)
                frames_to_process.append(interpolated)
        
        # Always process the original frame last
        frames_to_process.append(current_frame)
        
        # Process all frames (interpolated + original)
        for frame_to_process in frames_to_process:
            frame = frame_to_process.copy()
            frame_copy = frame.copy()
            
            # YOLO detection
            t0 = time.perf_counter()
            results = model(frame, device=device,
                           conf=confidence_threshold, imgsz=imgsz, verbose=False)
            det_ms = (time.perf_counter() - t0) * 1000.0
            detection_time += det_ms / 1000.0  # Convert to seconds

            # Prepare detections for tracker
            detections = []
            yolo_detections_count = 0
            rejected_detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    yolo_detections_count += 1
                    conf = float(box.conf.cpu().numpy()[0])
                    x1_raw, y1_raw, x2_raw, y2_raw = map(int, box.xyxy.cpu().numpy()[0])
                    bw_raw = x2_raw - x1_raw
                    bh_raw = y2_raw - y1_raw
                    area_raw = bw_raw * bh_raw
                    relative_area = area_raw / (w * h)

                    # Smart validation: Adaptive confidence based on size + shape validation
                    # This allows lower confidence for far plates, higher for close detections
                    if not is_valid_plate_detection((x1_raw, y1_raw, x2_raw, y2_raw), conf, w, h, confidence_threshold):
                        if debug_tracking and relative_area < 0.01:  # Debug small/far plates
                            rejected_detections.append(
                                f"Frame {frame_idx}: REJECTED (validation) - "
                                f"size={bw_raw}x{bh_raw} ({relative_area*100:.3f}%), conf={conf:.2f}"
                            )
                        continue

                    # Expand & clamp
                    margin = 5
                    x1 = max(0, x1_raw - margin)
                    y1 = max(0, y1_raw - margin)
                    x2 = min(w, x2_raw + margin)
                    y2 = min(h, y2_raw + margin)

                    bw, bh = (x2 - x1), (y2 - y1)
                    if bw <= 0 or bh <= 0:
                        continue

                    # Additional basic size check (after margin expansion)
                    if bw < min_box_w or bh < min_box_h:
                        if debug_tracking and relative_area < 0.01:
                            rejected_detections.append(
                                f"Frame {frame_idx}: REJECTED (size) - "
                                f"size={bw}x{bh} < min {min_box_w}x{min_box_h}, conf={conf:.2f}"
                            )
                        continue

                    # Crop for tracker
                    crop = frame_copy[y1:y2, x1:x2].copy()
                    detections.append(((x1, y1, x2, y2), conf, crop))
                    
                    # Debug: Log accepted small/far plates
                    if debug_tracking and relative_area < 0.01:
                        print(f"Frame {frame_idx}: ✅ ACCEPTED - "
                              f"size={bw}x{bh} ({relative_area*100:.3f}%), conf={conf:.2f}")
            
            # Debug: Show rejected detections for small plates
            if debug_tracking and rejected_detections:
                for msg in rejected_detections:
                    print(msg)

            # Update tracker with frame dimensions for completeness checking
            prev_track_count = len(tracker.tracks)
            active_tracks = tracker.update(detections, w, h, debug=debug_tracking, frame_idx=frame_idx)
            new_track_count = len(tracker.tracks)
            
            # Debug: Track lifecycle
            if debug_tracking and new_track_count > prev_track_count:
                # New tracks created
                for track in tracker.tracks:
                    if track.frames_tracked == 1:  # Newly created
                        x1, y1, x2, y2 = track.bbox
                        area = (x2-x1) * (y2-y1)
                        rel_area = area / (w * h)
                        adaptive_min = get_adaptive_min_frames(track.bbox, w, h, min_frames_to_confirm)
                        print(f"Frame {frame_idx}: 🆕 NEW TRACK ID={track.track_id} - "
                              f"size={x2-x1}x{y2-y1} ({rel_area*100:.3f}%), "
                              f"conf={track.confidence:.2f}, "
                              f"needs {adaptive_min} frame(s) to confirm")

            # Draw tracks on frame
            elapsed = time.time() - start_time
            cv2.putText(frame, f"{w}x{h} | det {det_ms:.1f} ms | elapsed {elapsed:.1f}s", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            for track in active_tracks:
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.frames_tracked <= 3 else (255, 150, 0)
                thickness = 3 if track.frames_tracked <= 3 else 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Display label with OCR result if available
                if track.plate_number:
                    label = f"ID:{track.track_id} | {track.plate_number}"
                else:
                    label = f"ID:{track.track_id} ({track.confidence:.2f})"
                
                cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Run OCR on confirmed tracks (filters out false detections)
                # IMPORTANT: We run OCR every N frames to update as best frame improves
                # Use adaptive min_frames - far plates need fewer frames
                adaptive_min = get_adaptive_min_frames(track.bbox, w, h, min_frames_to_confirm)
                
                # Debug: Print when track is confirmed (especially for far plates)
                if track.frames_tracked == adaptive_min:
                    x1, y1, x2, y2 = track.bbox
                    plate_area = (x2-x1) * (y2-y1)
                    relative_area = plate_area / (w * h)
                    print(f"[Track Confirmed] ID={track.track_id}, frames={track.frames_tracked}/{adaptive_min}, "
                          f"size={x2-x1}x{y2-y1} ({relative_area*100:.3f}%), conf={track.confidence:.2f}")
                
                if ocr_model and track.frames_tracked >= adaptive_min and track.best_frame_crop is not None:
                    # Re-run OCR every 10 frames OR if no OCR yet (to get latest best frame)
                    should_run_ocr = (
                        not track.plate_number or  # First time
                        track.frames_tracked % 10 == 0  # Every 10 frames to update with better quality
                    )
                    
                    if should_run_ocr:
                        try:
                            ocr_t0 = time.perf_counter()
                            plate_text = ocr_model.predict(track.best_frame_crop, format_output=True)
                            ocr_time += (time.perf_counter() - ocr_t0)
                            track.plate_number = plate_text
                        except Exception as e:
                            print(f"[OCR] Error on track {track.track_id}: {e}")
                
                # Save crops
                if save_crops and crops_dir:
                    # DEBUG MODE: Save ALL frames of each track for comparison
                    if debug_save_all_frames:
                        track_debug_dir = os.path.join(crops_dir, f"track_{track.track_id}_all_frames")
                        os.makedirs(track_debug_dir, exist_ok=True)
                        
                        # Get current frame crop
                        current_crop = frame_copy[y1:y2, x1:x2].copy()
                        current_area = (x2 - x1) * (y2 - y1)
                        
                        # Calculate quality for this frame
                        frame_quality = calculate_plate_quality_score(current_crop, (x1, y1, x2, y2), track.confidence, w, h)
                        
                        # Calculate individual metrics for debugging
                        completeness = calculate_completeness_score((x1, y1, x2, y2), w, h, current_crop)
                        sharpness = calculate_sharpness(current_crop)
                        
                        # Save with metadata in filename including quality score and completeness
                        debug_filename = f"f{frame_idx:05d}_Q{frame_quality:.3f}_C{completeness:.2f}_S{sharpness:.0f}_conf{track.confidence:.2f}.jpg"
                        debug_path = os.path.join(track_debug_dir, debug_filename)
                        
                        # Add overlay with info
                        crop_annotated = current_crop.copy()
                        if crop_annotated.shape[0] > 20 and crop_annotated.shape[1] > 50:
                            cv2.putText(crop_annotated, f"Q:{frame_quality:.3f} C:{completeness:.2f} S:{sharpness:.0f}", 
                                       (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            cv2.putText(crop_annotated, f"Frame:{frame_idx} Conf:{track.confidence:.2f} ID:{track.track_id}", 
                                       (5, crop_annotated.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.imwrite(debug_path, crop_annotated)
                    
                    # NORMAL MODE: Save only best frame (continuously updates)
                    # Only save if track is confirmed (use adaptive min_frames)
                    adaptive_min = get_adaptive_min_frames(track.bbox, w, h, min_frames_to_confirm)
                    if track.frames_tracked >= adaptive_min and track.best_frame_crop is not None:
                        crop_path = os.path.join(crops_dir, f"track_{track.track_id}_BEST.jpg")
                        
                        # Add text overlay showing it's the best frame with quality metrics
                        crop_with_info = track.best_frame_crop.copy()
                        h_crop, w_crop = crop_with_info.shape[:2]
                        if h_crop > 20 and w_crop > 80:
                            # Show OCR result if available
                            if track.plate_number:
                                cv2.putText(crop_with_info, f"OCR: {track.plate_number}", 
                                           (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(crop_with_info, f"Quality: {track.best_quality_score:.3f} ID:{track.track_id}", 
                                           (5, h_crop - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            else:
                                # Show quality score (most important metric)
                                cv2.putText(crop_with_info, f"BEST Quality: {track.best_quality_score:.3f}", 
                                           (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                cv2.putText(crop_with_info, f"Conf:{track.best_confidence:.2f} ID:{track.track_id}", 
                                           (5, h_crop - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.imwrite(crop_path, crop_with_info)
                        
                        # Also save OCR result to text file
                        if track.plate_number:
                            txt_path = os.path.join(crops_dir, f"track_{track.track_id}_OCR.txt")
                            with open(txt_path, 'w') as f:
                                f.write(track.plate_number)
                        
                        saved_crops[track.track_id] = crop_path

            unique_plates_count = len(tracker.get_confirmed_tracks_adaptive(base_min_frames=min_frames_to_confirm))
            
            cv2.putText(frame, f"Tracked: {len(active_tracks)} | Unique: {unique_plates_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out.write(frame)
            processed += 1
            frame_idx += 1
        
        # Store current frame for next interpolation
        prev_frame = current_frame.copy()

        if progress_callback and total:
            progress_callback(min(int(processed * 100 / total), 100))
        if status_callback and (processed % max(int(fps), 1) == 0):
            status_callback(
                f"Processed {processed}/{total or '?'} frames | Unique plates: {unique_plates_count}"
            )

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # FINAL OCR PASS: Run OCR on all tracks one final time with their absolute best frames
    if ocr_model and save_crops and crops_dir:
        if status_callback:
            status_callback("🔤 Running final OCR on best frames...")
        
        confirmed_tracks = tracker.get_confirmed_tracks_adaptive(base_min_frames=min_frames_to_confirm)
        ocr_count = 0
        
        for track in confirmed_tracks:
            if track.best_frame_crop is not None:
                try:
                    # Final OCR on absolute best frame
                    ocr_t0 = time.perf_counter()
                    plate_text = ocr_model.predict(track.best_frame_crop, format_output=True)
                    ocr_time += (time.perf_counter() - ocr_t0)
                    track.plate_number = plate_text
                    ocr_count += 1
                    
                    # Update saved crop with final OCR result
                    crop_path = os.path.join(crops_dir, f"track_{track.track_id}_BEST.jpg")
                    crop_with_info = track.best_frame_crop.copy()
                    h_crop, w_crop = crop_with_info.shape[:2]
                    
                    if h_crop > 20 and w_crop > 80:
                        cv2.putText(crop_with_info, f"OCR: {plate_text}", 
                                   (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(crop_with_info, f"Quality: {track.best_quality_score:.3f} ID:{track.track_id}", 
                                   (5, h_crop - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    cv2.imwrite(crop_path, crop_with_info)
                    
                    # Save OCR to text file
                    txt_path = os.path.join(crops_dir, f"track_{track.track_id}_OCR.txt")
                    with open(txt_path, 'w') as f:
                        f.write(plate_text)
                        
                except Exception as e:
                    print(f"[OCR] Final pass error on track {track.track_id}: {e}")
    
    # UPDATE FINAL COUNT: Get actual number of confirmed tracks
    confirmed_tracks_final = tracker.get_confirmed_tracks_adaptive(base_min_frames=min_frames_to_confirm)
    unique_plates_count = len(confirmed_tracks_final)
    
    # Debug: Print all confirmed tracks
    if confirmed_tracks_final:
        print(f"\n[Final Count] {unique_plates_count} confirmed track(s):")
        for track in confirmed_tracks_final:
            x1, y1, x2, y2 = track.bbox
            area = (x2-x1) * (y2-y1)
            adaptive_min = get_adaptive_min_frames(track.bbox, tracker.frame_width, tracker.frame_height, min_frames_to_confirm)
            print(f"  - ID={track.track_id}: {track.frames_tracked} frames (min={adaptive_min}), "
                  f"size={x2-x1}x{y2-y1}, conf={track.confidence:.2f}, "
                  f"OCR={'✓' if track.plate_number else '✗'}")
    
    # Calculate total time and display timing information
    total_time = time.time() - start_time
    
    # Format time display
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.0f}s"
    
    # Create timing summary
    fps_achieved = processed / total_time if total_time > 0 else 0
    tracking_time = total_time - detection_time - ocr_time
    
    timing_msg = (
        f"⏱️ Total: {format_time(total_time)} | "
        f"Detection: {format_time(detection_time)} | "
        f"Tracking: {format_time(tracking_time)}"
    )
    
    if ocr_model and save_crops:
        timing_msg += f" | OCR: {format_time(ocr_time)}"
        if status_callback:
            status_callback(f"✅ Complete! {processed} frames, {unique_plates_count} plates, {ocr_count} OCR results")
            status_callback(timing_msg)
            status_callback(f"⚡ Performance: {fps_achieved:.1f} FPS")
    else:
        if status_callback:
            status_callback(f"✅ Complete! {processed} frames, {unique_plates_count} unique plates tracked")
            status_callback(timing_msg)
            status_callback(f"⚡ Performance: {fps_achieved:.1f} FPS")
    
    # Print detailed timing to console
    print("\n" + "="*60)
    print("⏱️  PROCESSING TIME BREAKDOWN")
    print("="*60)
    print(f"Total Processing Time:    {format_time(total_time)}")
    print(f"  - YOLO Detection:       {format_time(detection_time)} ({detection_time/total_time*100:.1f}%)")
    print(f"  - Tracking:             {format_time(tracking_time)} ({tracking_time/total_time*100:.1f}%)")
    if ocr_time > 0:
        print(f"  - OCR:                  {format_time(ocr_time)} ({ocr_time/total_time*100:.1f}%)")
    print(f"\nFrames Processed:         {processed}")
    print(f"Average FPS:              {fps_achieved:.2f}")
    print(f"Unique Plates Detected:   {unique_plates_count}")
    if ocr_count > 0:
        print(f"OCR Results:              {ocr_count}")
        print(f"Avg OCR Time/Plate:       {ocr_time/ocr_count*1000:.1f}ms")
    print("="*60 + "\n")
    
    return final_out

# ──────────────────────────────────────────────────────────────────────────────
# Webcam (OpenCV window)


def process_webcam(confidence_threshold: float = 0.40):
    cap = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            break
    if not cap or not cap.isOpened():
        raise ValueError("Could not open webcam with any available backend")

    window_name = "License Plate Tracking (Press 'q' to quit)"
    tracker = SimpleTracker(iou_threshold=0.3, max_lost_frames=15)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # YOLO detection
        results = model(frame, device=device, conf=confidence_threshold, imgsz=960, verbose=False)
        
        # Prepare detections for tracker
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                
                # Smart validation with adaptive confidence
                if not is_valid_plate_detection((x1, y1, x2, y2), conf, w, h, confidence_threshold):
                    continue
                
                margin = 5
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                bw, bh = (x2 - x1), (y2 - y1)
                if bw > 40 and bh > 18:
                    crop = frame_copy[y1:y2, x1:x2].copy()
                    detections.append(((x1, y1, x2, y2), conf, crop))
        
        # Update tracker with frame dimensions
        active_tracks = tracker.update(detections, w, h)
        
        # Draw tracks
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0) if track.frames_tracked <= 3 else (255, 150, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track.track_id} ({track.confidence:.2f})"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        unique_count = len(tracker.get_confirmed_tracks(min_frames=3))
        cv2.putText(frame, f"Tracked: {len(active_tracks)} | Unique: {unique_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        frame_idx += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────────────────────
# RTSP stream → frames (for Streamlit)


def process_rtsp_stream(
    rtsp_url: str,
    confidence_threshold: float = 0.40,
    imgsz: int = 960,
    min_box_w: int = 40,
    min_box_h: int = 18,
    process_every_n_frames: int = 2,  # Process every Nth frame for speed
    enable_processing: bool = True,  # NEW: Toggle processing on/off
    enable_ocr: bool = True,  # Enable OCR
    ocr_model_path: Optional[str] = None,  # Path to OCR model
    min_frames_to_confirm: int = 3,  # Minimum frames to confirm a plate
):
    # Initialize OCR if available and enabled
    ocr_model = None
    if enable_processing and enable_ocr and OCR_AVAILABLE:
        try:
            ocr_model = get_ocr_model(ocr_model_path, device)
            print("[RTSP] OCR enabled")
        except Exception as e:
            print(f"[RTSP] OCR initialization failed: {e}")
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")

    tracker = SimpleTracker(iou_threshold=0.3, max_lost_frames=30)
    frame_idx = 0
    active_tracks = []  # Cache tracks between frames

    while True:
        ok, frame = cap.read()
        if not ok:
            cv2.waitKey(1)
            continue
        
        # OPTIMIZATION: Flush old frames from buffer to reduce latency
        # Grab multiple frames quickly to get to latest
        if not enable_processing:
            # In view-only mode, skip ahead to latest frame
            for _ in range(2):  # Skip 2 frames to stay current
                cap.grab()

        h, w = frame.shape[:2]
        
        # OPTIMIZATION: Only run YOLO detection every N frames AND when processing enabled
        if enable_processing and (frame_idx % process_every_n_frames == 0):
            frame_copy = frame.copy()
            
            # YOLO detection
            results = model(frame, device=device, conf=confidence_threshold, imgsz=imgsz, verbose=False)
            
            # Prepare detections for tracker
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf.cpu().numpy()[0])
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    
                    # Smart validation with adaptive confidence
                    if not is_valid_plate_detection((x1, y1, x2, y2), conf, w, h, confidence_threshold):
                        continue
                    
                    margin = 5
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)
                    
                    bw, bh = (x2 - x1), (y2 - y1)
                    if bw > min_box_w and bh > min_box_h:
                        crop = frame_copy[y1:y2, x1:x2].copy()
                        detections.append(((x1, y1, x2, y2), conf, crop))
            
            # Update tracker with frame dimensions
            active_tracks = tracker.update(detections, w, h)
        
        # Draw tracks and info (only when processing is enabled)
        if enable_processing:
            # Draw tracks (even on skipped frames, use cached tracks)
            for track in active_tracks:
                x1, y1, x2, y2 = track.bbox
                color = (0, 255, 0) if track.frames_tracked <= 3 else (255, 150, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Run OCR on confirmed tracks
                adaptive_min = get_adaptive_min_frames(track.bbox, w, h, min_frames_to_confirm)
                if ocr_model and track.frames_tracked >= adaptive_min and track.best_frame_crop is not None:
                    # Run OCR every 10 frames OR if no OCR yet
                    should_run_ocr = (
                        not track.plate_number or
                        track.frames_tracked % 10 == 0
                    )
                    
                    if should_run_ocr:
                        try:
                            plate_text = ocr_model.predict(track.best_frame_crop, format_output=True)
                            track.plate_number = plate_text
                        except Exception as e:
                            print(f"[RTSP OCR] Error on track {track.track_id}: {e}")
                
                # Display label with OCR result if available
                if track.plate_number:
                    label = f"ID:{track.track_id} | {track.plate_number}"
                else:
                    label = f"ID:{track.track_id} ({track.confidence:.2f})"
                
                cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            unique_count = len(tracker.get_confirmed_tracks_adaptive(base_min_frames=min_frames_to_confirm))
            cv2.putText(frame, f"Tracked: {len(active_tracks)} | Unique: {unique_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # View-only mode indicator
            cv2.putText(frame, "VIEW ONLY - Click 'Toggle Processing' to enable detection", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        # OPTIMIZATION: Skip frames for display (Streamlit rendering is slow)
        # Display every 3rd frame in view-only for smoother experience
        display_skip = 3 if not enable_processing else 1
        
        if frame_idx % display_skip == 0:
            # OPTIMIZATION: Resize frame for faster network transmission to Streamlit
            # In view-only mode, use smaller size for even faster display
            if enable_processing:
                display_width = 1280
            else:
                display_width = 960  # Smaller in view-only mode for speed
            
            if w > display_width:
                scale = display_width / w
                display_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                display_frame = frame
            
            yield display_frame
        
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

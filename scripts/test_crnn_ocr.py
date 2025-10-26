#!/usr/bin/env python3
"""
CRNN OCR Inference Script
Test the trained model on individual images or test set
"""

import os
import torch
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import matplotlib.pyplot as plt

# Import model from training script
import sys
sys.path.append('/home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV')
from train_crnn_ocr import CRNN

# Configuration
IMG_HEIGHT = 64
IMG_WIDTH = 200
NUM_CLASSES = 11
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = '/home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV/checkpoints/best_model.pth'

# Preprocessing
preprocess = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
])

# Character mapping
idx_to_char = {i: str(i) for i in range(10)}


def load_model(checkpoint_path):
    """Load trained model"""
    model = CRNN(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        num_classes=NUM_CLASSES,
        hidden_size=256
    ).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"   Best accuracy: {checkpoint.get('best_seq_acc', 0)*100:.2f}%")
    
    return model


def decode_prediction(output):
    """Decode CTC output to text"""
    _, preds = output.max(2)  # (T, B)
    preds = preds.squeeze(1).cpu().numpy()  # (T,)
    
    # Greedy decode: remove blanks and duplicates
    decoded = []
    prev = -1
    for p in preds:
        if p != 10 and p != prev:  # 10 is blank
            decoded.append(idx_to_char.get(p, '?'))
        prev = p
    
    return ''.join(decoded)


def predict_image(model, image_path):
    """Predict license plate number from image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    img = preprocess(image=img)['image']
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    img = img.to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(img)  # (T, 1, num_classes)
    
    # Decode
    prediction = decode_prediction(output)
    
    return prediction


def visualize_prediction(image_path, prediction, ground_truth=None):
    """Show image with prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 3))
    plt.imshow(img)
    title = f"Prediction: {prediction}"
    if ground_truth:
        match = "✅" if prediction == ground_truth else "❌"
        title += f"\nGround Truth: {ground_truth} {match}"
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def test_on_directory(model, directory, max_samples=20):
    """Test model on directory of images"""
    image_paths = list(Path(directory).rglob('*.jpg'))[:max_samples]
    
    correct = 0
    total = 0
    
    print(f"\n🧪 Testing on {len(image_paths)} images from {directory}")
    print("="*70)
    
    for img_path in image_paths:
        # Ground truth from filename
        gt = img_path.stem.split('_')[0].zfill(11)
        
        # Predict
        pred = predict_image(model, str(img_path))
        
        if pred:
            # Pad/truncate to 11 digits
            pred = pred[:11].ljust(11, '0')
            
            match = "✅" if pred == gt else "❌"
            if pred == gt:
                correct += 1
            total += 1
            
            print(f"{match} GT: {gt} | Pred: {pred}")
    
    accuracy = correct / total * 100 if total > 0 else 0
    print("="*70)
    print(f"📊 Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    return accuracy


def main():
    print("="*60)
    print("🔍 CRNN OCR INFERENCE")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ Model not found: {CHECKPOINT_PATH}")
        print("   Train the model first: python3 train_crnn_ocr.py")
        return
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model(CHECKPOINT_PATH)
    
    # Test on sample images
    test_dir = '/home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV/recognition/test'
    
    if os.path.exists(test_dir):
        test_on_directory(model, test_dir, max_samples=20)
    else:
        print(f"\n⚠️  Test directory not found: {test_dir}")
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()

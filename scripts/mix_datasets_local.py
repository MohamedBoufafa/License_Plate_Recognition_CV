#!/usr/bin/env python3
"""
LOCAL DATA MIXING SCRIPT
Mix synthetic + real plates for OCR training (80/20 ratio)
Run: python3 mix_datasets_local.py
"""

import os
import random
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not installed. Install with: pip install torch torchvision albumentations opencv-python")
    TORCH_AVAILABLE = False

# ============= CONFIGURATION =============
BASE_DIR = '/home/moamed/Desktop/3cs/projects/slm2/06_License_Plate_Recognition_CV'
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'synthetic_plates')
REAL_DIR = os.path.join(BASE_DIR, 'recognition')
OUTPUT_DIR = os.path.join(BASE_DIR, 'ocr_training_data')

SYNTHETIC_RATIO = 0.80  # 80% synthetic, 20% real
TARGET_HEIGHT = 64
TARGET_WIDTH = 200
BATCH_SIZE = 32

random.seed(42)
np.random.seed(42)

# ============= FUNCTIONS =============

def scan_dataset(directory):
    """Scan for images with valid license plate labels."""
    print(f"\n🔍 Scanning: {directory}")
    images, labels = [], []
    skipped = 0
    
    for img_path in Path(directory).rglob('*.jpg'):
        label = img_path.stem
        
        # Remove suffixes like _1, _2, etc.
        label = label.split('_')[0]
        
        # Accept 10 or 11 digits (some plates missing leading zero)
        if len(label) >= 10 and len(label) <= 11 and label.isdigit():
            # Pad to 11 digits if needed (add leading zero)
            label = label.zfill(11)
            images.append(str(img_path))
            labels.append(label)
        else:
            skipped += 1
    
    print(f"   Found: {len(images):,} plates")
    if skipped > 0:
        print(f"   Skipped: {skipped} (invalid format)")
    return images, labels


def create_augmentation_pipelines():
    """Create augmentation pipelines for different sources."""
    
    # Heavy (for real images)
    heavy = A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.7),
        A.Perspective(scale=(0.02, 0.05), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.6),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
    ])
    
    # Medium (for synthetic)
    medium = A.Compose([
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
    ])
    
    # None (for val/test)
    none = A.Compose([
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
    ])
    
    return heavy, medium, none


def mix_datasets(synth_imgs, synth_labs, real_imgs, real_labs, ratio=0.8):
    """Mix synthetic and real datasets."""
    total_real = len(real_imgs)
    synth_needed = int(total_real * ratio / (1 - ratio))
    synth_needed = min(synth_needed, len(synth_imgs))
    
    print(f"\n📊 Mixing datasets:")
    print(f"   Real:      {total_real:,}")
    print(f"   Synthetic: {synth_needed:,}")
    print(f"   Total:     {total_real + synth_needed:,}")
    print(f"   Ratio:     {ratio*100:.0f}% synth, {(1-ratio)*100:.0f}% real")
    
    # Sample synthetic
    if synth_needed < len(synth_imgs):
        indices = random.sample(range(len(synth_imgs)), synth_needed)
        synth_imgs = [synth_imgs[i] for i in indices]
        synth_labs = [synth_labs[i] for i in indices]
    
    # Combine
    all_imgs = synth_imgs + real_imgs
    all_labs = synth_labs + real_labs
    all_srcs = ['synthetic'] * len(synth_imgs) + ['real'] * len(real_imgs)
    
    # Shuffle
    combined = list(zip(all_imgs, all_labs, all_srcs))
    random.shuffle(combined)
    all_imgs, all_labs, all_srcs = zip(*combined)
    
    return list(all_imgs), list(all_labs), list(all_srcs)


def split_dataset(images, labels, sources, train_r=0.7, val_r=0.15):
    """Split into train/val/test."""
    total = len(images)
    train_end = int(total * train_r)
    val_end = train_end + int(total * val_r)
    
    splits = {
        'train': (images[:train_end], labels[:train_end], sources[:train_end]),
        'val': (images[train_end:val_end], labels[train_end:val_end], sources[train_end:val_end]),
        'test': (images[val_end:], labels[val_end:], sources[val_end:])
    }
    
    print(f"\n✂️ Dataset splits:")
    for name, (imgs, _, srcs) in splits.items():
        synth_count = srcs.count('synthetic')
        real_count = srcs.count('real')
        print(f"   {name.capitalize():5s}: {len(imgs):,} (synth: {synth_count}, real: {real_count})")
    
    return splits


def visualize_samples(images, labels, sources, augmentation, num_samples=12):
    """Show sample images with augmentation."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    fig.suptitle('Sample Images (After Augmentation)', fontsize=16, fontweight='bold')
    
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    for ax, idx in zip(axes.flat, indices):
        img = cv2.imread(images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if augmentation:
            img = augmentation(image=img)['image']
        
        ax.imshow(img)
        color = 'blue' if sources[idx] == 'synthetic' else 'red'
        ax.set_title(f"{labels[idx]}\n({sources[idx]})", fontsize=9, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_augmented.png'), dpi=150)
    print(f"\n✅ Saved sample visualization to: {OUTPUT_DIR}/sample_augmented.png")
    plt.show()


def save_dataset_info(splits, output_dir):
    """Save dataset information to JSON."""
    info = {
        'synthetic_ratio': SYNTHETIC_RATIO,
        'target_size': [TARGET_HEIGHT, TARGET_WIDTH],
        'batch_size': BATCH_SIZE,
        'num_classes': 11,
        'char_to_idx': {str(i): i for i in range(10)},
        'splits': {}
    }
    
    for name, (imgs, _, srcs) in splits.items():
        info['splits'][name] = {
            'total': len(imgs),
            'synthetic': srcs.count('synthetic'),
            'real': srcs.count('real')
        }
    
    json_path = os.path.join(output_dir, 'dataset_info.json')
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Dataset info saved to: {json_path}")


# ============= MAIN =============

def main():
    print("="*60)
    print("📊 LOCAL DATA MIXING FOR OCR TRAINING")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Scan datasets
    print("\n🔍 STEP 1: Scanning datasets...")
    synth_imgs, synth_labs = scan_dataset(SYNTHETIC_DIR)
    real_imgs, real_labs = scan_dataset(REAL_DIR)
    
    if len(synth_imgs) == 0:
        print(f"\n⚠️  WARNING: No synthetic plates found in {SYNTHETIC_DIR}")
        print(f"   Continuing with REAL plates only...")
        print(f"   To generate synthetic plates:")
        print(f"   1. Open Jupyter: jupyter notebook")
        print(f"   2. Run: synthetic_plate_generator.ipynb")
        
        # Use only real plates
        if len(real_imgs) == 0:
            print(f"\n❌ ERROR: No plates found at all!")
            return
        
        # Skip mixing, use real plates directly
        all_imgs, all_labs, all_srcs = real_imgs, real_labs, ['real'] * len(real_imgs)
    elif len(real_imgs) == 0:
        print(f"\n⚠️  WARNING: No real plates found in {REAL_DIR}")
        print(f"   Continuing with SYNTHETIC plates only...")
        all_imgs, all_labs, all_srcs = synth_imgs, synth_labs, ['synthetic'] * len(synth_imgs)
    else:
        # Mix datasets normally
        print("\n🔀 STEP 2: Mixing datasets (80/20)...")
        all_imgs, all_labs, all_srcs = mix_datasets(
            synth_imgs, synth_labs,
            real_imgs, real_labs,
            ratio=SYNTHETIC_RATIO
        )
    
    # Split
    print("\n✂️ STEP 3: Splitting into train/val/test...")
    splits = split_dataset(all_imgs, all_labs, all_srcs)
    
    # Create augmentation
    print("\n🎨 STEP 4: Creating augmentation pipelines...")
    heavy_aug, medium_aug, no_aug = create_augmentation_pipelines()
    print("   ✅ Heavy (real): Rotation, Perspective, Blur, Noise")
    print("   ✅ Medium (synth): Light rotation, brightness")
    print("   ✅ None (val/test): Just resize")
    
    # Visualize
    print("\n👀 STEP 5: Visualizing samples...")
    train_imgs, train_labs, train_srcs = splits['train']
    visualize_samples(train_imgs, train_labs, train_srcs, heavy_aug)
    
    # Save info
    print("\n💾 STEP 6: Saving dataset info...")
    save_dataset_info(splits, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nDataset ready for training:")
    print(f"  📁 Location: {OUTPUT_DIR}")
    print(f"  🖼️  Image size: {TARGET_HEIGHT}×{TARGET_WIDTH}")
    print(f"  📊 Train: {len(splits['train'][0]):,} images")
    print(f"  📊 Val:   {len(splits['val'][0]):,} images")
    print(f"  📊 Test:  {len(splits['test'][0]):,} images")
    print(f"\n🚀 Next step: Build CRNN model and start training!")
    print("="*60)
    
    if not TORCH_AVAILABLE:
        print("\n⚠️  Note: Install PyTorch to create DataLoaders for training:")
        print("   pip install torch torchvision albumentations opencv-python")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Local OCR Inference Script
Load trained CRNN model and predict license plate numbers
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from pathlib import Path
import argparse

# ============= MODEL DEFINITION =============

class CRNN(nn.Module):
    """CRNN: CNN + Bidirectional LSTM + CTC for OCR"""
    
    def __init__(self, img_height=64, img_width=200, num_classes=11, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        
        # LSTM
        self.cnn_output_height = img_height // 16
        self.cnn_output_width = img_width // 4
        self.rnn_input_size = 512 * self.cnn_output_height
        
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully Connected
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)
        
        # Reshape for RNN
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.reshape(b, w, c * h)
        
        # RNN
        rnn_out, _ = self.rnn(conv)
        
        # FC
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)
        
        return output


# ============= INFERENCE CLASS =============

class LicensePlateOCR:
    """Easy-to-use OCR inference class"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize OCR model
        
        Args:
            model_path: Path to best_model.pth
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"📦 Loading model from: {model_path}")
        self.model = CRNN(
            img_height=64,
            img_width=200,
            num_classes=11,
            hidden_size=256
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Preprocessing
        self.preprocess = A.Compose([
            A.Resize(height=64, width=200),
        ])
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: numpy array (BGR or RGB)
        
        Returns:
            torch tensor ready for model
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = self.preprocess(image=image)['image']
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
    
    def decode_prediction(self, output):
        """
        Decode CTC output to text
        
        Args:
            output: Model output tensor
        
        Returns:
            Decoded string
        """
        _, preds = output.max(2)
        preds = preds.squeeze(1).cpu().numpy()
        
        # Greedy CTC decode
        decoded = []
        prev = -1
        for p in preds:
            if p != 10 and p != prev:  # 10 is CTC blank
                decoded.append(str(p))
            prev = p
        
        return ''.join(decoded)
    
    def predict(self, image, format_output=True):
        """
        Predict license plate number from image
        
        Args:
            image: numpy array or path to image
            format_output: If True, clean up the prediction
        
        Returns:
            Predicted license plate number as string
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
        
        # Preprocess
        img_tensor = self.preprocess_image(image)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Decode
        prediction = self.decode_prediction(output)
        
        # Clean up prediction if requested
        if format_output and len(prediction) > 0:
            # Algerian plates: 10 digits (5-3-2) or 11 digits (6-3-2)
            # Don't force padding - keep what model predicts
            
            # Remove leading zeros ONLY if we have more than expected digits
            # (model sometimes adds extra 0 at start)
            if len(prediction) > 11:
                # Too many digits, remove leading zeros
                prediction = prediction.lstrip('0')
                # But keep at least 10 digits
                if len(prediction) < 10:
                    prediction = '0' + prediction
            
            # If still too long, take last 11 digits (most recent/rightmost)
            if len(prediction) > 11:
                prediction = prediction[-11:]
            
            # Validate: should be 10 or 11 digits
            if len(prediction) < 10:
                # Too short - model might have missed digits
                # Pad on right to at least 10 digits
                prediction = prediction.ljust(10, '0')
        
        return prediction
    
    def predict_batch(self, images, format_output=True):
        """
        Predict multiple images at once
        
        Args:
            images: List of numpy arrays or paths
            format_output: If True, pad to 11 digits
        
        Returns:
            List of predictions
        """
        predictions = []
        for image in images:
            pred = self.predict(image, format_output)
            predictions.append(pred)
        return predictions


# ============= COMMAND LINE INTERFACE =============

def main():
    parser = argparse.ArgumentParser(description='License Plate OCR Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to best_model.pth')
    parser.add_argument('--image', type=str,
                        help='Path to single image')
    parser.add_argument('--folder', type=str,
                        help='Path to folder of images')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--show', action='store_true',
                        help='Show images with predictions')
    
    args = parser.parse_args()
    
    # Initialize OCR
    ocr = LicensePlateOCR(args.model, device=args.device)
    
    if args.image:
        # Single image
        print(f"\n🔍 Processing: {args.image}")
        prediction = ocr.predict(args.image)
        print(f"📊 Prediction: {prediction}")
        
        if args.show:
            img = cv2.imread(args.image)
            cv2.putText(img, prediction, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prediction', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.folder:
        # Batch processing
        print(f"\n🔍 Processing folder: {args.folder}")
        images = list(Path(args.folder).rglob('*.jpg'))
        images.extend(list(Path(args.folder).rglob('*.png')))
        
        print(f"Found {len(images)} images")
        
        for img_path in images:
            prediction = ocr.predict(str(img_path))
            print(f"{img_path.name}: {prediction}")
    
    else:
        print("❌ Please provide --image or --folder")
        parser.print_help()


if __name__ == "__main__":
    main()

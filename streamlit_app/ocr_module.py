"""
OCR Module for License Plate Recognition
Loads trained CRNN model and performs OCR on detected plates
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A


# ============= CRNN MODEL =============

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


# ============= OCR CLASS =============

class PlateOCR:
    """Easy-to-use OCR class for license plate recognition"""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize OCR model
        
        Args:
            model_path: Path to best_model.pth
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"OCR model not found at: {model_path}")
        
        # Load model
        print(f"[OCR] Loading model from: {model_path}")
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
        
        print(f"[OCR] Model loaded successfully on {self.device}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: numpy array (BGR or RGB)
        
        Returns:
            torch tensor ready for model
        """
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            return None
        
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
            image: numpy array (BGR format from OpenCV)
            format_output: If True, clean up the prediction
        
        Returns:
            Predicted license plate number as string
        """
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image)
            if img_tensor is None:
                return ""
            
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
        except Exception as e:
            print(f"[OCR] Error during prediction: {e}")
            return ""


# ============= INITIALIZATION =============

# Singleton OCR instance
_ocr_instance = None

def get_ocr_model(model_path=None, device='cuda'):
    """
    Get or create OCR model instance (singleton pattern)
    
    Args:
        model_path: Path to model file (only needed on first call)
        device: 'cuda' or 'cpu'
    
    Returns:
        PlateOCR instance or None if model not available
    """
    global _ocr_instance
    
    if _ocr_instance is None:
        if model_path is None:
            # Try default paths
            default_paths = [
                "../best_model.pth",
                "../best_ocr_model.pth",
                "best_model.pth",
                "best_ocr_model.pth",
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            try:
                _ocr_instance = PlateOCR(model_path, device)
            except Exception as e:
                print(f"[OCR] Failed to load model: {e}")
                _ocr_instance = None
        else:
            print(f"[OCR] Model file not found. OCR disabled.")
            print(f"[OCR] Place 'best_model.pth' in project root to enable OCR.")
            _ocr_instance = None
    
    return _ocr_instance

#!/usr/bin/env python3
"""
Example Usage of OCR Inference
Shows different ways to use the trained model
"""

from ocr_inference import LicensePlateOCR
import cv2
from pathlib import Path

# ============= EXAMPLE 1: Single Image =============

def example_single_image():
    """Predict on a single image"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*60)
    
    # Initialize OCR
    ocr = LicensePlateOCR(
        model_path='best_model.pth',
        device='cuda'  # or 'cpu'
    )
    
    # Predict
    image_path = 'recognition/test/00012345678.jpg'  # Change this
    prediction = ocr.predict(image_path)
    
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")


# ============= EXAMPLE 2: From NumPy Array =============

def example_numpy_array():
    """Predict from already loaded image"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Predict from NumPy Array")
    print("="*60)
    
    ocr = LicensePlateOCR('best_model.pth', device='cuda')
    
    # Load image with OpenCV
    img = cv2.imread('recognition/test/00012345678.jpg')
    
    # Predict
    prediction = ocr.predict(img)
    
    print(f"Prediction: {prediction}")


# ============= EXAMPLE 3: Batch Processing =============

def example_batch():
    """Process multiple images"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    ocr = LicensePlateOCR('best_model.pth', device='cuda')
    
    # Get all images from folder
    folder = Path('recognition/test')
    images = list(folder.glob('*.jpg'))[:10]  # First 10 images
    
    # Predict batch
    predictions = ocr.predict_batch(images)
    
    # Show results
    for img_path, pred in zip(images, predictions):
        gt = img_path.stem.split('_')[0].zfill(11)
        match = "✅" if pred == gt else "❌"
        print(f"{match} GT: {gt} | Pred: {pred}")


# ============= EXAMPLE 4: Integration with YOLO =============

def example_yolo_integration():
    """Use OCR with YOLO detection"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Integration with YOLO Detection")
    print("="*60)
    
    ocr = LicensePlateOCR('best_model.pth', device='cuda')
    
    # Simulate YOLO detection (replace with actual YOLO)
    # Assume YOLO detected a plate and cropped it
    full_image = cv2.imread('path/to/full_image.jpg')
    
    # Example crop coordinates from YOLO
    x1, y1, x2, y2 = 100, 50, 300, 100  # Replace with YOLO output
    cropped_plate = full_image[y1:y2, x1:x2]
    
    # Run OCR on cropped plate
    plate_number = ocr.predict(cropped_plate)
    
    print(f"Detected Plate: {plate_number}")
    
    # Draw on original image
    cv2.rectangle(full_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(full_image, plate_number, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite('result_with_ocr.jpg', full_image)
    print("✅ Result saved to: result_with_ocr.jpg")


# ============= EXAMPLE 5: Real-time Webcam =============

def example_realtime():
    """Real-time OCR on webcam"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Real-time Webcam OCR")
    print("="*60)
    print("Press 'q' to quit")
    
    ocr = LicensePlateOCR('best_model.pth', device='cuda')
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # For demo, use entire frame
        # In real app, crop to detected plate first
        try:
            prediction = ocr.predict(frame)
            
            # Display prediction
            cv2.putText(frame, f"Plate: {prediction}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass
        
        cv2.imshow('Real-time OCR', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ============= EXAMPLE 6: Test Accuracy =============

def example_test_accuracy():
    """Test model accuracy on folder"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Test Accuracy on Dataset")
    print("="*60)
    
    ocr = LicensePlateOCR('best_model.pth', device='cuda')
    
    # Get test images
    test_folder = Path('recognition/test')
    images = list(test_folder.glob('*.jpg'))[:50]  # First 50
    
    correct = 0
    total = 0
    
    print(f"Testing on {len(images)} images...\n")
    
    for img_path in images:
        # Ground truth from filename
        gt = img_path.stem.split('_')[0].zfill(11)
        
        # Predict
        pred = ocr.predict(str(img_path))
        
        # Check
        match = "✅" if pred == gt else "❌"
        if pred == gt:
            correct += 1
        total += 1
        
        print(f"{match} GT: {gt} | Pred: {pred}")
    
    accuracy = correct / total * 100
    print(f"\n{'='*60}")
    print(f"📊 Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"{'='*60}")


# ============= RUN EXAMPLES =============

if __name__ == "__main__":
    import sys
    
    print("\n🚀 OCR Inference Examples")
    print("="*60)
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == '1':
            example_single_image()
        elif example == '2':
            example_numpy_array()
        elif example == '3':
            example_batch()
        elif example == '4':
            example_yolo_integration()
        elif example == '5':
            example_realtime()
        elif example == '6':
            example_test_accuracy()
        else:
            print("❌ Unknown example number")
    else:
        print("\nAvailable examples:")
        print("  1 - Single image prediction")
        print("  2 - Predict from NumPy array")
        print("  3 - Batch processing")
        print("  4 - YOLO integration")
        print("  5 - Real-time webcam")
        print("  6 - Test accuracy")
        print("\nUsage: python example_ocr_usage.py [1-6]")
        print("\nRunning Example 3 (Batch Processing)...")
        example_batch()

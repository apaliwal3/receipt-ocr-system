import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import pytesseract
from PIL import Image
from image_preprocessing import ImagePreprocessor
import re

def test_all_ocr_methods(image_path):
    print("🧾 Testing OCR with different preprocessing approaches")
    print("=" * 60)
    
    # Test 1: No preprocessing (original image)
    print("\n1️⃣ NO PREPROCESSING (Original Image)")
    try:
        original = cv2.imread(image_path)
        if original is not None:
            # Convert to grayscale for OCR but don't process further
            gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            original_text = pytesseract.image_to_string(Image.fromarray(gray_original))
            
            print("Raw text (first 300 chars):")
            print(original_text[:300] + "..." if len(original_text) > 300 else original_text)
            
            # Find money amounts
            original_amounts = re.findall(r'[£$]?\d+\.\d{2}', original_text)
            print(f"Found amounts: {original_amounts}")
            
    except Exception as e:
        print(f"Error with original: {e}")
        original_amounts = []
    
    print("\n" + "="*60)
    
    # Test 2: Minimal preprocessing
    print("\n2️⃣ MINIMAL PREPROCESSING (Grayscale + Resize)")
    try:
        preprocessor = ImagePreprocessor()
        minimal_processed = preprocessor.preprocess_receipt_minimal(image_path)
        
        minimal_text = pytesseract.image_to_string(Image.fromarray(minimal_processed))
        print("Minimal processed text (first 300 chars):")
        print(minimal_text[:300] + "..." if len(minimal_text) > 300 else minimal_text)
        
        minimal_amounts = re.findall(r'[£$]?\d+\.\d{2}', minimal_text)
        print(f"Found amounts: {minimal_amounts}")
        
    except Exception as e:
        print(f"Error with minimal preprocessing: {e}")
        minimal_amounts = []
    
    print("\n" + "="*60)
    
    # Test 3: Gentle preprocessing
    print("\n3️⃣ GENTLE PREPROCESSING (With Smart Enhancement)")
    try:
        gentle_processed = preprocessor.preprocess_receipt(image_path)
        
        gentle_text = pytesseract.image_to_string(Image.fromarray(gentle_processed))
        print("Gentle processed text (first 300 chars):")
        print(gentle_text[:300] + "..." if len(gentle_text) > 300 else gentle_text)
        
        gentle_amounts = re.findall(r'[£$]?\d+\.\d{2}', gentle_text)
        print(f"Found amounts: {gentle_amounts}")
        
    except Exception as e:
        print(f"Error with gentle preprocessing: {e}")
        gentle_amounts = []
    
    print("\n" + "="*60)
    print("🔍 COMPARISON SUMMARY:")
    print(f"Original:           {len(original_amounts)} amounts found")
    print(f"Minimal processing: {len(minimal_amounts)} amounts found") 
    print(f"Gentle processing:  {len(gentle_amounts)} amounts found")
    
    # Determine best method
    results = [
        ("Original", len(original_amounts)),
        ("Minimal", len(minimal_amounts)), 
        ("Gentle", len(gentle_amounts))
    ]
    
    best_method = max(results, key=lambda x: x[1])
    print(f"\n🏆 Best method: {best_method[0]} (found {best_method[1]} amounts)")
    
    # Return the best text for further parsing
    if best_method[0] == "Original":
        return original_text
    elif best_method[0] == "Minimal":
        return minimal_text
    else:
        return gentle_text

if __name__ == "__main__":
    # Make sure directories exist
    os.makedirs('data/processed_images', exist_ok=True)
    os.makedirs('data/test_outputs', exist_ok=True)
    
    best_text = test_all_ocr_methods("../data/sample_receipts/sample_receipt_2.webp")
    
    print("\n" + "="*60)
    print("💾 BEST OCR RESULT:")
    print(best_text)
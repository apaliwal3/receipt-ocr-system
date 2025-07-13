import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import pytesseract
from PIL import Image
from image_preprocessing import ImagePreprocessor
import re

def test_smart_ocr(image_path):
    print("Testing Smart OCR Pipeline")
    print("=" * 60)
    print(f"Testing image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return ""
    
    preprocessor = ImagePreprocessor()
    
    # Initialize variables
    original_text = ""
    original_analysis = {}
    smart_text = ""
    smart_analysis = {}
    
    # Test 1: Original image (just grayscale + resize)
    print("\nORIGINAL IMAGE (Minimal processing)")
    try:
        original = cv2.imread(image_path)
        if original is not None:
            # Just convert to grayscale and resize if needed
            gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
            resized_original = preprocessor.resize_if_needed(gray_original)
            
            original_text = pytesseract.image_to_string(Image.fromarray(resized_original))
            
            print("OCR Text Extract (first 400 chars):")
            print(original_text[:400] + "..." if len(original_text) > 400 else original_text)
            
            # Analyze what we found
            original_analysis = analyze_receipt_text(original_text)
            print(f"Found: {original_analysis['money_amounts']} amounts, {original_analysis['dates']} dates")
            
    except Exception as e:
        print(f"Error with original: {e}")
    
    print("\n" + "="*60)
    
    # Test 2: Smart preprocessing
    print("\nSMART PREPROCESSING")
    try:
        smart_processed = preprocessor.smart_preprocess(image_path)
        
        smart_text = pytesseract.image_to_string(Image.fromarray(smart_processed))
        print("OCR Text Extract (first 400 chars):")
        print(smart_text[:400] + "..." if len(smart_text) > 400 else smart_text)
        
        # Analyze what we found
        smart_analysis = analyze_receipt_text(smart_text)
        print(f"Found: {smart_analysis['money_amounts']} amounts, {smart_analysis['dates']} dates")
        
    except Exception as e:
        print(f"Error with smart preprocessing: {e}")
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print(f"Original: {original_analysis.get('money_amounts', 0)} amounts, {original_analysis.get('dates', 0)} dates")
    print(f"Smart:    {smart_analysis.get('money_amounts', 0)} amounts, {smart_analysis.get('dates', 0)} dates")
    
    # Return the better result
    if original_analysis.get('money_amounts', 0) >= smart_analysis.get('money_amounts', 0):
        print("Winner: Original image")
        return original_text
    else:
        print("Winner: Smart preprocessing")
        return smart_text

def analyze_receipt_text(text):
    """Analyze OCR text to extract key receipt information"""
    if not text:
        return {'money_amounts': 0, 'dates': 0, 'words': 0, 'lines': 0}
    
    # Find money amounts (support multiple currencies)
    money_patterns = [
        r'[£$€]\s*\d+\.\d{2}',  # £4.50, $4.50, €4.50
        r'\d+\.\d{2}',          # 4.50 (standalone)
    ]
    
    money_amounts = []
    for pattern in money_patterns:
        money_amounts.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Find dates
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY, DD-MM-YYYY
        r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',    # DD/MM/YYYY
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    
    return {
        'money_amounts': len(money_amounts),
        'money_list': money_amounts,
        'dates': len(dates),
        'date_list': dates,
        'words': len(text.split()),
        'lines': len([line for line in text.split('\n') if line.strip()]),
    }

if __name__ == "__main__":
    # Make sure directories exist
    os.makedirs('data/processed_images', exist_ok=True)
    os.makedirs('data/test_outputs', exist_ok=True)
    
    # Try your actual image files
    possible_paths = [
        "data/sample_receipts/sample_receipt.jpeg",  # Your main receipt
        "data/sample_receipts/sample_receipt_2.webp", # Your second receipt
        "data/sample_receipts/sample_receipt.jpg",   # Fallback
    ]
    
    print("Looking for receipt images...")
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\nFound image: {path}")
            try:
                best_text = test_smart_ocr(path)
                print(f"\nOCR completed for {path}")
                print("="*60)
                break
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
    else:
        print("No receipt images could be processed")
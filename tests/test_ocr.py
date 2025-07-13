import cv2
import pytesseract
from PIL import Image
import numpy as np

# Test if everything works
def test_basic_ocr():
    # Create a simple test image with text
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'TEST RECEIPT $12.34', (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save test image
    cv2.imwrite('test_receipt.png', img)
    
    # Try OCR
    text = pytesseract.image_to_string(Image.open('test_receipt.png'))
    print(f"OCR Result: '{text.strip()}'")
    
    if 'TEST' in text and '12.34' in text:
        print("OCR is working!")
        return True
    else:
        print("OCR might need configuration")
        return False

if __name__ == "__main__":
    test_basic_ocr()
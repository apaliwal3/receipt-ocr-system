import cv2
import pytesseract
from PIL import Image

def test_real_receipt(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic preprocessing
        processed = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # OCR
        text = pytesseract.image_to_string(processed)
        
        print("=== RAW OCR RESULT ===")
        print(text)
        print("=" * 50)
        
        # Look for money amounts
        import re
        money_pattern = r'\$?\d+\.\d{2}'
        amounts = re.findall(money_pattern, text)
        print(f"Found amounts: {amounts}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_real_receipt("sample_receipt_2.webp")
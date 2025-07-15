import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import os
from typing import Tuple, Dict

class ReceiptPreprocessor:
    """
    Combined image preprocessing and OCR text extraction for receipts
    Handles image enhancement and text cleaning in one pipeline
    """
    
    def __init__(self):
        # Keep OCR configs simple
        self.ocr_configs = {
            'simple': '',  # No config = default
            'basic': r'--oem 3 --psm 6',
        }
        # Resize parameters
        self.target_width = 1000
        self.min_height = 600
        self.max_height = 1600
    
    def process_receipt(self, image_path: str) -> Tuple[str, Dict]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        processed_image = self._preprocess_image(image_path)
        raw_text, ocr_method = self._extract_text_multi_method(processed_image)

        print("=== RAW OCR RESULT ===")
        print(raw_text)
        print("=" * 50)
        
        cleaned_text = self._clean_and_reconstruct_text(raw_text)
        
        print("=== CLEANED TEXT RESULT ===")
        print(cleaned_text)
        print("=" * 50)
        
        metrics = self._analyze_text_quality(cleaned_text)
        metrics['ocr_method'] = ocr_method
        metrics['raw_text_length'] = len(raw_text)
        metrics['cleaned_text_length'] = len(cleaned_text)
        
        return cleaned_text, metrics
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = self._resize_optimally(gray)
        blurred = cv2.GaussianBlur(resized, (1, 1), 0)
        
        return blurred
    
    def _resize_optimally(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w > h:
            target_w = self.target_width
            target_h = int(h * (self.target_width / w))
        else:
            if h < self.min_height:
                target_h = self.min_height
                target_w = int(w * (self.min_height / h))
            elif h > self.max_height:
                target_h = self.max_height
                target_w = int(w * (self.max_height / h))
            else:
                return image
        
        interpolation = cv2.INTER_CUBIC if target_w * target_h > w * h else cv2.INTER_AREA
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    
    def _extract_text_multi_method(self, image: np.ndarray) -> Tuple[str, str]:
        pil_image = Image.fromarray(image)
        try:
            text = pytesseract.image_to_string(pil_image).strip()
            if text and len(text) > 20:
                return text, "simple"
        except Exception as e:
            print(f"Simple OCR failed: {e}")
        
        try:
            text = pytesseract.image_to_string(pil_image, config=self.ocr_configs['basic']).strip()
            if text:
                return text, "fallback"
        except Exception as e:
            print(f"Fallback OCR failed: {e}")
        
        return "", "none"
    
    def _clean_and_reconstruct_text(self, raw_text: str) -> str:
        if not raw_text:
            return raw_text
        
        text = raw_text
        text = self._fix_minimal_errors(text)
        text = self._fix_character_errors(text)
        text = self._remove_artifact_characters(text)
        text = self._reconstruct_line_breaks(text)
        text = self._clean_lines(text)
        
        return text
    
    def _fix_minimal_errors(self, text: str) -> str:
        text = text.replace('-£0,50', '-£0.50')
        text = text.replace('£0,50', '£0.50')
        text = re.sub(r'-£-(\d)', r'-£\1', text)
        return text

    def _fix_character_errors(self, text: str) -> str:
        text = re.sub(r'(\d)[Oo](\d)', r'\g<1>0\g<2>', text)
        text = re.sub(r'(\d)[Il](\d)', r'\g<1>1\g<2>', text)
        text = re.sub(r'([£$€])[Oo](\d)', r'\g<1>0\g<2>', text)
        text = re.sub(r'([£$€])[Il](\d)', r'\g<1>1\g<2>', text)
        text = re.sub(r'(\d),(\d{2})', r'\1.\2', text)
        return text

    def _remove_artifact_characters(self, text: str) -> str:
        text = re.sub(r'(\d\.\d{2})\s*[a-zA-Z]\s+([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+[a-zA-Z]\s+([£$€]?\d)', r' \1', text)
        text = re.sub(r'([a-zA-Z]{3,})\s*[a-zA-Z]\s+', r'\1 ', text)
        text = re.sub(r'\s+[,\.\-\+\*]\s+', ' ', text)
        return text
    
    def _reconstruct_line_breaks(self, text: str) -> str:
        patterns = [
            (r'([a-z])([A-Z]{2,})', r'\1\n\2'),
            (r'(\d)([A-Z][a-z])', r'\1\n\2'),
            (r'([a-z])(RECEIPT)', r'\1\n\2'),
            (r'([a-z])(SALES)', r'\1\n\2'),
            (r'([a-z])(Qty)', r'\1\n\2'),
            (r'([a-z])(Item)', r'\1\n\2'),
            (r'([a-z])(Price)', r'\1\n\2'),
            (r'([a-z])(Sub)', r'\1\n\2'),
            (r'([a-z])(Total)', r'\1\n\2'),
            (r'([a-z])(Line)', r'\1\n\2'),
            (r'([a-z])(Transaction)', r'\1\n\2'),
            (r'([a-z])(Discount)', r'\1\n\2'),
            (r'([a-z])(THANK)', r'\1\n\2'),
            (r'([a-z])(Admin)', r'\1\n\2'),
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        text = re.sub(r'([a-z])(\d+x[A-Z])', r'\1\n\2', text)
        text = re.sub(r'([a-z])([£$€]\d)', r'\1\n\2', text)
        text = re.sub(r'([a-z])(\d+\.\d{2})', r'\1\n\2', text)
        text = re.sub(r'([a-z])(\d{1,2}/\d{1,2}/\d)', r'\1\n\2', text)
        text = re.sub(r'([a-z])(\d{1,2}-\d{1,2}-\d)', r'\1\n\2', text)
        text = re.sub(r'([a-z])(\d{1,2}:\d{2})', r'\1\n\2', text)
        return text
    
    def _clean_lines(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^[,\.\-\+\*\s]+$', line):
                continue
            if len(line) < 2:
                continue
            line = re.sub(r'\s+', ' ', line)
            line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
            line = re.sub(r'(\d)([A-Z][a-z])', r'\1 \2', line)
            line = re.sub(r'([a-z]):([A-Z])', r'\1: \2', line)
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def _analyze_text_quality(self, text: str) -> Dict:
        if not text:
            return {
                'text_length': 0,
                'lines_count': 0,
                'money_amounts': 0,
                'dates_found': 0,
                'quality_score': 0.0
            }
        lines = [line for line in text.split('\n') if line.strip()]
        money_amounts = len(re.findall(r'[£$€]?\d+\.\d{2}', text))
        dates = len(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text))
        quality_score = min(
            (money_amounts * 0.3 + dates * 0.2 + min(len(lines)/10, 1) * 0.3 + min(len(text)/500, 1) * 0.2),
            1.0
        )
        return {
            'text_length': len(text),
            'lines_count': len(lines),
            'money_amounts': money_amounts,
            'dates_found': dates,
            'quality_score': quality_score
        }

if __name__ == "__main__":
    preprocessor = ReceiptPreprocessor()
    test_images = [
        "../data/sample_receipts/sample_receipt.jpeg",
        "data/sample_receipts/sample_receipt.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"Testing preprocessing on: {image_path}")
            try:
                cleaned_text, metrics = preprocessor.process_receipt(image_path)
                print(f"OCR Method: {metrics['ocr_method']}")
                print(f"Quality Score: {metrics['quality_score']:.2%}")
                print(f"Lines: {metrics['lines_count']}")
                print(f"Money amounts: {metrics['money_amounts']}")
                print("\nCleaned text:")
                print("-" * 40)
                print(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("No test images found")

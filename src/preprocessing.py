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
        # Fix specific price formatting issues
        text = text.replace('-£0,50', '-£0.50')
        text = text.replace('£0,50', '£0.50')
        # Fix double negative in prices like -£-1.00 -> -£1.00
        text = re.sub(r'-£-(\d)', r'-£\1', text)
        return text

    def _fix_character_errors(self, text: str) -> str:
        # Fix common OCR character misreads in numbers
        # O/o mistaken for 0 in numbers
        text = re.sub(r'(\d)[Oo](\d)', r'\g<1>0\g<2>', text)
        # I/l mistaken for 1 in numbers
        text = re.sub(r'(\d)[Il](\d)', r'\g<1>1\g<2>', text)
        # Fix currency symbols followed by O/I
        text = re.sub(r'([£$€])[Oo](\d)', r'\g<1>0\g<2>', text)
        text = re.sub(r'([£$€])[Il](\d)', r'\g<1>1\g<2>', text)
        # Fix comma instead of decimal point in currency
        text = re.sub(r'([£$€])(\d+),(\d{2})(?!\d)', r'\1\2.\3', text)
        # Fix standalone price amounts with commas
        text = re.sub(r'(\d+),(\d{2})(?=\s|$)', r'\1.\2', text)
        return text

    def _remove_artifact_characters(self, text: str) -> str:
        # Remove single characters that appear between meaningful content
        # But be much more conservative to avoid removing valid word endings
        
        # Remove isolated single characters between price and next word
        # Only if there's a clear price pattern before it
        text = re.sub(r'(\d\.\d{2})\s+[a-zA-Z]\s+([A-Z][a-zA-Z]+)', r'\1\n\2', text)
        
        # Remove single characters that appear isolated between spaces
        # But only if they're clearly artifacts (not valid words)
        text = re.sub(r'\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\s+(?=[A-Z])', r' ', text)
        
        # Remove sequences of punctuation/symbols that are clearly artifacts
        text = re.sub(r'\s*[,\.\-\+\*]{2,}\s*', ' ', text)
        
        return text
    
    def _reconstruct_line_breaks(self, text: str) -> str:
        # Add line breaks in appropriate places
        # But be more conservative to avoid breaking valid compound words
        # Add line breaks after prices to separate items
        
        patterns = [
            # Break before receipt keywords that are likely new lines
            (r'([a-z])(\s*RECEIPT)', r'\1\n\2'),
            (r'([a-z])(\s*SALES)', r'\1\n\2'),
            (r'([a-z])(\s*THANK)', r'\1\n\2'),
            # Break before quantity patterns like "2x Items"
            (r'([a-z])(\s*\d+x\s*[A-Z])', r'\1\n\2'),
            # Break AFTER prices (currency amounts and decimal prices)
            (r'([£$€]\d+\.\d{2})(\s+[A-Z])', r'\1\n\2'),
            (r'(\d+\.\d{2})(\s+[A-Z][a-zA-Z])', r'\1\n\2'),
            # Break before dates
            (r'([a-z])(\s*\d{1,2}/\d{1,2}/\d)', r'\1\n\2'),
            (r'([a-z])(\s*\d{1,2}-\d{1,2}-\d)', r'\1\n\2'),
            # Break before times
            (r'([a-z])(\s*\d{1,2}:\d{2})', r'\1\n\2'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _clean_lines(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are just punctuation/symbols
            if re.match(r'^[,\.\-\+\*\s]+$', line):
                continue
            
            # Skip single character lines unless they're meaningful
            if len(line) == 1 and line not in ['A', 'I']:
                continue
            
            # Clean up multiple spaces
            line = re.sub(r'\s+', ' ', line)
            
            # Add space before capital letters only in specific contexts
            # Be more conservative to avoid breaking valid words
            line = re.sub(r'([a-z])([A-Z][a-z]{2,})', r'\1 \2', line)  # Only break if next part is 3+ chars
            line = re.sub(r'(\d)([A-Z][a-z]{2,})', r'\1 \2', line)     # Only break if next part is 3+ chars
            
            # Fix spacing around colons
            line = re.sub(r'([a-zA-Z]):([A-Z])', r'\1: \2', line)
            
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
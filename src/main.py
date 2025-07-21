#!/usr/bin/env python3
"""
Receipt Processing Package
Combines OCR preprocessing and receipt parsing into a single workflow
"""

import sys
import os
from preprocessing import ReceiptPreprocessor
from receipt_parser import ReceiptParser

def process_receipt_complete(image_path: str, verbose: bool = True) -> dict:
    """
    Complete receipt processing pipeline
    
    Args:
        image_path: Path to receipt image
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing all processing results
    """
    try:
        # Initialize processors
        preprocessor = ReceiptPreprocessor()
        parser = ReceiptParser()
        
        if verbose:
            print(f"Processing receipt: {image_path}")
            print("=" * 60)
        
        # Step 1: OCR and text cleaning
        cleaned_text, ocr_metrics = preprocessor.process_receipt(image_path)
        
        # Step 2: Parse structured data
        receipt_data = parser.parse_receipt(cleaned_text)
        formatted_output = parser.format_receipt_data(receipt_data)
        
        if verbose:
            print(formatted_output)
            print("\n" + "=" * 60)
            print(f"OCR Quality Score: {ocr_metrics['quality_score']:.2%}")
            print(f"OCR Method Used: {ocr_metrics['ocr_method']}")
        
        return {
            'success': True,
            'cleaned_text': cleaned_text,
            'receipt_data': receipt_data,
            'formatted_output': formatted_output,
            'ocr_metrics': ocr_metrics
        }
        
    except Exception as e:
        error_msg = f"Error processing receipt: {str(e)}"
        if verbose:
            print(error_msg)
        return {
            'success': False,
            'error': error_msg
        }

def main():
    """Main entry point for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [--quiet]")
        print("Example: python main.py receipt.jpg")
        print("         python main.py receipt.jpg --quiet")
        sys.exit(1)
    
    image_path = sys.argv[1]
    verbose = '--quiet' not in sys.argv
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    result = process_receipt_complete(image_path, verbose)
    
    if not result['success']:
        sys.exit(1)

if __name__ == "__main__":
    main()
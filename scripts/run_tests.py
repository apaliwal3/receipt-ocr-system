#!/usr/bin/env python3
import os
import sys
import subprocess

def run_tests():
    """Run all tests in the tests directory"""
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')
    
    test_files = [
        'test_ocr.py',
        'test_real_receipt.py', 
        'test_improved_ocr.py'
    ]
    
    print("Running Receipt OCR Tests")
    print("=" * 40)
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            print(f"\nRunning {test_file}...")
            try:
                subprocess.run([sys.executable, test_path], check=True)
                print(f"{test_file} passed")
            except subprocess.CalledProcessError:
                print(f"{test_file} failed")
        else:
            print(f"{test_file} not found")

if __name__ == "__main__":
    run_tests()
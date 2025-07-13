import os
import platform

class Config:
    # Tesseract Configuration
    TESSERACT_PATHS = {
        'Windows': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        'Linux': '/usr/bin/tesseract',
        'Darwin': '/usr/local/bin/tesseract'  # macOS
    }
    
    # OCR Settings
    OCR_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$-: '
    
    # Directories
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    SAMPLE_RECEIPTS_DIR = os.path.join(DATA_DIR, 'sample_receipts')
    PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, 'processed_images')
    TEST_OUTPUTS_DIR = os.path.join(DATA_DIR, 'test_outputs')
    EXPORTS_DIR = os.path.join(PROJECT_ROOT, 'exports')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    
    @classmethod
    def get_tesseract_path(cls):
        system = platform.system()
        return cls.TESSERACT_PATHS.get(system, '/usr/bin/tesseract')
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        dirs = [
            cls.DATA_DIR, cls.SAMPLE_RECEIPTS_DIR, 
            cls.PROCESSED_IMAGES_DIR, cls.TEST_OUTPUTS_DIR,
            cls.EXPORTS_DIR, cls.MODELS_DIR
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
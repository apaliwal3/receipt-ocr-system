# Template for configuration - copy to config_local.py and customize
import os

# Tesseract Configuration
TESSERACT_PATH = {
    'windows': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    'linux': '/usr/bin/tesseract',
    'darwin': '/usr/local/bin/tesseract'  # macOS
}

# OCR Settings
OCR_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$-: '

# Expense Categories
EXPENSE_CATEGORIES = [
    'Food & Dining',
    'Groceries', 
    'Transportation',
    'Shopping',
    'Entertainment',
    'Healthcare',
    'Utilities',
    'Other'
]
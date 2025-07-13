import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self):
        pass
    
    def smart_preprocess(self, image_path):
        """Smart preprocessing - only process if the image actually needs it"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Analyze image quality
        quality_metrics = self.analyze_image_quality(gray)
        
        print("📊 Image Quality Analysis:")
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value}")
        
        # Decide if preprocessing is needed
        needs_processing = self.needs_preprocessing(quality_metrics)
        
        if needs_processing:
            print("🔧 Image needs preprocessing...")
            processed = self.minimal_preprocessing(gray)
            cv2.imwrite('../data/processed_images/processed_receipt.png', processed)
            return processed
        else:
            print("✅ Image quality is good - using original with minimal changes")
            # Just resize if needed and convert to grayscale
            resized = self.resize_if_needed(gray)
            cv2.imwrite('../data/processed_images/processed_receipt.png', resized)
            return resized
    
    def analyze_image_quality(self, image):
        """Analyze image quality to determine if preprocessing is needed"""
        
        # Calculate various quality metrics
        metrics = {}
        
        # 1. Contrast (standard deviation of pixel values)
        metrics['contrast'] = np.std(image)
        
        # 2. Brightness (mean pixel value)
        metrics['brightness'] = np.mean(image)
        
        # 3. Sharpness (variance of Laplacian - edge detection)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        metrics['sharpness'] = laplacian.var()
        
        # 4. Noise level (estimate)
        # High frequency noise estimation
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        filtered = cv2.filter2D(image, -1, kernel)
        metrics['noise_level'] = np.std(filtered)
        
        # 5. Resolution
        h, w = image.shape[:2]
        metrics['resolution'] = f"{w}x{h}"
        metrics['pixel_count'] = w * h
        
        return metrics
    
    def needs_preprocessing(self, metrics):
        """Determine if image needs preprocessing based on quality metrics"""
        
        # Good quality thresholds
        good_contrast_min = 30      # Minimum contrast
        good_brightness_min = 50    # Minimum brightness
        good_brightness_max = 200   # Maximum brightness
        good_sharpness_min = 100    # Minimum sharpness
        max_noise_level = 15        # Maximum acceptable noise
        
        # Check if image has quality issues
        issues = []
        
        if metrics['contrast'] < good_contrast_min:
            issues.append('low_contrast')
        
        if metrics['brightness'] < good_brightness_min:
            issues.append('too_dark')
        elif metrics['brightness'] > good_brightness_max:
            issues.append('too_bright')
        
        if metrics['sharpness'] < good_sharpness_min:
            issues.append('blurry')
        
        if metrics['noise_level'] > max_noise_level:
            issues.append('noisy')
        
        if issues:
            print(f"Image issues detected: {', '.join(issues)}")
            return True
        else:
            print("Image quality is good!")
            return False
    
    def minimal_preprocessing(self, image):
        """Very minimal preprocessing for images that actually need it"""
        
        # Only do the absolute minimum
        processed = image.copy()
        
        # 1. Resize if needed
        processed = self.resize_if_needed(processed)
        
        # 2. Very light contrast adjustment if needed
        if np.std(processed) < 30:  # Only if very low contrast
            processed = cv2.convertScaleAbs(processed, alpha=1.1, beta=5)
        
        return processed
    
    def resize_if_needed(self, image, min_height=800, max_height=1200):
        """Resize only if the image is too small or too large"""
        h, w = image.shape[:2]
        
        if h < min_height:
            # Image is too small, upscale
            ratio = min_height / h
            new_width = int(w * ratio)
            resized = cv2.resize(image, (new_width, min_height), interpolation=cv2.INTER_CUBIC)
            print(f"    Upscaled from {w}x{h} to {new_width}x{min_height}")
            return resized
        elif h > max_height:
            # Image is too large, downscale
            ratio = max_height / h
            new_width = int(w * ratio)
            resized = cv2.resize(image, (new_width, max_height), interpolation=cv2.INTER_AREA)
            print(f"    Downscaled from {w}x{h} to {new_width}x{max_height}")
            return resized
        else:
            print(f"    Size {w}x{h} is good, no resizing needed")
            return image
    
    def preprocess_receipt(self, image_path):
        """Main preprocessing method - now uses smart preprocessing"""
        return self.smart_preprocess(image_path)

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    
    try:
        result = preprocessor.smart_preprocess("../data/sample_receipts/sample_receipt_2.webp")
        print("Smart preprocessing completed!")
        
    except Exception as e:
        print(f"Error: {e}")
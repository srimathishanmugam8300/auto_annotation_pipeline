"""
Lightweight Visual Feature Extractor (No PyTorch required)
Uses classical computer vision techniques for feature extraction
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
import config


class LightweightExtractor:
    """
    Lightweight visual feature extractor using classical CV.
    No deep learning, no downloads, works instantly.
    """
    
    def __init__(self):
        """Initialize lightweight feature extractor"""
        print("🔄 Initializing lightweight feature extractor...")
        
        # Use multiple feature descriptors for robustness
        self.sift = cv2.SIFT_create(nfeatures=100)
        self.orb = cv2.ORB_create(nfeatures=100)
        
        # Also use color histograms
        self.hist_bins = 32
        
        print("✓ Lightweight extractor ready (CPU-only, no downloads)")
    
    def extract_image_features(self, image_path: Path) -> np.ndarray:
        """
        Extract visual features from a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector (numpy array)
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Resize for consistency
            img = cv2.resize(img, (224, 224))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract multiple feature types
            features = []
            
            # 1. SIFT features (shape and texture)
            kp, des = self.sift.detectAndCompute(gray, None)
            if des is not None and len(des) > 0:
                # Average pooling of SIFT descriptors
                sift_feat = np.mean(des, axis=0)
            else:
                sift_feat = np.zeros(128)
            features.append(sift_feat)
            
            # 2. ORB features (corners and edges)
            kp, des = self.orb.detectAndCompute(gray, None)
            if des is not None and len(des) > 0:
                orb_feat = np.mean(des.astype(float), axis=0)
            else:
                orb_feat = np.zeros(32)
            features.append(orb_feat)
            
            # 3. Color histograms (RGB)
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [self.hist_bins], [0, 256])
                hist = hist.flatten() / (224 * 224)  # Normalize
                features.append(hist)
            
            # 4. Texture features (Gabor-like using simple filters)
            texture_feat = self._extract_texture_features(gray)
            features.append(texture_feat)
            
            # Concatenate all features
            feature_vector = np.concatenate(features)
            
            return feature_vector
            
        except Exception as e:
            print(f"⚠️  Error processing {image_path.name}: {e}")
            return None
    
    def _extract_texture_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Extract simple texture features using edge detection"""
        # Sobel edges in X and Y
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Edge magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Simple statistics
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(sobel_x),
            np.std(sobel_x),
            np.mean(sobel_y),
            np.std(sobel_y)
        ]
        
        return np.array(features)
    
    def extract_batch_features(self, image_paths: List[Path]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (features matrix, valid image paths)
        """
        features_list = []
        valid_paths = []
        
        print(f"📊 Extracting visual features from {len(image_paths)} images...")
        
        for path in tqdm(image_paths, desc="Processing images"):
            features = self.extract_image_features(path)
            
            if features is not None:
                features_list.append(features)
                valid_paths.append(str(path))
        
        if features_list:
            features_matrix = np.vstack(features_list)
            print(f"✓ Extracted {features_matrix.shape[0]} feature vectors (dim={features_matrix.shape[1]})")
            return features_matrix, valid_paths
        else:
            print("❌ No valid images processed")
            return np.array([]), []
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        # SIFT(128) + ORB(32) + RGB_histograms(32*3) + Texture(6)
        return 128 + 32 + (self.hist_bins * 3) + 6


def load_image_paths(image_folder: Path) -> List[Path]:
    """
    Load all valid image paths from a folder.
    
    Args:
        image_folder: Path to folder containing images
        
    Returns:
        List of image file paths
    """
    image_paths = []
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    for ext in supported_formats:
        image_paths.extend(image_folder.glob(f"*{ext}"))
        image_paths.extend(image_folder.glob(f"*{ext.upper()}"))
    
    # Also check subdirectories
    for ext in supported_formats:
        image_paths.extend(image_folder.glob(f"**/*{ext}"))
        image_paths.extend(image_folder.glob(f"**/*{ext.upper()}"))
    
    # Remove duplicates
    image_paths = list(set(image_paths))
    
    print(f"📂 Found {len(image_paths)} images in {image_folder}")
    return sorted(image_paths)


if __name__ == "__main__":
    # Test feature extraction
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python feature_extractor_lightweight.py <image_folder>")
        sys.exit(1)
    
    image_folder = Path(sys.argv[1])
    if not image_folder.exists():
        print(f"❌ Folder not found: {image_folder}")
        sys.exit(1)
    
    # Load images
    image_paths = load_image_paths(image_folder)
    
    # Extract features
    extractor = LightweightExtractor()
    features, valid_paths = extractor.extract_batch_features(image_paths)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Shape: {features.shape}")
    print(f"  Valid images: {len(valid_paths)}")

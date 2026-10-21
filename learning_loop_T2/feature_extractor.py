"""
DINOv2 Visual Feature Extractor
Pure visual encoding - no text, no labels, no domain assumptions
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
import config


class DINOv2Extractor:
    """
    Universal visual feature extractor using DINOv2.
    Works for any image domain without requiring labels or text.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize DINOv2 feature extractor.
        
        Args:
            model_name: DINOv2 model variant (default from config)
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name or config.DINOV2_MODEL
        self.device = device or config.DEVICE
        
        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        print(f"🔄 Loading {self.model_name}...")
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
        
        print(f"✓ DINOv2 loaded on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load DINOv2 model from torch hub"""
        try:
            model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            return model
        except Exception as e:
            print(f"❌ Error loading DINOv2: {e}")
            print("Attempting to load from local cache...")
            # Fallback to local loading if hub fails
            model = torch.hub.load('facebookresearch/dinov2', self.model_name, 
                                   force_reload=False, skip_validation=True)
            return model
    
    def extract_image_features(self, image_path: Path) -> np.ndarray:
        """
        Extract visual features from a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector (numpy array)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convert to numpy
            features = features.cpu().numpy().flatten()
            return features
            
        except Exception as e:
            print(f"⚠️  Error processing {image_path.name}: {e}")
            return None
    
    def extract_batch_features(self, image_paths: List[Path]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple images in batches.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (features matrix, valid image paths)
        """
        features_list = []
        valid_paths = []
        
        print(f"📊 Extracting visual features from {len(image_paths)} images...")
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(image_paths), config.BATCH_SIZE), desc="Processing batches"):
            batch_paths = image_paths[i:i + config.BATCH_SIZE]
            batch_images = []
            batch_valid_paths = []
            
            # Load batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                    batch_valid_paths.append(str(path))
                except Exception as e:
                    print(f"⚠️  Skipping {path.name}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Stack and process batch
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # Store results
            features_list.append(batch_features.cpu().numpy())
            valid_paths.extend(batch_valid_paths)
        
        # Combine all features
        if features_list:
            features_matrix = np.vstack(features_list)
            print(f"✓ Extracted {features_matrix.shape[0]} feature vectors (dim={features_matrix.shape[1]})")
            return features_matrix, valid_paths
        else:
            print("❌ No valid images processed")
            return np.array([]), []
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        return config.EMBEDDING_DIM


def load_image_paths(image_folder: Path) -> List[Path]:
    """
    Load all valid image paths from a folder.
    
    Args:
        image_folder: Path to folder containing images
        
    Returns:
        List of image file paths
    """
    image_paths = []
    
    for ext in config.SUPPORTED_IMAGE_FORMATS:
        image_paths.extend(image_folder.glob(f"*{ext}"))
        image_paths.extend(image_folder.glob(f"*{ext.upper()}"))
    
    # Also check subdirectories
    for ext in config.SUPPORTED_IMAGE_FORMATS:
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
        print("Usage: python feature_extractor.py <image_folder>")
        sys.exit(1)
    
    image_folder = Path(sys.argv[1])
    if not image_folder.exists():
        print(f"❌ Folder not found: {image_folder}")
        sys.exit(1)
    
    # Load images
    image_paths = load_image_paths(image_folder)
    
    # Extract features
    extractor = DINOv2Extractor()
    features, valid_paths = extractor.extract_batch_features(image_paths)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Shape: {features.shape}")
    print(f"  Valid images: {len(valid_paths)}")

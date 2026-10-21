"""
Semantic Matching Module (Optional)
Maps visual clusters to user-provided class names using CLIP
Falls back to generic names if matching fails
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional

# Optional dependencies for semantic matching
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    torch = None
    CLIPProcessor = None
    CLIPModel = None

import config


class SemanticMatcher:
    """
    Optional semantic layer to map visual clusters to text labels.
    
    This is a lightweight bridge between visual clusters and human-readable names.
    If it fails (e.g., for MRI scans), the system gracefully falls back to generic names.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize semantic matching with CLIP.
        
        Args:
            model_name: CLIP model to use
            device: Device to run on
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "Semantic matching requires 'transformers' package. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name or config.SEMANTIC_MODEL
        self.device = device or config.DEVICE
        
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        
        self.model = None
        self.processor = None
        
        if config.USE_SEMANTIC_MATCHING:
            self._load_model()
    
    def _load_model(self):
        """Load CLIP model for semantic matching"""
        try:
            print(f"🔄 Loading semantic matcher ({self.model_name})...")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Semantic matcher loaded on {self.device}")
        except Exception as e:
            print(f"⚠️  Could not load semantic matcher: {e}")
            print("   Will use generic class names instead")
            self.model = None
    
    def match_clusters_to_names(
        self,
        image_paths: List[str],
        cluster_labels: np.ndarray,
        class_names: List[str],
        n_samples_per_cluster: int = 10
    ) -> Dict[int, str]:
        """
        Match visual clusters to user-provided class names.
        
        Args:
            image_paths: List of image file paths
            cluster_labels: Cluster assignment for each image
            class_names: User-provided class names
            n_samples_per_cluster: Number of images to sample per cluster
            
        Returns:
            Dictionary mapping cluster_id -> class_name
        """
        print(f"\n🏷️  Matching clusters to class names...")
        
        if self.model is None:
            print("⚠️  Semantic matching unavailable, using generic names")
            return self._generic_mapping(cluster_labels, class_names)
        
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Check if number of classes matches
        if len(class_names) != n_clusters:
            print(f"⚠️  Mismatch: {n_clusters} clusters but {len(class_names)} class names")
            print(f"   Adjusting...")
            
            if len(class_names) < n_clusters:
                # Add generic names for extra clusters
                for i in range(len(class_names), n_clusters):
                    class_names.append(f"Class_{i+1}")
            else:
                # Trim excess class names
                class_names = class_names[:n_clusters]
        
        # Sample representative images from each cluster
        cluster_representatives = self._sample_cluster_representatives(
            image_paths, cluster_labels, unique_clusters, n_samples_per_cluster
        )
        
        # Compute similarity scores
        cluster_to_name = {}
        used_names = set()
        
        # Compute similarity matrix: clusters x class_names
        similarity_matrix = np.zeros((n_clusters, len(class_names)))
        
        for i, cluster_id in enumerate(unique_clusters):
            representatives = cluster_representatives[cluster_id]
            
            # Compute average similarity to each class name
            for j, class_name in enumerate(class_names):
                similarities = []
                
                for img_path in representatives:
                    sim = self._compute_image_text_similarity(img_path, class_name)
                    if sim is not None:
                        similarities.append(sim)
                
                if similarities:
                    similarity_matrix[i, j] = np.mean(similarities)
        
        # Hungarian algorithm for optimal matching
        from scipy.optimize import linear_sum_assignment
        
        # We want to maximize similarity, so negate for minimization
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Create mapping
        for cluster_idx, name_idx in zip(row_ind, col_ind):
            cluster_id = unique_clusters[cluster_idx]
            class_name = class_names[name_idx]
            confidence = similarity_matrix[cluster_idx, name_idx]
            
            if confidence < config.CONFIDENCE_THRESHOLD:
                print(f"   ⚠️  Low confidence for Cluster {cluster_id} -> '{class_name}' ({confidence:.3f})")
                class_name = f"Class_{cluster_id}"
            
            cluster_to_name[int(cluster_id)] = class_name
            print(f"   Cluster {cluster_id} -> '{class_name}' (confidence: {confidence:.3f})")
        
        return cluster_to_name
    
    def _sample_cluster_representatives(
        self,
        image_paths: List[str],
        cluster_labels: np.ndarray,
        unique_clusters: np.ndarray,
        n_samples: int
    ) -> Dict[int, List[str]]:
        """
        Sample representative images from each cluster.
        
        Args:
            image_paths: All image paths
            cluster_labels: Cluster labels
            unique_clusters: Unique cluster IDs
            n_samples: Number of samples per cluster
            
        Returns:
            Dictionary mapping cluster_id -> list of image paths
        """
        representatives = {}
        
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Sample randomly or take first N
            if len(cluster_indices) > n_samples:
                sampled_indices = np.random.choice(
                    cluster_indices, n_samples, replace=False
                )
            else:
                sampled_indices = cluster_indices
            
            representatives[cluster_id] = [
                image_paths[idx] for idx in sampled_indices
            ]
        
        return representatives
    
    def _compute_image_text_similarity(
        self,
        image_path: str,
        text: str
    ) -> Optional[float]:
        """
        Compute similarity between an image and text using CLIP.
        
        Args:
            image_path: Path to image
            text: Text description
            
        Returns:
            Similarity score (0-1) or None if failed
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Compute similarity
            with torch.no_grad():
                outputs = self.model(**inputs)
                similarity = outputs.logits_per_image[0, 0].cpu().item()
            
            # Normalize to 0-1 range (CLIP logits are typically -10 to 10)
            normalized_similarity = (similarity + 10) / 20
            normalized_similarity = max(0, min(1, normalized_similarity))
            
            return normalized_similarity
            
        except Exception as e:
            print(f"⚠️  Error computing similarity for {Path(image_path).name}: {e}")
            return None
    
    def _generic_mapping(
        self,
        cluster_labels: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[int, str]:
        """
        Create generic mapping when semantic matching fails.
        
        Args:
            cluster_labels: Cluster labels
            class_names: Optional class names to use
            
        Returns:
            Dictionary mapping cluster_id -> class_name
        """
        unique_clusters = np.unique(cluster_labels)
        
        if class_names and len(class_names) == len(unique_clusters):
            # Use provided names in order
            return {
                int(cluster_id): class_names[i]
                for i, cluster_id in enumerate(unique_clusters)
            }
        else:
            # Use generic names
            return {
                int(cluster_id): f"Class_{cluster_id}"
                for cluster_id in unique_clusters
            }


def validate_class_names(class_names: List[str]) -> List[str]:
    """
    Validate and clean class names.
    
    Args:
        class_names: List of class names
        
    Returns:
        Cleaned class names
    """
    cleaned = []
    
    for name in class_names:
        # Remove invalid characters for folder names
        cleaned_name = "".join(
            c if c.isalnum() or c in (' ', '_', '-') else '_'
            for c in name
        )
        cleaned_name = cleaned_name.strip()
        
        if not cleaned_name:
            cleaned_name = "unnamed_class"
        
        cleaned.append(cleaned_name)
    
    return cleaned


if __name__ == "__main__":
    # Test semantic matching
    print("Testing semantic matcher...")
    
    matcher = SemanticMatcher()
    
    # Test with generic mapping
    labels = np.array([0, 0, 1, 1, 2, 2])
    class_names = ["cat", "dog", "bird"]
    
    mapping = matcher._generic_mapping(labels, class_names)
    print(f"\nGeneric mapping: {mapping}")
    
    print("\n✓ Semantic matching test complete!")

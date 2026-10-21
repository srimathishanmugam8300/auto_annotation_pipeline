"""
Configuration for Universal Auto-Annotation Pipeline
"""

# DINOv2 Model Configuration
DINOV2_MODEL = "dinov2_vitb14"  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
EMBEDDING_DIM = 768  # 384 for vits14, 768 for vitb14, 1024 for vitl14, 1536 for vitg14
BATCH_SIZE = 32
DEVICE = "cuda"  # or "cpu"

# Image Processing
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Clustering Configuration
MIN_CLUSTER_SIZE = 10  # For HDBSCAN
MIN_SAMPLES = 5  # For HDBSCAN
CLUSTER_SELECTION_EPSILON = 0.0

# Learning Loop Configuration
MAX_ITERATIONS = 10
CONVERGENCE_THRESHOLD = 0.01  # Stop if less than 1% of images change clusters
PROTOTYPE_UPDATE_MOMENTUM = 0.9  # Exponential moving average for prototypes

# Semantic Matching Configuration
USE_SEMANTIC_MATCHING = True
SEMANTIC_MODEL = "openai/clip-vit-base-patch32"  # CLIP model for semantic matching
CONFIDENCE_THRESHOLD = 0.3  # Minimum similarity for semantic matching

# Dataset Split Configuration
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Output Configuration
CREATE_VISUALIZATION = True
SAVE_EMBEDDINGS = True
GENERATE_REPORT = True

# File Formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

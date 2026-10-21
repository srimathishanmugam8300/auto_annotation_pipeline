import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def get_preprocessing_transforms():
    """Returns torchvision transforms for DINOv2."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image, transform):
    """Preprocesses a PIL image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return transform(image).unsqueeze(0)

import os
from PIL import Image

def load_images(input_dir):
    """Loads images from the input directory."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(root, file))

    image_paths.sort()
    return image_paths

def read_image(image_path):
    """Reads an image and converts it to RGB."""
    try:
        with Image.open(image_path) as img:
            return img.convert('RGB')
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

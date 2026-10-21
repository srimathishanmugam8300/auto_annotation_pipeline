import os
import random
import shutil

# Paths
TEST_OPEN = r"D:\my_projects\Data_automation\dataset\test\open"
TEST_CLOSED = r"D:\my_projects\Data_automation\dataset\test\closed"
UNLABELED = r"D:\my_projects\Data_automation\unlabeled"

# How many images to take from EACH class
NUM_IMAGES = 100  # you can change this

os.makedirs(UNLABELED, exist_ok=True)

def copy_random_images(src_folder, count):
    images = os.listdir(src_folder)
    selected = random.sample(images, min(count, len(images)))

    for img in selected:
        src_path = os.path.join(src_folder, img)
        dst_path = os.path.join(UNLABELED, img)
        shutil.copy(src_path, dst_path)

        print(f"Copied: {img}")

# Copy from both classes
copy_random_images(TEST_OPEN, NUM_IMAGES)
copy_random_images(TEST_CLOSED, NUM_IMAGES)

print("✅ Random images copied to unlabeled folder")

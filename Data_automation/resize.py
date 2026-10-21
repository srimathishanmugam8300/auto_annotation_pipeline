import cv2
import os

TARGET_SIZE = (64, 64)

BASE_DIR = "dataset"  # matches your current folder
SPLITS = ["train", "test"]
CLASSES = ["open", "closed"]

for split in SPLITS:
    for cls in CLASSES:
        folder_path = os.path.join(BASE_DIR, split, cls)

        if not os.path.exists(folder_path):
            print("❌ Missing:", folder_path)
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Skipped:", img_path)
                continue

            resized = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(img_path, resized)

print("✅ All images resized to 64x64")

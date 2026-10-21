import os
import cv2
import shutil
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("eye_open_close_model.keras")

UNLABELED_DIR = "unlabeled"
OUTPUT_DIR = "labeled"

os.makedirs(os.path.join(OUTPUT_DIR, "open"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "closed"), exist_ok=True)

for img_name in os.listdir(UNLABELED_DIR):
    img_path = os.path.join(UNLABELED_DIR, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "open"
    else:
        label = "closed"

    dest = os.path.join(OUTPUT_DIR, label, img_name)
    shutil.copy(img_path, dest)

    print(f"{img_name} → {label} ({pred:.4f})")

print("✅ Automatic labeling completed")

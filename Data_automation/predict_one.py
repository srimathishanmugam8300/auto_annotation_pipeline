import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("eye_open_close_model.keras")

# Image path (change this)
img_path = r"D:\my_projects\Data_automation\dataset\test\closed\s0016_00022_1_0_0_0_1_01.png"

# Read image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = img.reshape(1, 64, 64, 1)

# Predict
pred = model.predict(img)[0][0]

if pred > 0.5:
    print("Prediction: OPEN EYE")
else:
    print("Prediction: CLOSED EYE")

print("Confidence:", pred)

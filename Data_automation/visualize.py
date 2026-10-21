import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# paths
MODEL_PATH = r"D:\my_projects\Data_automation\eye_open_close_model.keras"
UNLABELED_DIR = r"D:\my_projects\Data_automation\eye_open_close_model.h5"
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

images = os.listdir(UNLABELED_DIR)[:9]  # show 9 images
plt.figure(figsize=(10, 10))

for i, img_name in enumerate(images):
    img_path = os.path.join(UNLABELED_DIR, img_name)

    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = model.predict(arr)[0][0]

    if prob < 0.5:
        label = "CLOSED"
        confidence = (1 - prob)
    else:
        label = "OPEN"
        confidence = prob

    if confidence > 0.9:
        color = "green"
    elif confidence > 0.6:
        color = "orange"
    else:
        color = "red"

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{label} ({confidence*100:.2f}%)", color=color)

plt.suptitle("AI-Assisted Auto Labeling (Unlabeled Data)", fontsize=16)
plt.show()

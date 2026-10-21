import os
import uuid
import random

folder_path = r"D:\my_projects\learning_loop_T3\sign_lang"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# Get all image files
files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

# Step 1: Rename everything to temporary names
temp_files = []
for i, filename in enumerate(files):
    ext = os.path.splitext(filename)[1]
    temp_name = f"temp_{uuid.uuid4().hex}{ext}"
    
    old_path = os.path.join(folder_path, filename)
    temp_path = os.path.join(folder_path, temp_name)
    
    os.rename(old_path, temp_path)
    temp_files.append(temp_name)

# Step 2: Generate completely random new names
random_names = [uuid.uuid4().hex for _ in temp_files]

# Shuffle again for extra randomness
random.shuffle(random_names)

# Step 3: Rename temp files to random names
for temp_file, rand_name in zip(temp_files, random_names):
    
    ext = os.path.splitext(temp_file)[1]
    
    old_path = os.path.join(folder_path, temp_file)
    new_path = os.path.join(folder_path, rand_name + ext)
    
    os.rename(old_path, new_path)

print("Images scrambled completely.")
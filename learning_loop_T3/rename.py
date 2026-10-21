import os

# change this to your input folder path
folder_path = r"D:\my_projects\learning_loop_T3\sign_lang"

# supported image extensions
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# get all image files
files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

# sort files for consistent order
files.sort()

# rename files
for index, filename in enumerate(files, start=1):
    
    old_path = os.path.join(folder_path, filename)
    
    extension = os.path.splitext(filename)[1]
    
    new_filename = f"{index}{extension}"
    
    new_path = os.path.join(folder_path, new_filename)
    
    os.rename(old_path, new_path)

print("Renaming completed successfully.")
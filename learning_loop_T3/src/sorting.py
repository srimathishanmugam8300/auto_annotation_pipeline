import os
import shutil

INVALID_PATH_CHARS = '<>:"/\\|?*'


def _safe_label_name(name):
    cleaned = "".join("_" if ch in INVALID_PATH_CHARS else ch for ch in name)
    cleaned = cleaned.strip().strip(".")
    return cleaned or "unnamed_label"


def _folder_name_from_label(label, label_names):
    if label == -1:
        return "noise"
    if label_names and 0 <= label < len(label_names):
        return _safe_label_name(label_names[label])
    return str(label + 1)


def create_folders_and_sort(
    image_paths,
    labels,
    output_dir,
    label_names=None,
    clear_output=False,
):
    """Creates folders and copies images based on predicted labels."""
    if clear_output and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Pre-create requested label folders so users can inspect empty buckets too.
    if label_names:
        for label_name in label_names:
            os.makedirs(os.path.join(output_dir, _safe_label_name(label_name)), exist_ok=True)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        folder_name = _folder_name_from_label(label, label_names)
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)

    for img_path, label in zip(image_paths, labels):
        folder_name = _folder_name_from_label(int(label), label_names)
        dest_folder = os.path.join(output_dir, folder_name)

        filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_folder, filename)

        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
            counter += 1

        shutil.copy2(img_path, dest_path)

    print(f"Successfully sorted {len(image_paths)} images into {output_dir}")

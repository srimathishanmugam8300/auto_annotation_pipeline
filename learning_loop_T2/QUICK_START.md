# 🎯 Visual Clustering Pipeline - Quick Start

## What Changed

Your pipeline is now **pure visual clustering** — no text models, no training splits, just organized class folders.

---

## ✅ Two Modes

### Mode 1: With Class Names
```bash
python auto_pipeline.py data/images sign_output two three four six seven eight nine
```

**Output:**
```
sign_output/
  two/
  three/
  four/
  six/
  seven/
  eight/
  nine/
```

---

### Mode 2: With Number Only
```bash
python auto_pipeline.py data/images sign_output 7
```

**Output:**
```
sign_output/
  class_1/
  class_2/
  class_3/
  class_4/
  class_5/
  class_6/
  class_7/
```

---

## 🔧 How It Works

1. **Visual Feature Extraction** → Converts images to vectors
2. **Clustering** → Groups similar images together
3. **Refinement** → Learning loop improves purity
4. **Folder Creation** → Copies images directly to class folders

**No train/val/test splits**
**No CLIP or transformers**
**No text understanding**

Just pure visual similarity.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or minimal:
```bash
pip install opencv-python scikit-learn hdbscan umap-learn numpy pillow tqdm
```

---

## 🚀 Run Examples

### Sign Language Dataset
```bash
python auto_pipeline.py data/images sign_output two three four six seven eight nine
```

### Medical Images (7 classes)
```bash
python auto_pipeline.py medical_scans output 7
```

### Industrial Defects (with names)
```bash
python auto_pipeline.py factory_images output scratch dent crack normal
```

---

## 📂 Old vs New Output

### ❌ Old (Training-focused)
```
output/
  train/
    class_1/
    class_2/
  val/
    class_1/
    class_2/
  test/
    class_1/
    class_2/
```

### ✅ New (Pure Organization)
```
output/
  class_1/
  class_2/
  class_3/
```

---

## 💡 Why This Works Better

| Scenario | CLIP/Text Models | Visual Clustering |
|----------|------------------|-------------------|
| Sign Language | ❌ Fails | ✅ Works |
| MRI Scans | ❌ Fails | ✅ Works |
| X-Rays | ❌ Fails | ✅ Works |
| Industrial Defects | ❌ Fails | ✅ Works |
| Natural Images | ✅ Works | ✅ Works |

**Visual clustering works universally because it groups by appearance, not meaning.**

---

## 🎨 What Gets Organized

- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- Any size, any format
- Preserves original quality

---

## 🔍 Quality Metrics

The pipeline automatically calculates:
- **Silhouette Score** → How well-separated clusters are
- **Davies-Bouldin Index** → Cluster compactness
- **Calinski-Harabasz Score** → Cluster density

Saved in: `output/pipeline_results.json`

---

## ⚙️ Advanced Options

```bash
# Use CPU instead of GPU
python auto_pipeline.py data/images output 7 --device cpu

# Works automatically - no other flags needed!
```

---

## 📊 Metadata

Each run creates `metadata.json`:
```json
{
  "total_images": 350,
  "n_classes": 7,
  "class_counts": {
    "class_1": 50,
    "class_2": 48,
    "class_3": 52,
    ...
  }
}
```

---

## 🐛 Troubleshooting

### No images found
- Check that `data/images` contains valid image files
- Supported: `.jpg`, `.png`, `.bmp`, `.tiff`, `.webp`

### Wrong number of clusters
- Provide exact class count: `python auto_pipeline.py input output 7`
- Or provide exact class names

### Dependencies missing
```bash
pip install -r requirements.txt
```

---

## 🎯 Summary

```
Images → Visual Features → Clustering → Class Folders
```

**Simple. Universal. No training.**

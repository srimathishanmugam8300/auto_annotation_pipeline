# 🔥 Visual Clustering Pipeline

**Automatically organize unlabeled images into class folders using pure visual similarity.**

Works for **any** image domain:
- 🤟 Sign language
- 🏥 Medical imaging (MRI/X-ray)
- 🏭 Industrial defects
- 🔬 Microscopy
- 🐾 Animals
- 🎨 Abstract art
- ❓ Unknown datasets

**No text models. No training. No train/val/test splits.**

---

## 🚀 Quick Start

### Installation

```bash
# Install core dependencies
pip install opencv-python scikit-learn hdbscan umap-learn numpy pillow tqdm
```

### Two Modes

#### Mode 1: With Class Names
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

#### Mode 2: With Number Only
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

## 📊 Pipeline Overview

```
Unlabeled Images
       ↓
Visual Feature Extraction (shape, texture, structure)
       ↓
Clustering (group by visual similarity)
       ↓
Learning Loop Refinement (improve purity)
       ↓
Class Folders (direct output, no splits)
```

---

## 🔹 How It Works

### Step 1: Visual Feature Extraction

Each image is converted to a **visual feature vector**.

- **DINOv2** (if available): Deep learning features
- **Lightweight CV** (fallback): Classical computer vision

No text understanding needed.

### Step 2: Clustering

Images are grouped based on how they **look**, not what they **mean**:

- **K-Means**: Fixed number of clusters
- Visual similarity only

### Step 3: Learning Loop Refinement

Clusters are refined through iterative updates:

1. Compute cluster prototypes
2. Reassign borderline images
3. Update prototypes
4. Repeat until stable

Result: **Cleaner, visually consistent groups**

### Step 4: Folder Creation

Images are copied directly into class folders:

- Mode 1: Use your provided names
- Mode 2: Generate `class_1`, `class_2`, etc.

**No train/val/test splits**
**No ML training logic**
**Just organized folders**

---

## 🎯 Use Cases

### 1. Sign Language Dataset
```bash
python auto_pipeline.py signs output two three four six seven eight nine
```

### 2. Medical Imaging
```bash
python auto_pipeline.py mri_scans output 4
```

### 3. Industrial Defect Detection
```bash
python auto_pipeline.py factory_images output normal crack scratch dent
```

### 4. Animal Classification
```bash
python auto_pipeline.py zoo_images output dog cat tiger lion elephant giraffe
```

### 5. Unknown Dataset Exploration
```bash
python auto_pipeline.py mystery_data output 10
```

---

## 💡 Why This Works for Any Domain

| Problem | Text-Based (CLIP) | Visual Clustering |
|---------|-------------------|-------------------|
| Sign Language | ❌ No text match | ✅ Groups by hand shape |
| MRI Scans | ❌ No semantic meaning | ✅ Groups by visual pattern |
| X-Rays | ❌ No descriptions | ✅ Groups by structure |
| Industrial Defects | ❌ Hard to describe | ✅ Groups by appearance |
| Abstract Art | ❌ Subjective | ✅ Groups by style |

**The key:** Visual clustering asks "Which images look similar?" not "What is this image called?"

---

## ⚙️ Configuration

Edit [config.py](config.py) to customize:

```python
# Image Processing
IMAGE_SIZE = (224, 224)

# Clustering
MIN_CLUSTER_SIZE = 10
MAX_ITERATIONS = 10  # Learning loop iterations
TEST_SPLIT = 0.15

# Device
DEVICE = "cuda"  # or "cpu"
```

---

## 📦 Output Structure

### With Splits (default)

```
output/
├── train/
│   ├── dog/
│   ├── cat/
│   └── tiger/
## 📂 Output Structure

### Simple Class Folders (Default)

```
output/
├── two/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── three/
├── four/
├── six/
├── seven/
├── eight/
├── nine/
├── metadata.json              # Dataset statistics
└── pipeline_results.json      # Pipeline execution details
```

Or with numeric names:

```
output/
├── class_1/
├── class_2/
├── class_3/
├── class_4/
├── metadata.json
└── pipeline_results.json
```

---

## 🧪 Testing

Run the test script to verify setup:

```bash
python test_pipeline.py
```

Then try one of the examples it suggests.

---

## 🔧 Options

### Specify Device
```bash
python auto_pipeline.py data output two three four --device cuda
python auto_pipeline.py data output 7 --device cpu
```

---

## 📈 Quality Metrics

The pipeline computes clustering quality metrics:

- **Silhouette Score**: Cluster separation quality
- **Davies-Bouldin Index**: Cluster compactness
- **Calinski-Harabasz Score**: Cluster density
- **Intra-cluster Distance**: Within-cluster similarity
- **Inter-cluster Distance**: Between-cluster separation

Results are saved in `pipeline_results.json`.

---

## 🎓 Using the Organized Dataset

### For Manual Labeling
Just review each class folder and verify the groupings are correct.

### For Training
Use standard frameworks:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load directly from class folders
dataset = datasets.ImageFolder("output", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Start training!
for images, labels in dataloader:
    # Your training code here
    pass
```

---

## 🧠 Key Design Principles

### 1. **Pure Visual Similarity**
   - Groups by appearance, not meaning
   - No text models required

### 2. **Universal Application**
   - Works for any visual domain
   - No domain-specific configuration

### 3. **Simple Output**
   - Direct class folders
   - No unnecessary complexity

### 4. **Iterative Refinement**
   - Learning loop improves purity
   - Converges to stable clusters

---

## 🔍 Visual Features

The pipeline can use:

- **DINOv2** (if available): State-of-the-art self-supervised vision model
- **Lightweight CV** (fallback): Classical computer vision features

Both work well - DINOv2 is more powerful, lightweight is faster and doesn't require PyTorch.

---

## 📊 Example Output

After running the pipeline, you'll see:

```
==============================================================
UNIVERSAL AUTO-ANNOTATION PIPELINE
==============================================================

Input:  data/animals
Output: output/animals_labeled

──────────────────────────────────────────────────────────────
STEP 1: Loading Images
──────────────────────────────────────────────────────────────
📂 Found 500 images in data/animals

──────────────────────────────────────────────────────────────
STEP 2: Visual Feature Extraction (DINOv2)
──────────────────────────────────────────────────────────────
🔄 Loading dinov2_vitb14...
✓ DINOv2 loaded on cuda
📊 Extracting visual features from 500 images...
✓ Extracted 500 feature vectors (dim=768)

──────────────────────────────────────────────────────────────
STEP 3: Unsupervised Clustering
──────────────────────────────────────────────────────────────
🔍 Clustering 500 images...
  Method: K-Means (k=5)
✓ Discovered 5 visual clusters

──────────────────────────────────────────────────────────────
STEP 4: Learning Loop Refinement
──────────────────────────────────────────────────────────────
🔄 Starting cluster refinement loop...
   Iteration 1: 45 samples reassigned (9.00%)
   Iteration 2: 12 samples reassigned (2.40%)
   Iteration 3: 4 samples reassigned (0.80%)
✓ Converged after 3 iterations!

──────────────────────────────────────────────────────────────
STEP 5: Class Name Assignment
──────────────────────────────────────────────────────────────
🏷️  Matching clusters to class names...
   Cluster 0 → 'dog' (confidence: 0.892)
   Cluster 1 → 'cat' (confidence: 0.915)
   Cluster 2 → 'tiger' (confidence: 0.878)
   Cluster 3 → 'lion' (confidence: 0.901)
   Cluster 4 → 'giraffe' (confidence: 0.856)

──────────────────────────────────────────────────────────────
STEP 6: Dataset Organization
──────────────────────────────────────────────────────────────
📁 Organizing dataset into output/animals_labeled...
  Creating train/val/test splits...
    dog: 70 train, 15 val, 15 test
    cat: 70 train, 15 val, 15 test
    tiger: 70 train, 15 val, 15 test
    lion: 70 train, 15 val, 15 test
    giraffe: 70 train, 15 val, 15 test

✓ Dataset organized successfully!
  Location: output/animals_labeled

==============================================================
✓ PIPELINE COMPLETED SUCCESSFULLY!
==============================================================
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # or 8
```

### Too Many/Few Clusters
```bash
# Adjust HDBSCAN parameters in config.py
MIN_CLUSTER_SIZE = 5   # Smaller = more clusters
MIN_SAMPLES = 3
```

### Semantic Matching Fails
```bash
# Use --no-semantic flag
python auto_pipeline.py data output --n-clusters 5 --no-semantic
```

---

## 📚 Module Reference

### [auto_pipeline.py](auto_pipeline.py)
Main entry point for the pipeline

### [feature_extractor.py](feature_extractor.py)
DINOv2-based visual feature extraction

### [clustering.py](clustering.py)
Unsupervised clustering (K-Means, HDBSCAN)

### [refinement.py](refinement.py)
Learning loop for cluster refinement

### [semantic_matching.py](semantic_matching.py)
CLIP-based semantic name matching

### [dataset_organizer.py](dataset_organizer.py)
Dataset organization and splitting

### [config.py](config.py)
Configuration parameters

---

## 🤝 Contributing

This is a universal pipeline designed to work for any image domain. If you encounter issues with specific datasets, please:

1. Test with `--no-semantic` flag
2. Adjust clustering parameters in `config.py`
3. Try different DINOv2 model variants

---

## 📝 License

MIT License - Use freely for research and commercial projects

---

## 🎯 Summary

This pipeline provides:

✅ **Universal**: Works for any image domain  
✅ **Automatic**: No manual labeling needed  
✅ **Visual-only**: No text understanding required  
✅ **Iterative**: Learning loop refinement  
✅ **Flexible**: Optional semantic matching  
✅ **Production-ready**: Organized output with splits  

**Perfect for:**
- Rapid dataset creation
- Dataset exploration
- Domain-specific classification
- Research experiments
- Production ML pipelines

---

**Ready to auto-annotate your dataset? 🚀**

```bash
python auto_pipeline.py your_images output --n-clusters 10
```

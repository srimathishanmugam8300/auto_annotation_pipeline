# 📋 Documentation Index

Welcome to the **Visual Clustering Pipeline** - a pure visual similarity-based image organization system.

---

## 🚀 Start Here

1. **[QUICK_START.md](QUICK_START.md)** - Get started in 2 minutes
2. **[COMMANDS.md](COMMANDS.md)** - Command reference and examples
3. **[README.md](README.md)** - Full documentation

---

## 📚 Documentation Files

### Getting Started
- **[QUICK_START.md](QUICK_START.md)** - Installation and basic usage
- **[COMMANDS.md](COMMANDS.md)** - Command syntax and examples
- **[EXAMPLES.md](EXAMPLES.md)** - Real-world use cases

### Understanding Changes
- **[TRANSFORMATION_SUMMARY.md](TRANSFORMATION_SUMMARY.md)** - What changed and why
- **[CHANGES.md](CHANGES.md)** - Detailed before/after comparison

### Complete Reference
- **[README.md](README.md)** - Full system documentation
- **[config.py](config.py)** - Configuration options

---

## 🛠️ Scripts

### Main Pipeline
- **[auto_pipeline.py](auto_pipeline.py)** - Main clustering pipeline

### Testing & Examples
- **[test_pipeline.py](test_pipeline.py)** - Quick test script
- **[run_example.py](run_example.py)** - Full example (both modes)

### Modules
- **[feature_extractor.py](feature_extractor.py)** - DINOv2 feature extraction
- **[feature_extractor_lightweight.py](feature_extractor_lightweight.py)** - Classical CV features
- **[clustering.py](clustering.py)** - K-Means/HDBSCAN clustering
- **[refinement.py](refinement.py)** - Learning loop refinement

---

## 💡 Quick Examples

### Mode 1: With Class Names
```bash
python auto_pipeline.py data/images output two three four six seven eight nine
```

### Mode 2: With Number
```bash
python auto_pipeline.py data/images output 7
```

---

## 📊 What Changed

This pipeline was transformed from a **training-oriented system** to a **pure visual clustering system**.

### Before
- Created train/val/test splits
- Used CLIP for semantic matching
- Complex command-line flags
- Focused on ML training preparation

### After
- Direct class folder output
- Pure visual similarity (no text)
- Simple two-mode interface
- Universal application

**See [TRANSFORMATION_SUMMARY.md](TRANSFORMATION_SUMMARY.md) for details.**

---

## 🎯 Use Cases

Perfect for:
- 🤟 Sign language datasets
- 🏥 Medical imaging (MRI, X-rays)
- 🏭 Industrial defect detection
- 🔬 Microscopy images
- 🎨 Art/style classification
- ❓ Any unlabeled visual dataset

---

## 📦 Installation

```bash
# Minimum
pip install opencv-python scikit-learn numpy pillow tqdm

# Recommended (includes visualization)
pip install -r requirements.txt

# Optional (for DINOv2)
pip install torch torchvision
```

---

## 🔍 Recommended Reading Order

1. **New users:**
   - [QUICK_START.md](QUICK_START.md)
   - [COMMANDS.md](COMMANDS.md)
   - Run `python test_pipeline.py`

2. **Upgrading from old version:**
   - [TRANSFORMATION_SUMMARY.md](TRANSFORMATION_SUMMARY.md)
   - [CHANGES.md](CHANGES.md)
   - [COMMANDS.md](COMMANDS.md)

3. **Deep dive:**
   - [README.md](README.md)
   - Individual module files
   - [config.py](config.py)

---

## 🎓 Core Concepts

### Visual Clustering
Groups images by **appearance**, not meaning. Works universally.

### Two Modes
1. **Named:** Provide class names → `two/`, `three/`, etc.
2. **Numbered:** Provide count → `class_1/`, `class_2/`, etc.

### Learning Loop
Iteratively refines clusters to improve visual consistency.

### No Text Models
No CLIP, no transformers, no semantic understanding needed.

---

## 📞 Quick Help

### See available commands
```bash
python auto_pipeline.py --help
```

### Test your setup
```bash
python test_pipeline.py
```

### Run full example
```bash
python run_example.py
```

---

## 🗂️ File Structure

```
learning_loop_T2/
├── Documentation/
│   ├── QUICK_START.md
│   ├── COMMANDS.md
│   ├── README.md
│   ├── EXAMPLES.md
│   ├── CHANGES.md
│   └── TRANSFORMATION_SUMMARY.md
├── Core Scripts/
│   ├── auto_pipeline.py
│   ├── clustering.py
│   ├── refinement.py
│   ├── feature_extractor.py
│   └── feature_extractor_lightweight.py
├── Helpers/
│   ├── test_pipeline.py
│   ├── run_example.py
│   └── config.py
└── Config/
    └── requirements.txt
```

---

## ✨ Key Benefits

1. **Universal** - Works for any visual domain
2. **Simple** - Two-command interface
3. **Fast** - No heavy text models
4. **Reliable** - Pure visual similarity
5. **Flexible** - Named or numbered classes

---

## 🎉 Get Started

```bash
# Install
pip install -r requirements.txt

# Test
python test_pipeline.py

# Run on your data
python auto_pipeline.py your_images output class1 class2 class3
```

---

**Happy clustering! 🚀**

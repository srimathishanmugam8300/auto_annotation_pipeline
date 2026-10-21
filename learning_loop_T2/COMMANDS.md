# 🚀 Command Reference

## Basic Usage

```bash
# Mode 1: Named classes
python auto_pipeline.py <input_folder> <output_folder> <name1> <name2> <name3> ...

# Mode 2: Numbered classes
python auto_pipeline.py <input_folder> <output_folder> <number>
```

---

## Examples

### Sign Language
```bash
python auto_pipeline.py data/images sign_output two three four six seven eight nine
```

### Medical Imaging (7 types)
```bash
python auto_pipeline.py mri_scans output 7
```

### Industrial Defects
```bash
python auto_pipeline.py factory_images output normal crack scratch dent corrosion
```

### Unknown Dataset
```bash
python auto_pipeline.py mystery_data output 10
```

---

## Output

### Mode 1 Result
```
output/
  two/
    img001.jpg
    img002.jpg
  three/
    img003.jpg
  four/
    img004.jpg
  ...
```

### Mode 2 Result
```
output/
  class_1/
    img001.jpg
  class_2/
    img002.jpg
  class_3/
    img003.jpg
  ...
```

---

## Options

### CPU Mode
```bash
python auto_pipeline.py data output 5 --device cpu
```

---

## Testing

```bash
# Quick test
python test_pipeline.py

# Full example run (both modes)
python run_example.py
```

---

## Dependencies

```bash
# Minimum
pip install opencv-python scikit-learn numpy pillow

# Recommended
pip install -r requirements.txt
```

---

## What You Get

Each run produces:

1. **Class folders** - Organized images by visual similarity
2. **metadata.json** - Class counts and statistics
3. **pipeline_results.json** - Quality metrics

---

## Troubleshooting

### No images found
- Check input folder has `.jpg`, `.png`, or `.jpeg` files
- Use full or relative path

### Wrong cluster count
- Increase/decrease the number or add/remove class names
- Quality metrics in `pipeline_results.json` show clustering quality

### Slow execution
- Use `--device cpu` if no GPU
- Reduce image count for testing

---

## Quick Comparison

| Old Command | New Command |
|-------------|-------------|
| `python auto_pipeline.py data output dog cat --no-semantic --no-splits` | `python auto_pipeline.py data output dog cat` |
| `python auto_pipeline.py data output --n-clusters 5 --no-splits` | `python auto_pipeline.py data output 5` |

**Simpler. Clearer. Better.**

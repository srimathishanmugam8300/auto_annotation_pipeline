# Example Usage Scripts

This folder contains example usage patterns for different scenarios.

## Quick Examples

### 1. Animal Dataset
```bash
python auto_pipeline.py examples/animals output/animals dog cat tiger lion elephant
```

### 2. Sign Language (26 letters)
```bash
python auto_pipeline.py examples/sign_language output/signs \
  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
```

### 3. Medical Imaging (no semantic names)
```bash
python auto_pipeline.py examples/mri_scans output/mri --n-clusters 4 --no-semantic
```

### 4. Industrial Defects
```bash
python auto_pipeline.py examples/factory output/defects normal crack scratch dent rust
```

### 5. Unknown Dataset Exploration
```bash
python auto_pipeline.py examples/unknown output/explored --n-clusters 8
```

### 6. Microscopy Images
```bash
python auto_pipeline.py examples/cells output/cells --n-clusters 6 --no-semantic
```

### 7. Fashion Items
```bash
python auto_pipeline.py examples/fashion output/fashion \
  tshirt trouser pullover dress coat sandal shirt sneaker bag boot
```

### 8. Flowers
```bash
python auto_pipeline.py examples/flowers output/flowers \
  rose tulip sunflower daisy orchid
```

### 9. Vehicles
```bash
python auto_pipeline.py examples/vehicles output/vehicles \
  car truck bus motorcycle bicycle
```

### 10. Food Items
```bash
python auto_pipeline.py examples/food output/food \
  pizza burger sushi pasta salad
```

## Advanced Examples

### Custom Configuration
```python
# modify_config.py
import config

# Use larger model
config.DINOV2_MODEL = "dinov2_vitl14"
config.EMBEDDING_DIM = 1024

# More refinement iterations
config.MAX_ITERATIONS = 20

# Different split ratios
config.TRAIN_SPLIT = 0.8
config.VAL_SPLIT = 0.1
config.TEST_SPLIT = 0.1
```

### Batch Processing Multiple Datasets
```bash
# batch_process.sh
#!/bin/bash

datasets=("animals" "signs" "flowers" "vehicles")
for dataset in "${datasets[@]}"; do
    python auto_pipeline.py "data/$dataset" "output/$dataset" --n-clusters 5
done
```

### Integration with Training Pipeline
```python
# train_on_annotated.py
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load auto-annotated dataset
dataset_path = Path("output/animals/train")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Your training code here
model = YourModel(num_classes=len(train_dataset.classes))
# ... training loop ...
```

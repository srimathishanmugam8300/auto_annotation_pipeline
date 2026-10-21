"""
Dataset Organizer
Creates organized labeled dataset with train/val/test splits
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict
import config


class DatasetOrganizer:
    """
    Organizes labeled images into a structured dataset format.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize dataset organizer.
        
        Args:
            output_dir: Root output directory
        """
        self.output_dir = Path(output_dir)
        self.metadata = {
            'total_images': 0,
            'n_classes': 0,
            'class_distribution': {},
            'splits': {}
        }
    
    def organize_dataset(
        self,
        image_paths: List[str],
        cluster_labels: np.ndarray,
        cluster_to_name: Dict[int, str],
        create_splits: bool = True
    ):
        """
        Organize images into labeled folders.
        
        Args:
            image_paths: List of image file paths
            cluster_labels: Cluster label for each image
            cluster_to_name: Mapping from cluster ID to class name
            create_splits: Whether to create train/val/test splits
        """
        print(f"\n📁 Organizing dataset into {self.output_dir}...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group images by class
        class_images = defaultdict(list)
        for img_path, label in zip(image_paths, cluster_labels):
            class_name = cluster_to_name.get(int(label), f"Class_{label}")
            class_images[class_name].append(img_path)
        
        self.metadata['n_classes'] = len(class_images)
        self.metadata['total_images'] = len(image_paths)
        
        if create_splits:
            self._create_split_structure(class_images)
        else:
            self._create_simple_structure(class_images)
        
        # Save metadata
        self._save_metadata()
        
        # Generate report
        self._generate_report(class_images)
        
        print(f"✓ Dataset organized successfully!")
        print(f"  Location: {self.output_dir}")
    
    def _create_simple_structure(self, class_images: Dict[str, List[str]]):
        """
        Create simple class folder structure (no splits).
        
        Structure:
        output/
          class_1/
            image_001.jpg
            image_002.jpg
          class_2/
            ...
        """
        print("  Creating class folders...")
        
        for class_name, img_paths in class_images.items():
            class_dir = self.output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Copy images
            for img_path in img_paths:
                src = Path(img_path)
                dst = class_dir / src.name
                
                # Handle duplicate names
                if dst.exists():
                    dst = class_dir / f"{src.stem}_{hash(img_path) % 10000}{src.suffix}"
                
                shutil.copy2(src, dst)
            
            self.metadata['class_distribution'][class_name] = len(img_paths)
            print(f"    {class_name}: {len(img_paths)} images")
    
    def _create_split_structure(self, class_images: Dict[str, List[str]]):
        """
        Create train/val/test split structure.
        
        Structure:
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
        """
        print("  Creating train/val/test splits...")
        
        splits = ['train', 'val', 'test']
        split_ratios = [config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT]
        
        # Create split directories
        for split in splits:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        split_counts = {split: {} for split in splits}
        
        for class_name, img_paths in class_images.items():
            # Shuffle images
            img_paths = np.array(img_paths)
            np.random.shuffle(img_paths)
            
            n_total = len(img_paths)
            n_train = int(n_total * split_ratios[0])
            n_val = int(n_total * split_ratios[1])
            
            # Split indices
            train_imgs = img_paths[:n_train]
            val_imgs = img_paths[n_train:n_train + n_val]
            test_imgs = img_paths[n_train + n_val:]
            
            # Copy images to respective splits
            for split, imgs in zip(splits, [train_imgs, val_imgs, test_imgs]):
                split_class_dir = self.output_dir / split / class_name
                split_class_dir.mkdir(exist_ok=True)
                
                for img_path in imgs:
                    src = Path(img_path)
                    dst = split_class_dir / src.name
                    
                    # Handle duplicate names
                    if dst.exists():
                        dst = split_class_dir / f"{src.stem}_{hash(img_path) % 10000}{src.suffix}"
                    
                    shutil.copy2(src, dst)
                
                split_counts[split][class_name] = len(imgs)
            
            # Update metadata
            self.metadata['class_distribution'][class_name] = n_total
            print(f"    {class_name}: {n_train} train, {n_val} val, {n_total - n_train - n_val} test")
        
        self.metadata['splits'] = split_counts
    
    def _save_metadata(self):
        """Save dataset metadata to JSON"""
        metadata_path = self.output_dir / "metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n  ✓ Metadata saved to {metadata_path.name}")
    
    def _generate_report(self, class_images: Dict[str, List[str]]):
        """Generate human-readable report"""
        report_path = self.output_dir / "dataset_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET ORGANIZATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Images: {self.metadata['total_images']}\n")
            f.write(f"Number of Classes: {self.metadata['n_classes']}\n\n")
            
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            for class_name, count in sorted(self.metadata['class_distribution'].items()):
                percentage = 100 * count / self.metadata['total_images']
                f.write(f"  {class_name:20s}: {count:5d} ({percentage:5.1f}%)\n")
            
            if self.metadata.get('splits'):
                f.write("\n\nSPLIT DISTRIBUTION:\n")
                f.write("-" * 40 + "\n")
                
                for split in ['train', 'val', 'test']:
                    if split in self.metadata['splits']:
                        total = sum(self.metadata['splits'][split].values())
                        f.write(f"\n{split.upper()}:\n")
                        for class_name, count in sorted(self.metadata['splits'][split].items()):
                            f.write(f"  {class_name:20s}: {count:5d}\n")
                        f.write(f"  {'TOTAL':20s}: {total:5d}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Dataset ready for training!\n")
            f.write("=" * 60 + "\n")
        
        print(f"  ✓ Report saved to {report_path.name}")


def create_data_loader_example(output_dir: Path):
    """
    Create example code for loading the dataset.
    """
    example_path = output_dir / "load_dataset_example.py"
    
    example_code = '''"""
Example code for loading the organized dataset
"""

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load datasets
dataset_root = Path(__file__).parent

train_dataset = datasets.ImageFolder(
    root=dataset_root / "train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=dataset_root / "val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=dataset_root / "test",
    transform=transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Print dataset info
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")
'''
    
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"  ✓ Example loader saved to {example_path.name}")


if __name__ == "__main__":
    # Test organizer
    print("Testing dataset organizer...")
    
    # Create dummy data
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "organized_dataset"
        
        # Simulate some image paths and labels
        image_paths = [f"image_{i}.jpg" for i in range(100)]
        labels = np.random.randint(0, 3, 100)
        cluster_to_name = {0: "cat", 1: "dog", 2: "bird"}
        
        # Note: This test won't actually copy files since they don't exist
        print("  (Skipping actual file operations in test mode)")
        
        print("\n✓ Organizer test complete!")

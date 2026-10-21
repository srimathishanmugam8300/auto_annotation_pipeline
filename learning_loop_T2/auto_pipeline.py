"""
Universal Visual Clustering Pipeline
Pure visual organization - no text, no training, no splits

Usage:
  # Mode 1: With class names
  python auto_pipeline.py data/images sign_output two three four six seven eight nine
  
  # Mode 2: With number of classes
  python auto_pipeline.py data/images sign_output 7

Output:
  sign_output/
    two/ (or class_1/)
    three/ (or class_2/)
    four/ (or class_3/)
    ...
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
import shutil
from datetime import datetime

# Try DINOv2 first, fallback to lightweight extractor
try:
    from feature_extractor import DINOv2Extractor, load_image_paths
    DINOV2_AVAILABLE = True
except Exception:
    DINOV2_AVAILABLE = False

# Lightweight extractor (always available)
from feature_extractor_lightweight import LightweightExtractor, load_image_paths

from clustering import VisualClusterer, compute_clustering_quality
from refinement import ClusterRefinementLoop, analyze_cluster_purity
import config


class VisualClusteringPipeline:
    """
    Pure visual clustering pipeline.
    
    Groups images by visual similarity → outputs to class folders.
    No text models, no training logic, no train/val/test splits.
    """
    
    def __init__(self, input_folder: Path, output_folder: Path):
        """
        Initialize pipeline.
        
        Args:
            input_folder: Folder containing unlabeled images
            output_folder: Folder to save organized classes
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        if not self.input_folder.exists():
            raise ValueError(f"Input folder not found: {input_folder}")
        
        print("=" * 60)
        print("VISUAL CLUSTERING PIPELINE")
        print("=" * 60)
        print(f"\nInput:  {self.input_folder}")
        print(f"Output: {self.output_folder}")
        print("\nMode: Pure visual similarity (no text, no training)\n")
        
        self.results = {
            'start_time': datetime.now().isoformat(),
            'input_folder': str(self.input_folder),
            'output_folder': str(self.output_folder)
        }
    
    def run(self, class_names: list = None, n_clusters: int = None):
        """
        Run the visual clustering pipeline.
        
        Args:
            class_names: List of class names (Mode 1)
            n_clusters: Number of clusters (Mode 2)
        """
        try:
            # Step 1: Load images
            print("\n" + "─" * 60)
            print("STEP 1: Loading Images")
            print("─" * 60)
            image_paths = load_image_paths(self.input_folder)
            
            if len(image_paths) == 0:
                raise ValueError("No valid images found in input folder")
            
            print(f"✓ Found {len(image_paths)} images")
            self.results['n_images'] = len(image_paths)
            
            # Step 2: Extract visual features
            print("\n" + "─" * 60)
            print("STEP 2: Visual Feature Extraction")
            print("─" * 60)
            
            # Try DINOv2, fallback to lightweight
            try:
                if DINOV2_AVAILABLE:
                    print("Using DINOv2 (deep learning)...")
                    extractor = DINOv2Extractor()
                else:
                    raise ImportError("DINOv2 not available")
            except Exception as e:
                print(f"⚠️  DINOv2 unavailable: {e}")
                print("📦 Using lightweight extractor (classical CV)...")
                extractor = LightweightExtractor()
            
            embeddings, valid_paths = extractor.extract_batch_features(image_paths)
            
            print(f"✓ Extracted features for {len(valid_paths)} images")
            print(f"  Embedding dimension: {embeddings.shape[1]}")
            
            self.results['n_valid_images'] = len(valid_paths)
            self.results['embedding_dim'] = embeddings.shape[1]
            
            # Step 3: Visual clustering
            print("\n" + "─" * 60)
            print("STEP 3: Unsupervised Clustering")
            print("─" * 60)
            
            # Determine number of clusters
            if n_clusters is None and class_names:
                n_clusters = len(class_names)
            
            if n_clusters is None:
                raise ValueError("Must specify either class names or number of clusters")
            
            clusterer = VisualClusterer(n_clusters=n_clusters)
            initial_labels = clusterer.fit(embeddings)
            
            n_clusters_found = len(np.unique(initial_labels))
            print(f"✓ Created {n_clusters_found} clusters")
            
            self.results['n_clusters'] = n_clusters_found
            
            # Compute initial quality
            quality_metrics = compute_clustering_quality(embeddings, initial_labels)
            self.results['initial_quality'] = quality_metrics
            
            # Step 4: Cluster refinement (learning loop)
            print("\n" + "─" * 60)
            print("STEP 4: Learning Loop Refinement")
            print("─" * 60)
            
            refiner = ClusterRefinementLoop()
            refined_labels, prototypes = refiner.refine(embeddings, initial_labels)
            
            print(f"✓ Refinement completed in {len(refiner.history)} iterations")
            
            self.results['refinement_iterations'] = len(refiner.history)
            
            # Compute refined quality
            refined_quality = refiner.get_cluster_quality_metrics(
                embeddings, refined_labels, prototypes
            )
            self.results['refined_quality'] = refined_quality
            
            # Step 5: Map clusters to class names
            print("\n" + "─" * 60)
            print("STEP 5: Class Name Assignment")
            print("─" * 60)
            
            unique_clusters = sorted(np.unique(refined_labels))
            
            if class_names and len(class_names) == len(unique_clusters):
                # Mode 1: Use provided names
                cluster_to_name = {
                    int(cluster_id): class_names[i]
                    for i, cluster_id in enumerate(unique_clusters)
                }
                print("✓ Using provided class names")
            else:
                # Mode 2: Generate class_1, class_2, etc.
                cluster_to_name = {
                    int(cluster_id): f"class_{i+1}"
                    for i, cluster_id in enumerate(unique_clusters)
                }
                print("✓ Using generated class names (class_1, class_2, ...)")
            
            for cluster_id, name in cluster_to_name.items():
                count = np.sum(refined_labels == cluster_id)
                print(f"  Cluster {cluster_id} → {name} ({count} images)")
            
            self.results['class_mapping'] = cluster_to_name
            
            # Step 6: Organize into class folders
            print("\n" + "─" * 60)
            print("STEP 6: Creating Class Folders")
            print("─" * 60)
            
            self._organize_into_folders(valid_paths, refined_labels, cluster_to_name)
            
            # Save pipeline results
            self._save_results()
            
            # Final summary
            self._print_summary(cluster_to_name, refined_labels)
            
            print("\n" + "=" * 60)
            print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"\nOutput: {self.output_folder}")
            print("\nFolder structure:")
            for name in sorted(cluster_to_name.values()):
                print(f"  {self.output_folder.name}/{name}/")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _organize_into_folders(self, image_paths: list, labels: np.ndarray, 
                                cluster_to_name: dict):
        """
        Organize images into class folders (no train/val/test splits).
        
        Args:
            image_paths: List of image file paths
            labels: Cluster labels for each image
            cluster_to_name: Mapping from cluster ID to class name
        """
        # Create output folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Create class folders and copy images
        class_counts = {}
        
        for img_path, label in zip(image_paths, labels):
            class_name = cluster_to_name[int(label)]
            class_folder = self.output_folder / class_name
            class_folder.mkdir(exist_ok=True)
            
            # Copy image
            dest_path = class_folder / Path(img_path).name
            
            # Handle duplicate names
            counter = 1
            while dest_path.exists():
                stem = Path(img_path).stem
                suffix = Path(img_path).suffix
                dest_path = class_folder / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(img_path, dest_path)
            
            # Track counts
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n✓ Organized {len(image_paths)} images into {len(cluster_to_name)} folders")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} images")
        
        # Save metadata
        metadata = {
            'total_images': len(image_paths),
            'n_classes': len(cluster_to_name),
            'class_counts': class_counts,
            'class_names': list(cluster_to_name.values())
        }
        
        metadata_path = self.output_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved metadata to {metadata_path.name}")
    
    def _save_results(self):
        """Save pipeline results to JSON"""
        self.results['end_time'] = datetime.now().isoformat()
        
        results_path = self.output_folder / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  ✓ Pipeline results saved to {results_path.name}")
    
    def _print_summary(self, cluster_to_name: dict, labels: np.ndarray):
        """Print final summary"""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n✓ Processed: {self.results['n_valid_images']} images")
        print(f"✓ Created: {self.results['n_clusters']} classes")
        
        print("\nClass Distribution:")
        for cluster_id, class_name in sorted(cluster_to_name.items()):
            count = np.sum(labels == cluster_id)
            print(f"  {class_name}: {count} images")
        
        if self.results.get('refined_quality'):
            print("\nClustering Quality:")
            for metric, value in self.results['refined_quality'].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Visual Clustering Pipeline - Pure visual organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: With class names
  python auto_pipeline.py data/images sign_output two three four six seven eight nine
  
  # Mode 2: With number of classes
  python auto_pipeline.py data/images sign_output 7

Output Structure:
  sign_output/
    two/ (or class_1/)
    three/ (or class_2/)
    four/ (or class_3/)
    ...

Features:
  ✓ Pure visual similarity clustering
  ✓ No text models (no CLIP, no transformers)
  ✓ No training/validation/test splits
  ✓ Direct class folder output
  ✓ Works for any visual domain (signs, MRI, X-rays, etc.)
        """
    )
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Folder containing unlabeled images'
    )
    
    parser.add_argument(
        'output_folder',
        type=str,
        help='Folder to save organized classes'
    )
    
    parser.add_argument(
        'class_info',
        nargs='+',
        help='Either class names (mode 1) or single number (mode 2)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Update config if device specified
    if args.device:
        config.DEVICE = args.device
    
    # Parse class_info: check if it's a single number or list of names
    class_names = None
    n_clusters = None
    
    if len(args.class_info) == 1 and args.class_info[0].isdigit():
        # Mode 2: Single number
        n_clusters = int(args.class_info[0])
        print(f"\n📊 Mode 2: Creating {n_clusters} classes (class_1, class_2, ...)")
    else:
        # Mode 1: Class names
        class_names = args.class_info
        print(f"\n📊 Mode 1: Using {len(class_names)} provided class names")
    
    # Create and run pipeline
    pipeline = VisualClusteringPipeline(
        args.input_folder,
        args.output_folder
    )
    
    pipeline.run(
        class_names=class_names,
        n_clusters=n_clusters
    )


if __name__ == "__main__":
    main()

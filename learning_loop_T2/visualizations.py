"""
Visualization utilities for the pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import umap


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Visual Clusters",
    method: str = "umap"
):
    """
    Visualize high-dimensional embeddings in 2D.
    
    Args:
        embeddings: Feature embeddings (n_samples, n_features)
        labels: Cluster labels
        output_path: Path to save visualization
        title: Plot title
        method: Dimensionality reduction method ('umap' or 'tsne')
    """
    print(f"📊 Creating {method.upper()} visualization...")
    
    # Reduce to 2D
    if method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f"Cluster {label}",
            alpha=0.6,
            s=50
        )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(f"{method.upper()} Dimension 1", fontsize=12)
    plt.ylabel(f"{method.upper()} Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {output_path.name}")


def plot_cluster_sizes(
    labels: np.ndarray,
    output_path: Path,
    class_names: dict = None
):
    """
    Plot cluster size distribution.
    
    Args:
        labels: Cluster labels
        output_path: Path to save plot
        class_names: Optional mapping from cluster ID to name
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    
    # Create labels
    if class_names:
        bar_labels = [class_names.get(int(label), f"Cluster {label}") 
                     for label in unique_labels]
    else:
        bar_labels = [f"Cluster {label}" for label in unique_labels]
    
    # Create bar plot
    bars = plt.bar(range(len(unique_labels)), counts, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique_labels)), bar_labels, rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Cluster sizes saved to {output_path.name}")


def plot_refinement_history(
    history: list,
    output_path: Path
):
    """
    Plot learning loop refinement history.
    
    Args:
        history: List of iteration statistics
        output_path: Path to save plot
    """
    iterations = [h['iteration'] for h in history]
    n_changed = [h['n_changed'] for h in history]
    change_ratios = [h['change_ratio'] * 100 for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of changes
    ax1.plot(iterations, n_changed, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Reassignments', fontsize=12, fontweight='bold')
    ax1.set_title('Refinement Progress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Change ratio
    ax2.plot(iterations, change_ratios, marker='s', linewidth=2, 
             markersize=8, color='coral')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Change Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Refinement history saved to {output_path.name}")


def create_sample_grid(
    image_paths: list,
    labels: np.ndarray,
    cluster_id: int,
    output_path: Path,
    n_samples: int = 16,
    title: str = None
):
    """
    Create a grid of sample images from a cluster.
    
    Args:
        image_paths: List of all image paths
        labels: Cluster labels
        cluster_id: Which cluster to visualize
        output_path: Path to save grid
        n_samples: Number of samples to show
        title: Optional title
    """
    from PIL import Image
    
    # Get images from this cluster
    cluster_mask = labels == cluster_id
    cluster_paths = np.array(image_paths)[cluster_mask]
    
    # Sample randomly
    if len(cluster_paths) > n_samples:
        indices = np.random.choice(len(cluster_paths), n_samples, replace=False)
        cluster_paths = cluster_paths[indices]
    
    # Determine grid size
    n_cols = 4
    n_rows = int(np.ceil(len(cluster_paths) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, ax in enumerate(axes):
        if i < len(cluster_paths):
            try:
                img = Image.open(cluster_paths[i])
                ax.imshow(img)
                ax.axis('off')
            except:
                ax.axis('off')
        else:
            ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Sample grid saved to {output_path.name}")


if __name__ == "__main__":
    print("✓ Visualization utilities loaded")

"""
Learning Loop for Cluster Refinement
Iteratively improves cluster quality through prototype updates
"""

import numpy as np
from typing import Tuple, List
import config


class ClusterRefinementLoop:
    """
    Refines clusters through iterative prototype-based learning.
    
    Process:
    1. Compute cluster prototypes (centroids)
    2. Reassign samples to nearest prototype
    3. Update prototypes with momentum
    4. Repeat until convergence
    """
    
    def __init__(
        self,
        max_iterations: int = None,
        convergence_threshold: float = None,
        momentum: float = None
    ):
        """
        Initialize refinement loop.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_threshold: Stop if fewer than this fraction of samples change
            momentum: Exponential moving average coefficient for prototype updates
        """
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self.convergence_threshold = convergence_threshold or config.CONVERGENCE_THRESHOLD
        self.momentum = momentum or config.PROTOTYPE_UPDATE_MOMENTUM
        
        self.history = []
    
    def refine(
        self,
        embeddings: np.ndarray,
        initial_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine cluster assignments through iterative learning.
        
        Args:
            embeddings: Feature embeddings (n_samples, n_features)
            initial_labels: Initial cluster assignments
            
        Returns:
            Tuple of (refined_labels, final_prototypes)
        """
        print(f"\n🔄 Starting cluster refinement loop...")
        print(f"   Max iterations: {self.max_iterations}")
        print(f"   Convergence threshold: {self.convergence_threshold*100:.1f}%")
        
        labels = initial_labels.copy()
        n_samples = len(labels)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Initialize prototypes
        prototypes = self._compute_prototypes(embeddings, labels, unique_labels)
        
        for iteration in range(1, self.max_iterations + 1):
            # Store old labels to check convergence
            old_labels = labels.copy()
            
            # Reassign samples to nearest prototype
            labels = self._reassign_to_nearest(embeddings, prototypes, unique_labels)
            
            # Update prototypes with momentum
            new_prototypes = self._compute_prototypes(embeddings, labels, unique_labels)
            prototypes = self.momentum * prototypes + (1 - self.momentum) * new_prototypes
            
            # Check convergence
            n_changed = np.sum(labels != old_labels)
            change_ratio = n_changed / n_samples
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'n_changed': n_changed,
                'change_ratio': change_ratio
            })
            
            print(f"   Iteration {iteration}: {n_changed} samples reassigned ({change_ratio*100:.2f}%)")
            
            # Check for convergence
            if change_ratio < self.convergence_threshold:
                print(f"✓ Converged after {iteration} iterations!")
                break
        else:
            print(f"✓ Completed {self.max_iterations} iterations")
        
        # Final statistics
        self._print_final_stats(labels, unique_labels)
        
        return labels, prototypes
    
    def _compute_prototypes(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute cluster prototypes (centroids).
        
        Args:
            embeddings: Feature embeddings
            labels: Cluster labels
            unique_labels: Array of unique label values
            
        Returns:
            Cluster prototypes (n_clusters, n_features)
        """
        prototypes = np.array([
            embeddings[labels == label].mean(axis=0)
            for label in unique_labels
        ])
        return prototypes
    
    def _reassign_to_nearest(
        self,
        embeddings: np.ndarray,
        prototypes: np.ndarray,
        unique_labels: np.ndarray
    ) -> np.ndarray:
        """
        Reassign each sample to its nearest prototype.
        
        Args:
            embeddings: Feature embeddings
            prototypes: Cluster prototypes
            unique_labels: Array of unique label values
            
        Returns:
            New cluster labels
        """
        # Compute distances to all prototypes
        distances = np.linalg.norm(
            embeddings[:, np.newaxis, :] - prototypes[np.newaxis, :, :],
            axis=2
        )
        
        # Assign to nearest
        nearest_indices = np.argmin(distances, axis=1)
        new_labels = unique_labels[nearest_indices]
        
        return new_labels
    
    def _print_final_stats(self, labels: np.ndarray, unique_labels: np.ndarray):
        """Print final cluster statistics"""
        print("\n  Final cluster distribution:")
        for label in unique_labels:
            count = np.sum(labels == label)
            percentage = 100 * count / len(labels)
            print(f"    Cluster {label}: {count} images ({percentage:.1f}%)")
    
    def get_cluster_quality_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        prototypes: np.ndarray
    ) -> dict:
        """
        Compute quality metrics for refined clusters.
        
        Args:
            embeddings: Feature embeddings
            labels: Cluster labels
            prototypes: Cluster prototypes
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Intra-cluster compactness
        intra_cluster_distances = []
        for label in np.unique(labels):
            cluster_embeddings = embeddings[labels == label]
            prototype = prototypes[label]
            distances = np.linalg.norm(cluster_embeddings - prototype, axis=1)
            intra_cluster_distances.extend(distances)
        
        metrics['mean_intra_cluster_distance'] = np.mean(intra_cluster_distances)
        metrics['std_intra_cluster_distance'] = np.std(intra_cluster_distances)
        
        # Inter-cluster separation
        inter_cluster_distances = []
        n_clusters = len(prototypes)
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                distance = np.linalg.norm(prototypes[i] - prototypes[j])
                inter_cluster_distances.append(distance)
        
        metrics['mean_inter_cluster_distance'] = np.mean(inter_cluster_distances)
        metrics['std_inter_cluster_distance'] = np.std(inter_cluster_distances)
        
        # Separation ratio (higher is better)
        if metrics['mean_intra_cluster_distance'] > 0:
            metrics['separation_ratio'] = (
                metrics['mean_inter_cluster_distance'] / 
                metrics['mean_intra_cluster_distance']
            )
        else:
            metrics['separation_ratio'] = float('inf')
        
        return metrics


def analyze_cluster_purity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_samples_per_cluster: int = 10
) -> dict:
    """
    Analyze the purity of each cluster by examining sample similarity.
    
    Args:
        embeddings: Feature embeddings
        labels: Cluster labels
        n_samples_per_cluster: Number of samples to analyze per cluster
        
    Returns:
        Dictionary with purity analysis for each cluster
    """
    purity_analysis = {}
    
    for label in np.unique(labels):
        cluster_embeddings = embeddings[labels == label]
        
        if len(cluster_embeddings) < 2:
            continue
        
        # Compute pairwise similarities within cluster
        cluster_center = cluster_embeddings.mean(axis=0)
        distances_to_center = np.linalg.norm(
            cluster_embeddings - cluster_center,
            axis=1
        )
        
        purity_analysis[int(label)] = {
            'size': len(cluster_embeddings),
            'mean_distance_to_center': float(np.mean(distances_to_center)),
            'std_distance_to_center': float(np.std(distances_to_center)),
            'compactness_score': float(1.0 / (1.0 + np.mean(distances_to_center)))
        }
    
    return purity_analysis


if __name__ == "__main__":
    # Test refinement loop
    print("Testing cluster refinement...")
    
    # Generate synthetic data with 3 clusters
    np.random.seed(42)
    n_per_cluster = 100
    n_features = 768
    
    # Create 3 distinct clusters with some noise
    cluster1 = np.random.randn(n_per_cluster, n_features) + np.array([5, 0] + [0]*(n_features-2))
    cluster2 = np.random.randn(n_per_cluster, n_features) + np.array([0, 5] + [0]*(n_features-2))
    cluster3 = np.random.randn(n_per_cluster, n_features) + np.array([-5, -5] + [0]*(n_features-2))
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    
    # Create noisy initial labels
    true_labels = np.array([0]*n_per_cluster + [1]*n_per_cluster + [2]*n_per_cluster)
    noise = np.random.randint(0, 3, len(true_labels))
    noisy_labels = np.where(np.random.rand(len(true_labels)) < 0.2, noise, true_labels)
    
    print(f"Initial accuracy: {np.mean(noisy_labels == true_labels)*100:.1f}%")
    
    # Refine
    refiner = ClusterRefinementLoop(max_iterations=10)
    refined_labels, prototypes = refiner.refine(embeddings, noisy_labels)
    
    # Check improvement (need to align labels)
    print(f"\n✓ Refinement complete!")
    
    # Compute quality metrics
    metrics = refiner.get_cluster_quality_metrics(embeddings, refined_labels, prototypes)
    print(f"\nQuality metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

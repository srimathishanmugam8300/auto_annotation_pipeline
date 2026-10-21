"""
Unsupervised Clustering Module
Groups images based purely on visual similarity
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap
from typing import Tuple, Optional
import config


class VisualClusterer:
    """
    Clusters image embeddings using unsupervised learning.
    No text, no labels, purely visual structure.
    """
    
    def __init__(self, n_clusters: Optional[int] = None, method: str = "auto"):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters (if known). None for auto-detection.
            method: "kmeans", "hdbscan", or "auto"
        """
        self.n_clusters = n_clusters
        self.method = method
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.labels_ = None
        
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings into visual groups.
        
        Args:
            embeddings: Feature matrix (n_samples, n_features)
            
        Returns:
            Cluster labels for each sample
        """
        print(f"\n🔍 Clustering {embeddings.shape[0]} images...")
        
        # Normalize features
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Choose clustering method
        if self.method == "kmeans" or (self.method == "auto" and self.n_clusters is not None):
            labels = self._kmeans_clustering(embeddings_scaled)
        elif self.method == "hdbscan" or (self.method == "auto" and self.n_clusters is None):
            labels = self._hdbscan_clustering(embeddings_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        self.labels_ = labels
        
        # Print clustering statistics
        self._print_statistics(labels)
        
        return labels
    
    def _kmeans_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        K-Means clustering (fixed number of clusters).
        
        Args:
            embeddings: Normalized embeddings
            
        Returns:
            Cluster labels
        """
        print(f"  Method: K-Means (k={self.n_clusters})")
        
        self.cluster_model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        
        labels = self.cluster_model.fit_predict(embeddings)
        return labels
    
    def _hdbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        HDBSCAN clustering (automatic cluster discovery).
        
        Args:
            embeddings: Normalized embeddings
            
        Returns:
            Cluster labels
        """
        print("  Method: HDBSCAN (automatic)")
        
        # Optional: Reduce dimensionality with UMAP for better clustering
        if embeddings.shape[1] > 100:
            print("  Reducing dimensionality with UMAP...")
            reducer = umap.UMAP(
                n_components=50,
                metric='cosine',
                random_state=42
            )
            embeddings = reducer.fit_transform(embeddings)
        
        self.cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=config.MIN_CLUSTER_SIZE,
            min_samples=config.MIN_SAMPLES,
            cluster_selection_epsilon=config.CLUSTER_SELECTION_EPSILON,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = self.cluster_model.fit_predict(embeddings)
        
        # Handle noise points (label -1)
        n_noise = np.sum(labels == -1)
        if n_noise > 0:
            print(f"  ⚠️  {n_noise} images marked as noise (will be assigned to nearest cluster)")
            labels = self._assign_noise_points(embeddings, labels)
        
        return labels
    
    def _assign_noise_points(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Assign noise points to nearest cluster centroid.
        
        Args:
            embeddings: Feature embeddings
            labels: Cluster labels with -1 for noise
            
        Returns:
            Updated labels with no noise points
        """
        # Find noise points
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0]
        
        if len(noise_indices) == 0:
            return labels
        
        # Compute cluster centroids
        unique_labels = np.unique(labels[~noise_mask])
        centroids = np.array([
            embeddings[labels == label].mean(axis=0)
            for label in unique_labels
        ])
        
        # Assign each noise point to nearest centroid
        for idx in noise_indices:
            distances = np.linalg.norm(centroids - embeddings[idx], axis=1)
            nearest_cluster = unique_labels[np.argmin(distances)]
            labels[idx] = nearest_cluster
        
        return labels
    
    def _print_statistics(self, labels: np.ndarray):
        """Print clustering statistics"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        print(f"✓ Discovered {n_clusters} visual clusters")
        
        # Cluster sizes
        print("\n  Cluster sizes:")
        for label in unique_labels:
            count = np.sum(labels == label)
            percentage = 100 * count / len(labels)
            print(f"    Cluster {label}: {count} images ({percentage:.1f}%)")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign new embeddings to existing clusters.
        
        Args:
            embeddings: New feature embeddings
            
        Returns:
            Cluster labels
        """
        if self.cluster_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        
        if isinstance(self.cluster_model, KMeans):
            return self.cluster_model.predict(embeddings_scaled)
        else:
            # For HDBSCAN, use approximate membership
            labels, _ = hdbscan.approximate_predict(self.cluster_model, embeddings_scaled)
            return labels
    
    def get_cluster_prototypes(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute prototype (centroid) for each cluster.
        
        Args:
            embeddings: Feature embeddings
            
        Returns:
            Cluster prototypes (n_clusters, n_features)
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        unique_labels = np.unique(self.labels_)
        prototypes = np.array([
            embeddings[self.labels_ == label].mean(axis=0)
            for label in unique_labels
        ])
        
        return prototypes


def compute_clustering_quality(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: Feature embeddings
        labels: Cluster labels
        
    Returns:
        Dictionary of quality metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    metrics = {}
    
    try:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
    except Exception as e:
        print(f"⚠️  Error computing quality metrics: {e}")
    
    return metrics


if __name__ == "__main__":
    # Test clustering
    print("Testing clustering module...")
    
    # Generate synthetic embeddings
    np.random.seed(42)
    n_samples = 500
    n_features = 768
    embeddings = np.random.randn(n_samples, n_features)
    
    # Test K-Means
    print("\n--- K-Means ---")
    clusterer = VisualClusterer(n_clusters=5, method="kmeans")
    labels = clusterer.fit(embeddings)
    
    # Test HDBSCAN
    print("\n--- HDBSCAN ---")
    clusterer = VisualClusterer(method="hdbscan")
    labels = clusterer.fit(embeddings)
    
    print("\n✓ Clustering test complete!")

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class ClusterEngine:
    """
    Embeddings -> Normalize -> PCA -> KMeans (auto-k or forced-k)
    Optional centroid merge is conservative to avoid collapsing all samples into one bucket.
    """

    def __init__(
        self,
        num_clusters=None,
        max_clusters=20,
        refine_clusters=False,
        similarity_threshold=0.98,
        tiny_cluster_size=1,
        balance_clusters=False,
        silhouette_sample_size=2000,
        minibatch_threshold=4000,
        kmeans_batch_size=2048,
    ):
        self.num_clusters = num_clusters
        self.max_clusters = max_clusters
        self.refine_clusters = refine_clusters
        self.similarity_threshold = similarity_threshold
        self.tiny_cluster_size = max(1, tiny_cluster_size)
        self.balance_clusters = balance_clusters
        self.silhouette_sample_size = None if silhouette_sample_size is None else int(silhouette_sample_size)
        self.minibatch_threshold = max(0, int(minibatch_threshold))
        self.kmeans_batch_size = max(256, int(kmeans_batch_size))

    def _reindex_labels(self, labels):
        unique_labels = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique_labels)}
        return np.array([remap[lbl] for lbl in labels], dtype=int)

    def _use_minibatch_kmeans(self, n_samples):
        return self.minibatch_threshold > 0 and n_samples >= self.minibatch_threshold

    def _make_clusterer(self, n_clusters, n_samples, n_init):
        if self._use_minibatch_kmeans(n_samples):
            return MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=n_init,
                batch_size=self.kmeans_batch_size,
                max_no_improvement=20,
                reassignment_ratio=0.01,
            )
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init)

    def _resolve_silhouette_sample_size(self, n_samples):
        if self.silhouette_sample_size is None or self.silhouette_sample_size <= 0:
            return None
        if n_samples <= self.silhouette_sample_size:
            return None
        return int(self.silhouette_sample_size)

    def score_clusters(self, embeddings, labels, prefix=""):
        sample_size = self._resolve_silhouette_sample_size(len(labels))
        if sample_size is not None:
            print(
                f"{prefix}Using sampled silhouette score on "
                f"{sample_size}/{len(labels)} samples."
            )

        kwargs = {}
        if sample_size is not None:
            kwargs["sample_size"] = sample_size
            kwargs["random_state"] = 42

        return float(silhouette_score(embeddings, labels, **kwargs))

    def _select_k(self, reduced_embeddings, n_samples):
        if self.num_clusters is not None:
            forced_k = max(1, min(int(self.num_clusters), n_samples))
            print(f"Forcing {forced_k} clusters...")
            return forced_k

        if n_samples <= 2:
            return n_samples

        max_k = min(max(2, self.max_clusters), n_samples - 1)
        best_score = -1.0
        best_k = 2

        print("Detecting optimal cluster count using silhouette score...")
        for k in range(2, max_k + 1):
            clusterer = self._make_clusterer(k, n_samples, n_init=20)
            candidate_labels = clusterer.fit_predict(reduced_embeddings)
            unique_count = len(np.unique(candidate_labels))

            # Silhouette is invalid when all points collapse into one cluster.
            if unique_count < 2:
                continue

            score = self.score_clusters(reduced_embeddings, candidate_labels, prefix=f"k={k}: ")
            if score > best_score:
                best_score = score
                best_k = k

        print(f"Selected k={best_k} (silhouette={best_score:.4f})")
        return best_k

    def _refine_by_centroid_similarity(self, labels, normalized_embeddings):
        print("Refinement enabled: attempting conservative tiny-cluster merging...")

        while True:
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 2:
                break

            cluster_sizes = {
                lbl: int(np.sum(labels == lbl))
                for lbl in unique_labels
            }

            centroids = []
            for lbl in unique_labels:
                pts = normalized_embeddings[labels == lbl]
                centroids.append(np.mean(pts, axis=0))

            centroids = normalize(np.array(centroids))
            similarity_matrix = cosine_similarity(centroids)
            np.fill_diagonal(similarity_matrix, -1.0)

            i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            max_sim = float(similarity_matrix[i, j])
            if max_sim < self.similarity_threshold:
                break

            lbl_i = unique_labels[i]
            lbl_j = unique_labels[j]

            # Only merge if one cluster is tiny, preventing large-cluster collapse.
            if (
                cluster_sizes[lbl_i] > self.tiny_cluster_size
                and cluster_sizes[lbl_j] > self.tiny_cluster_size
            ):
                break

            # Merge the smaller tiny cluster into the larger neighbor.
            source, target = (lbl_i, lbl_j)
            if cluster_sizes[lbl_i] > cluster_sizes[lbl_j]:
                source, target = (lbl_j, lbl_i)

            print(
                f"Merging tiny cluster {source} -> {target} "
                f"(similarity={max_sim:.3f}, size={cluster_sizes[source]})"
            )
            labels[labels == source] = target

        return self._reindex_labels(labels)

    def _rebalance_labels(self, embeddings, centroids, labels):
        """
        Reassign samples to make cluster sizes near-equal while minimizing distance increase.
        Useful when user explicitly requests a fixed number of labels/classes.
        """
        n_samples = embeddings.shape[0]
        n_clusters = centroids.shape[0]
        if n_clusters <= 1 or n_samples < n_clusters:
            return labels

        target_sizes = np.full(n_clusters, n_samples // n_clusters, dtype=int)
        target_sizes[: n_samples % n_clusters] += 1
        counts = np.bincount(labels, minlength=n_clusters).astype(int)

        # Squared euclidean distances: [n_samples, n_clusters]
        distances = np.sum((embeddings[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

        def overfull_clusters():
            return [c for c in range(n_clusters) if counts[c] > target_sizes[c]]

        def underfull_clusters():
            return [c for c in range(n_clusters) if counts[c] < target_sizes[c]]

        moves = 0
        while True:
            over = overfull_clusters()
            under = underfull_clusters()
            if not over or not under:
                break

            best_move = None
            best_penalty = np.inf

            for src in over:
                sample_ids = np.where(labels == src)[0]
                if sample_ids.size == 0:
                    continue

                base = distances[sample_ids, src]
                alt = distances[np.ix_(sample_ids, under)]
                penalties = alt - base[:, None]

                local_idx = np.unravel_index(np.argmin(penalties), penalties.shape)
                candidate_penalty = float(penalties[local_idx])
                sample_id = int(sample_ids[local_idx[0]])
                dst = int(under[local_idx[1]])

                if candidate_penalty < best_penalty:
                    best_penalty = candidate_penalty
                    best_move = (sample_id, src, dst)

            if best_move is None:
                break

            sample_id, src, dst = best_move
            labels[sample_id] = dst
            counts[src] -= 1
            counts[dst] += 1
            moves += 1

        if moves > 0:
            print(f"Balanced assignment applied with {moves} sample reassignments.")
        return labels

    def cluster(self, embeddings):
        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        n_samples = embeddings_np.shape[0]

        if n_samples == 0:
            return np.array([], dtype=int)
        if n_samples == 1:
            print("Only one sample provided; creating one cluster.")
            return np.array([0], dtype=int)

        normalized_embeddings = normalize(embeddings_np).astype(np.float32)
        n_components = min(64, normalized_embeddings.shape[1], n_samples - 1)

        if n_components >= 2:
            print(f"Applying PCA to reduce dimensions to {n_components}...")
            pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")
            reduced_embeddings = pca.fit_transform(normalized_embeddings).astype(np.float32)
        else:
            reduced_embeddings = normalized_embeddings

        if self._use_minibatch_kmeans(n_samples):
            print(
                f"Large dataset detected ({n_samples} samples); "
                f"using MiniBatchKMeans with batch_size={self.kmeans_batch_size}."
            )

        optimal_k = self._select_k(reduced_embeddings, n_samples)
        kmeans = self._make_clusterer(optimal_k, n_samples, n_init=50)
        labels = kmeans.fit_predict(reduced_embeddings)
        labels = self._reindex_labels(labels)

        print(f"Clusters before refinement: {len(np.unique(labels))}")

        if self.balance_clusters and optimal_k > 1:
            labels = self._rebalance_labels(
                embeddings=reduced_embeddings,
                centroids=kmeans.cluster_centers_,
                labels=labels.copy(),
            )

        if self.refine_clusters and self.num_clusters is None:
            labels = self._refine_by_centroid_similarity(labels, normalized_embeddings)

        final_count = len(np.unique(labels))
        print(f"Final cluster count: {final_count}")
        return labels

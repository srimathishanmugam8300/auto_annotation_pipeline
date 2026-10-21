import os
import json
import csv
import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .clustering import ClusterEngine
from .feature_extraction import DINOv2Extractor
from .ingestion import load_images, read_image
from .preprocessing import get_preprocessing_transforms, preprocess_image
from .sorting import create_folders_and_sort


class UAISPipeline:
    def __init__(
        self,
        num_clusters=None,
        max_clusters=20,
        refine_clusters=False,
        merge_threshold=0.98,
        label_names=None,
        prototype_dir=None,
        clear_output=True,
        expected_per_label=None,
        review_threshold=0.03,
        embedding_views=3,
        prototype_samples_per_label=120,
        prototype_centroids_per_label=3,
        silhouette_sample_size=2000,
        minibatch_threshold=4000,
        kmeans_batch_size=2048,
    ):
        self.label_names = list(label_names) if label_names else None
        self.prototype_dir = prototype_dir
        self.clear_output = clear_output
        self.expected_per_label = expected_per_label
        self.review_threshold = float(review_threshold)
        self.embedding_views = max(1, int(embedding_views))
        self.prototype_samples_per_label = max(1, int(prototype_samples_per_label))
        self.prototype_centroids_per_label = max(1, int(prototype_centroids_per_label))
        self._rng = random.Random(42)

        if self.label_names and num_clusters is not None and num_clusters != len(self.label_names):
            print(
                f"Warning: --clusters={num_clusters} conflicts with {len(self.label_names)} label names. "
                "Using label count."
            )
            num_clusters = len(self.label_names)

        if self.label_names and num_clusters is None and not self.prototype_dir:
            num_clusters = len(self.label_names)

        self.extractor = DINOv2Extractor()
        self.cluster_engine = ClusterEngine(
            num_clusters=num_clusters,
            max_clusters=max_clusters,
            refine_clusters=refine_clusters,
            similarity_threshold=merge_threshold,
            silhouette_sample_size=silhouette_sample_size,
            minibatch_threshold=minibatch_threshold,
            kmeans_batch_size=kmeans_batch_size,
        )
        self.transform = get_preprocessing_transforms()

    def _generate_views(self, image):
        """Generate deterministic multi-view crops for more stable embeddings."""
        views = [image]
        if self.embedding_views <= 1:
            return views

        width, height = image.size
        ratios = [0.92, 0.84, 0.76]
        for ratio in ratios[: self.embedding_views - 1]:
            crop_w = max(32, int(width * ratio))
            crop_h = max(32, int(height * ratio))
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
            views.append(image.crop((left, top, right, bottom)).resize((width, height)))
        return views

    def _aggregate_view_embeddings(self, embedding_list):
        matrix = np.array(embedding_list, dtype=np.float32)
        matrix = normalize(matrix)
        merged = np.mean(matrix, axis=0)
        norm = np.linalg.norm(merged) + 1e-12
        return (merged / norm).astype(np.float32)

    def _compute_centroids(self, normalized_embeddings, labels, k):
        centroids = []
        global_center = np.mean(normalized_embeddings, axis=0)
        for cluster_id in range(k):
            points = normalized_embeddings[labels == cluster_id]
            if len(points) == 0:
                centroids.append(global_center)
            else:
                centroids.append(np.mean(points, axis=0))
        return normalize(np.array(centroids))

    def _balanced_target_counts(self, n_samples, k):
        if self.expected_per_label is not None:
            target = np.full(k, int(self.expected_per_label), dtype=int)
            if int(target.sum()) != int(n_samples):
                print(
                    f"Warning: expected_per_label={self.expected_per_label} does not match "
                    f"{n_samples} samples across {k} labels. Using nearest balanced split instead."
                )
            else:
                return target

        base = n_samples // k
        rem = n_samples - (base * k)
        target = np.full(k, base, dtype=int)
        if rem > 0:
            target[:rem] += 1
        return target

    def _rebalance_labels(self, embeddings, labels):
        inferred_k = int(np.max(labels)) + 1 if len(labels) > 0 else 0
        k = len(self.label_names) if self.label_names else inferred_k
        labels = labels.astype(int).copy()
        n_samples = len(labels)
        if n_samples == 0 or k <= 1:
            return labels

        target = self._balanced_target_counts(n_samples, k)
        counts = np.bincount(labels, minlength=k)
        if np.array_equal(counts, target):
            return labels

        normalized_embeddings = normalize(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
        centroids = self._compute_centroids(normalized_embeddings, labels, k)
        cost = 1.0 - (normalized_embeddings @ centroids.T)

        moves = 0
        max_moves = n_samples * 5
        while moves < max_moves:
            counts = np.bincount(labels, minlength=k)
            surplus = counts - target
            over = np.where(surplus > 0)[0]
            under = np.where(surplus < 0)[0]
            if len(over) == 0 or len(under) == 0:
                break

            best_move = None
            best_penalty = None

            for to_cluster in under:
                for from_cluster in over:
                    idx = np.where(labels == from_cluster)[0]
                    if len(idx) == 0:
                        continue

                    penalties = cost[idx, to_cluster] - cost[idx, from_cluster]
                    local_pos = int(np.argmin(penalties))
                    candidate_index = int(idx[local_pos])
                    penalty = float(penalties[local_pos])

                    if best_penalty is None or penalty < best_penalty:
                        best_penalty = penalty
                        best_move = (candidate_index, int(from_cluster), int(to_cluster))

            if best_move is None:
                break

            sample_idx, from_cluster, to_cluster = best_move
            labels[sample_idx] = to_cluster
            moves += 1

        final_counts = np.bincount(labels, minlength=k)
        print("Balanced reassignment counts:", final_counts.tolist())
        return labels

    def _cluster_quality(self, embeddings, labels):
        unique = np.unique(labels)
        if len(unique) < 2 or len(unique) >= len(labels):
            return None
        try:
            emb = normalize(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
            return self.cluster_engine.score_clusters(emb, labels, prefix="Quality report: ")
        except Exception:
            return None

    def _write_quality_reports(self, output_dir, image_paths, embeddings, labels):
        os.makedirs(output_dir, exist_ok=True)

        normalized_embeddings = normalize(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
        k = int(np.max(labels)) + 1
        centroids = self._compute_centroids(normalized_embeddings, labels, k)
        similarity = normalized_embeddings @ centroids.T

        top_idx = np.argsort(similarity, axis=1)[:, ::-1]
        top1_idx = top_idx[:, 0]
        top2_idx = top_idx[:, 1] if similarity.shape[1] > 1 else top_idx[:, 0]
        top1_score = similarity[np.arange(len(labels)), top1_idx]
        top2_score = similarity[np.arange(len(labels)), top2_idx]
        margins = top1_score - top2_score

        low_conf_mask = margins < self.review_threshold
        low_conf_count = int(np.sum(low_conf_mask))

        csv_path = os.path.join(output_dir, "assignment_report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "file_name",
                    "source_path",
                    "predicted_label_id",
                    "predicted_label_name",
                    "top1_similarity",
                    "top2_similarity",
                    "margin",
                    "needs_review",
                ]
            )
            for i, image_path in enumerate(image_paths):
                label_id = int(labels[i])
                label_name = (
                    self.label_names[label_id]
                    if self.label_names and 0 <= label_id < len(self.label_names)
                    else str(label_id + 1)
                )
                writer.writerow(
                    [
                        os.path.basename(image_path),
                        image_path,
                        label_id,
                        label_name,
                        float(top1_score[i]),
                        float(top2_score[i]),
                        float(margins[i]),
                        bool(low_conf_mask[i]),
                    ]
                )

        counts = np.bincount(labels, minlength=k)
        count_map = {}
        for cluster_id, count in enumerate(counts.tolist()):
            name = (
                self.label_names[cluster_id]
                if self.label_names and 0 <= cluster_id < len(self.label_names)
                else str(cluster_id + 1)
            )
            count_map[name] = int(count)

        quality = {
            "num_images": int(len(labels)),
            "num_clusters": int(k),
            "silhouette_score": self._cluster_quality(embeddings, labels),
            "review_threshold": float(self.review_threshold),
            "needs_review_count": low_conf_count,
            "needs_review_ratio": float(low_conf_count / max(1, len(labels))),
            "cluster_counts": count_map,
        }

        json_path = os.path.join(output_dir, "quality_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2)

        print(f"Wrote quality report: {json_path}")
        print(f"Wrote assignment report: {csv_path}")
        if quality["needs_review_ratio"] > 0.25:
            print(
                "Warning: More than 25% samples are low-confidence; "
                "prototype images or stronger supervision is recommended."
            )

    def _extract_embedding_from_path(self, path):
        img = read_image(path)
        if img is None:
            return None
        views = self._generate_views(img)
        view_embeddings = []
        for view in views:
            preprocessed = preprocess_image(view, self.transform)
            emb = self.extractor.extract_features(preprocessed)
            view_embeddings.append(emb)
        return self._aggregate_view_embeddings(view_embeddings)

    def _assign_with_prototypes(self, embeddings):
        if not self.prototype_dir:
            return None

        if not os.path.isdir(self.prototype_dir):
            print(f"Prototype directory not found: {self.prototype_dir}. Falling back to clustering.")
            return None

        if self.label_names:
            candidate_labels = self.label_names
        else:
            candidate_labels = sorted(
                d for d in os.listdir(self.prototype_dir)
                if os.path.isdir(os.path.join(self.prototype_dir, d))
            )

        prototype_vectors_by_label = []
        valid_labels = []

        for label_name in candidate_labels:
            label_dir = os.path.join(self.prototype_dir, label_name)
            if not os.path.isdir(label_dir):
                print(f"Skipping missing prototype label folder: {label_dir}")
                continue

            prototype_paths = load_images(label_dir)
            if len(prototype_paths) > self.prototype_samples_per_label:
                prototype_paths = self._rng.sample(prototype_paths, self.prototype_samples_per_label)
            label_embeddings = []
            for path in prototype_paths:
                emb = self._extract_embedding_from_path(path)
                if emb is not None:
                    label_embeddings.append(emb)

            if not label_embeddings:
                print(f"Skipping label '{label_name}' (no valid prototype images).")
                continue

            label_embeddings = normalize(np.asarray(label_embeddings, dtype=np.float32)).astype(np.float32)
            k_local = min(self.prototype_centroids_per_label, len(label_embeddings))
            if k_local == 1:
                sub_prototypes = normalize(np.mean(label_embeddings, axis=0, keepdims=True))
            else:
                kmeans = KMeans(n_clusters=k_local, random_state=42, n_init=20)
                kmeans.fit(label_embeddings)
                sub_prototypes = normalize(kmeans.cluster_centers_)

            prototype_vectors_by_label.append(sub_prototypes)
            valid_labels.append(label_name)

        if not prototype_vectors_by_label:
            print("No valid prototypes found. Falling back to clustering.")
            return None

        embedding_matrix = normalize(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
        label_scores = []
        for sub_prototypes in prototype_vectors_by_label:
            sim = embedding_matrix @ sub_prototypes.T
            # Robust score for each label = best matching sub-prototype.
            label_scores.append(np.max(sim, axis=1))
        label_scores = np.stack(label_scores, axis=1)
        labels = np.argmax(label_scores, axis=1).astype(int)

        self.label_names = valid_labels
        self.cluster_engine.num_clusters = len(valid_labels)
        print(
            f"Assigned images using {len(valid_labels)} prototype labels "
            f"({self.prototype_centroids_per_label} sub-prototypes each, "
            f"max {self.prototype_samples_per_label} samples/label)."
        )
        return labels

    def _stabilize_cluster_to_label_order(self, labels, embeddings):
        if not self.label_names:
            return labels

        unique_labels = sorted(np.unique(labels).tolist())
        if len(unique_labels) != len(self.label_names):
            print(
                "Cluster count does not match label count; mapping available clusters to labels in index order."
            )
            remap = {old: idx for idx, old in enumerate(unique_labels)}
            return np.array([remap[lbl] for lbl in labels], dtype=int)

        normalized = normalize(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
        centroids = []
        for lbl in unique_labels:
            pts = normalized[labels == lbl]
            centroids.append(np.mean(pts, axis=0))
        centroids = np.array(centroids)

        # Deterministic ordering makes runs reproducible when mapping cluster ids to label names.
        projection = PCA(n_components=1, random_state=42).fit_transform(centroids).reshape(-1)
        sorted_pairs = sorted(zip(projection, unique_labels), key=lambda x: x[0])
        remap = {old_label: new_idx for new_idx, (_, old_label) in enumerate(sorted_pairs)}
        return np.array([remap[lbl] for lbl in labels], dtype=int)

    def run(self, input_dir, output_dir):
        print(f"Starting UAISP Pipeline on {input_dir}...")

        image_paths = load_images(input_dir)
        if not image_paths:
            print("No images found in the input directory.")
            return

        print(f"Found {len(image_paths)} images.")

        embeddings = []
        valid_image_paths = []

        print("Processing images (Resize/Normalize + Feature Extraction)...")
        for path in tqdm(image_paths):
            embedding = self._extract_embedding_from_path(path)
            if embedding is None:
                continue
            embeddings.append(embedding)
            valid_image_paths.append(path)

        if not embeddings:
            print("No valid embeddings extracted.")
            return

        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        labels = self._assign_with_prototypes(embeddings_np)
        if labels is None:
            if self.label_names and not self.prototype_dir:
                print(
                    "Label names were provided without prototypes. "
                    "Using unsupervised clustering and mapping clusters to labels deterministically."
                )
            print("Clustering embeddings...")
            labels = self.cluster_engine.cluster(embeddings_np)
            labels = self._stabilize_cluster_to_label_order(labels, embeddings_np)

        # Optional balancing for known setups (e.g., equal expected count per label).
        if self.label_names or self.expected_per_label is not None:
            labels = self._rebalance_labels(embeddings_np, labels)

        print("Sorting images into folders...")
        create_folders_and_sort(
            valid_image_paths,
            labels,
            output_dir,
            label_names=self.label_names,
            clear_output=self.clear_output,
        )

        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster distribution:")
        for lbl, count in zip(unique, counts):
            if self.label_names and 0 <= int(lbl) < len(self.label_names):
                label_text = self.label_names[int(lbl)]
            else:
                label_text = str(int(lbl) + 1)
            print(f"  {label_text}: {int(count)} images")

        self._write_quality_reports(output_dir, valid_image_paths, embeddings_np, labels)

        print("Pipeline execution completed successfully!")

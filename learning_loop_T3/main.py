import argparse
import os

from src.pipeline import UAISPipeline


def _parse_label_names(args):
    labels = []

    if args.labels:
        labels.extend([part.strip() for part in args.labels.split(",") if part.strip()])

    if args.labels_file:
        if not os.path.exists(args.labels_file):
            raise FileNotFoundError(f"Labels file not found: {args.labels_file}")
        with open(args.labels_file, "r", encoding="utf-8") as f:
            for line in f:
                label = line.strip()
                if label:
                    labels.append(label)

    # Keep order, remove duplicates.
    deduped = list(dict.fromkeys(labels))
    return deduped or None

def main():
    parser = argparse.ArgumentParser(description="Universal Automated Image Segregation Pipeline (UAISP)")
    parser.add_argument("input", type=str, help="Path to the input folder containing images")
    parser.add_argument("output", type=str, help="Path to the output folder for sorted images")
    parser.add_argument(
        "-c",
        "--clusters",
        type=int,
        default=None,
        help="Number of clusters to force. If labels are provided, this is ignored and label count is used.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated label names, e.g. 'no dementia,very mild,mild,moderate'.",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default=None,
        help="Path to a text file with one label name per line.",
    )
    parser.add_argument(
        "--prototype-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of label prototypes. Use one subfolder per label containing example images. "
            "If set, assignment is nearest-prototype instead of pure clustering."
        ),
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=20,
        help="Maximum clusters to consider during auto-k silhouette search (default: 20).",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.98,
        help="Cosine similarity threshold for optional tiny-cluster refinement (default: 0.98).",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Enable conservative tiny-cluster refinement after KMeans (disabled by default).",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing output folder before writing new results.",
    )
    parser.add_argument(
        "--expected-per-label",
        type=int,
        default=None,
        help=(
            "Optional expected images per label for balanced reassignment. "
            "Useful for controlled tests (e.g., 30 images per label)."
        ),
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.03,
        help=(
            "Margin threshold used to flag low-confidence assignments in quality reports "
            "(default: 0.03)."
        ),
    )
    parser.add_argument(
        "--embedding-views",
        type=int,
        default=3,
        help=(
            "Number of deterministic views per image for embedding averaging "
            "(default: 3, stronger but slower)."
        ),
    )
    parser.add_argument(
        "--prototype-samples-per-label",
        type=int,
        default=120,
        help=(
            "Max prototype images sampled per label from --prototype-dir "
            "(default: 120)."
        ),
    )
    parser.add_argument(
        "--prototype-centroids-per-label",
        type=int,
        default=3,
        help=(
            "Number of sub-prototypes (centroids) per label when using --prototype-dir "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=2000,
        help=(
            "Maximum sample count used for silhouette scoring during auto-k search and "
            "quality reporting. Smaller datasets still use the full set. Set <= 0 to disable sampling."
        ),
    )
    parser.add_argument(
        "--minibatch-threshold",
        type=int,
        default=4000,
        help=(
            "Switch to MiniBatchKMeans when the dataset reaches this many images "
            "(default: 4000)."
        ),
    )
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=2048,
        help="MiniBatchKMeans batch size used for large datasets (default: 2048).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    try:
        label_names = _parse_label_names(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    pipeline = UAISPipeline(
        num_clusters=args.clusters,
        max_clusters=args.max_clusters,
        refine_clusters=args.refine,
        merge_threshold=args.merge_threshold,
        label_names=label_names,
        prototype_dir=args.prototype_dir,
        clear_output=args.clear_output,
        expected_per_label=args.expected_per_label,
        review_threshold=args.review_threshold,
        embedding_views=args.embedding_views,
        prototype_samples_per_label=args.prototype_samples_per_label,
        prototype_centroids_per_label=args.prototype_centroids_per_label,
        silhouette_sample_size=args.silhouette_sample_size,
        minibatch_threshold=args.minibatch_threshold,
        kmeans_batch_size=args.kmeans_batch_size,
    )
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()

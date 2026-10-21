# Universal Automated Image Segregation Pipeline (UAISP)

A clustering and label-assignment pipeline for datasets where CLIP often struggles
(medical, sign language, industrial defects, microscopy, etc.).

This pipeline relies on:
* Deep visual features (DINOv2)
* KMeans + silhouette auto-k (or forced k)
* Optional prototype-guided assignment (few-shot examples per label)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Keep DINOv2 checkpoint in project root:
`dinov2_vitb14_pretrain.pth`

## Usage

To run with automatic cluster detection:
```bash
python main.py path/to/input_folder path/to/output_folder
```

To force a fixed number of clusters:
```bash
python main.py path/to/input_folder path/to/output_folder -c 4
```

To force labels by name (folder names will match label names):
```bash
python main.py path/to/input_folder path/to/output_folder --labels "no dementia,very mild,mild,moderate"
```

To provide labels from a file (one label per line):
```bash
python main.py path/to/input_folder path/to/output_folder --labels-file labels.txt
```

To run prototype-guided assignment (recommended for medical/sign domains):
```bash
python main.py path/to/input_folder path/to/output_folder --labels "no dementia,very mild,mild,moderate" --prototype-dir path/to/prototypes
```

Important:
* `--labels` without `--prototype-dir` only names clusters; it does not guarantee semantic stage mapping.
* For class names like dementia stages, always provide prototype images per label.

Prototype directory format:
```text
prototypes/
  no dementia/
    ex1.jpg
    ex2.jpg
  very mild/
    ex1.jpg
  mild/
    ex1.jpg
  moderate/
    ex1.jpg
```

Extra options:
```bash
--max-clusters 20
--refine
--merge-threshold 0.98
--clear-output
--expected-per-label 30
--review-threshold 0.03
--embedding-views 3
--prototype-samples-per-label 120
--prototype-centroids-per-label 3
--silhouette-sample-size 2000
--minibatch-threshold 4000
--kmeans-batch-size 2048
```

For balanced testing datasets (example: 10 labels, exactly 30 images each):
```bash
python main.py sign_lang2 sign_output2 --labels "0,1,2,3,4,5,6,7,8,9" --expected-per-label 30 --clear-output
```

For stronger accuracy on larger datasets, use prototype supervision from the original labeled set:
```bash
python main.py sign_lang2 sign_output2 --labels "0,1,2,3,4,5,6,7,8,9" --prototype-dir sign --expected-per-label 30 --embedding-views 3 --prototype-samples-per-label 120 --prototype-centroids-per-label 3 --clear-output
```

Large-dataset behavior:
* Silhouette scoring is automatically sampled above `--silhouette-sample-size`, which avoids the full quadratic memory/time cost on very large image sets.
* Clustering automatically switches to `MiniBatchKMeans` above `--minibatch-threshold`.
* If you already know the label count, prefer `--labels` or `--clusters` so the pipeline does not spend extra time searching for `k`.

After each run, these files are created in output folder:
* `quality_report.json` (overall quality + cluster counts + low-confidence ratio)
* `assignment_report.csv` (per-image assignment + confidence margin + needs_review flag)

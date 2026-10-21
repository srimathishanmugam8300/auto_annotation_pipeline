# Project Architecture

## Overview

The **Auto_annotation** ecosystem is a multi-modal AI platform engineered for automated dataset labeling, feature extraction, cluster-based active learning, and human-in-the-loop validation. It handles vision (images and video streams), audio signals, and specialized downstream classification tasks.

## Tech Stack

- **Languages**: Python 3.10+, JavaScript (ES6+), HTML5, CSS3
- **Deep Learning & Computer Vision**: PyTorch, Keras, TensorFlow, OpenCV, torchvision, ONNX Runtime
- **Foundation Models**:
  - **DINOv2** (`dinov2_vitb14_pretrain.pth`): Self-supervised visual feature representation
  - **Segment Anything Model (SAM)** (`sam_vit_b_01ec64.pth`): Zero-shot instance segmentation
  - **OpenCLIP / ResNet**: Dense visual embedding generation
- **Audio Processing**: Librosa, SciPy, SoundFile, MFCC & Mel-Spectrogram Extractors
- **Machine Learning & Clustering**: HDBSCAN, Scikit-learn (DecisionTree, RandomForest, KMeans, PCA, UMAP)
- **LLM Integration**: OpenAI GPT-4o API, Google Gemini API for automated zero-shot batch labeling and rule synthesis
- **Backend Frameworks**: FastAPI, Uvicorn, Flask / Python Standard HTTP Server
- **Frontend Frameworks**: Vite, Vanilla JavaScript, CSS3 Glassmorphism UI
- **Dataset Formats**: YOLO (`.txt`), COCO (`.json`), Pascal VOC (`.xml`), Pickled Feature Stores (`.pkl`)

## System Components

### 1. Vision Auto-Annotation Suite (`autonew`, `auto_annotation_uni`, `annotation_auto`)
- **Embedding Pipeline**: Computes dense high-dimensional vectors for images via DINOv2 / CLIP / ResNet.
- **Clustering Engine**: Runs HDBSCAN on feature embeddings to detect natural semantic groupings without pre-defining class counts.
- **Representative Sampling**: Selects centroid and mid-region samples for efficient human or LLM inspection.
- **LLM Annotator**: Sends representative image batches/prompts to LLM APIs for automated zero-shot label assignment.
- **Rule Engine**: Fits interpretable Decision Trees to derive clear feature decision rules and propagates labels to remaining cluster samples via similarity scoring.

### 2. Foundation Model & Segmentation Hub (`learning_loop_T3`)
- Leverages local pre-trained DINOv2 ViT-B/14 and SAM ViT-B weights.
- Generates precise pixel-level masks and fine-grained feature representations for complex domains such as Sign Language gesture recognition.

### 3. Active Learning & Interactive Web UI (`learning_loop_T1`, `learning_loop_T2`)
- **Tier 1 (`learning_loop_T1`)**: Interactive web portal (`server.py`, `start_web_ui.bat`) enabling human annotators to upload datasets, trigger auto-pipelines, review candidate labels, and commit corrections.
- **Tier 2 (`learning_loop_T2`)**: High-efficiency active refinement engine supporting dual feature extractors (standard PyTorch & lightweight ONNX), semantic matching, and visualization of decision boundaries.

### 4. Audio Active Learning Loop (`Audio_learning_loop`)
- Specialized audio auto-labeling pipeline (`auto_pipeline.py`).
- Transforms raw audio into spectral features (MFCCs, Mel-Spectrograms, Chroma).
- Clusters audio embeddings, selects query samples, tags via LLM/transcription rules, and trains downstream audio classifiers iteratively.

### 5. Video Annotation & Tracking (`learning_loop_video`)
- FastAPI web service (`app/main.py`) designed for video stream ingestion.
- Frame extraction, object detection, multi-object tracking across frames, and temporal label smoothing.

### 6. Eye Open/Close Detection Model (`Data_automation`)
- Custom Keras CNN architecture for driver monitoring / eye state classification.
- Complete data preprocessing, sizing, auto-labeling, and validation utility scripts.

### 7. Showcase & Web Dashboard (`website`)
- Vite-based web client presenting project performance metrics, event data logs, and visual interactive guides.

## Folder Structure

```
Auto_annotation/
├── current.md                          # Live project state documentation
├── architecture.md                     # System architecture & design
├── plan.md                             # Execution plan & milestones
├── rules.md                            # Documentation governance rules
├── annotation_auto/                    # Vector embedding & indexing pipeline
├── auto_annotation_uni/                # Universal image auto-annotation sub-engine
├── autonew/                            # Core image auto-labeling & rule derivation framework
│   ├── configs/                        # System configurations
│   ├── prompts/                        # LLM prompts for batch labeling
│   ├── src/                            # Embeddings, clustering, rules, export modules
│   ├── main.py                         # Entrypoint script
│   └── project_summary.ipynb           # Notebook analytics
├── Audio_learning_loop/                # Audio active learning & annotation pipeline
│   ├── auto_pipeline.py                # Main audio processing script
│   └── input/ / output/                # Audio dataset directories
├── Automation2/                        # Config-driven automated labeling engine
├── Data_automation/                    # Eye state classification CNN & data scripts
│   ├── eye_open_close_model.keras      # Trained model checkpoint
│   └── train_model.py / auto_label.py  # Model training & inference
├── learning_loop_T1/                   # Web UI & active learning server (Tier 1)
│   ├── server.py                       # Python Web UI server
│   ├── auto_pipeline.py                # Auto-annotation pipeline script
│   └── start_web_ui.bat                # Web UI launcher
├── learning_loop_T2/                   # Refinement engine & semantic matching (Tier 2)
│   ├── dataset_organizer.py            # Dataset reorganization tool
│   └── feature_extractor_lightweight.py# Lightweight ONNX feature extractor
├── learning_loop_T3/                   # Foundation models (DINOv2 + SAM) engine
│   ├── dinov2_vitb14_pretrain.pth      # DINOv2 model weights
│   └── sam_vit_b_01ec64.pth            # SAM model weights
├── learning_loop_video/                # FastAPI video annotation & tracking service
│   └── app/                            # API endpoints, services, schemas
└── website/                            # Vite frontend project presentation website
```

## Data Flow

1. **Ingestion**: Raw multi-modal data (images, videos, audio clips) uploaded via Web UI or filesystem.
2. **Embedding Extraction**: Deep feature vectors extracted using CLIP, DINOv2, ONNX, or Librosa.
3. **Clustering & Sampling**: Unsupervised HDBSCAN partitions embeddings into semantic clusters; centroid/mid-region representative samples are selected.
4. **Auto-Labeling / Prompting**: Representative samples are annotated automatically via zero-shot LLM prompts or domain CNNs.
5. **Rule Derivation & Propagation**: Interpretable decision rules are derived and propagated to all samples in the cluster with similarity confidence scoring.
6. **Human Verification & Active Refinement**: Annotators inspect low-confidence samples in the Web UI, correct errors, and commit refined labels.
7. **Export & Model Training**: Finalized dataset exported to standard formats (YOLO, COCO) or used directly to retrain downstream classification models.

## Architecture Decisions

- **Decision 1**: Dual LLM and Rule-based Auto-Labeling Architecture
  - *Reason*: Avoids sending every individual data point to expensive cloud LLM APIs. Only representative cluster samples are sent to LLMs; remaining samples are auto-annotated via derived decision rules and embedding similarity matching.
- **Decision 2**: Integration of Local Foundation Models (DINOv2 & SAM)
  - *Reason*: Provides state-of-the-art zero-shot visual features and pixel-precise object segmentation locally without external network dependencies.
- **Decision 3**: Multi-Tiered Modular Structure
  - *Reason*: Separates light web-based workflows (Tier 1), heavy feature refinement engines (Tier 2), foundation model inference (Tier 3), and distinct modalities (Audio, Video) to maximize code maintainability and scalability.

# Current Project State

## Project Overview

Auto_annotation is an end-to-end, multi-modal automated data annotation, labeling, and active learning ecosystem. The codebase consists of 11 top-level subfolders, of which **10 are fully working functional engines/applications** and **1 is an empty skeleton template**.

## Folder Analysis & Working Status Breakdown

| Folder Name | Modality / Purpose | Working Status | Key Capabilities & Highlights |
| :--- | :--- | :---: | :--- |
| **`learning_loop_T1`** | Vision / Interactive Web UI | 🟢 **Working** | Web UI dashboard (`server.py`, `start_web_ui.bat`), drag-and-drop batch uploader, human-in-the-loop review, and complete `auto_pipeline.py`. |
| **`learning_loop_T2`** | Vision / Refinement Engine | 🟢 **Working** | Active refinement (`refinement.py`), dual feature extractors (standard PyTorch & ONNX `feature_extractor_lightweight.py`), semantic matching, and visualization tools. |
| **`learning_loop_T3`** | Vision / Foundation Models | 🟢 **Working** | Integrated Meta **DINOv2** (`dinov2_vitb14_pretrain.pth` ~346MB) and **SAM** (`sam_vit_b_01ec64.pth` ~375MB) for zero-shot feature extraction and segmentation masks. |
| **`learning_loop_video`** | Video / Object Tracking | 🟢 **Working** | FastAPI web application (`app/main.py`), keyframe extraction, object tracking, and temporal label smoothing. |
| **`Audio_learning_loop`** | Audio / Active Learning | 🟢 **Working** | End-to-end audio pipeline (`auto_pipeline.py`), MFCC/Mel-spectrogram/Chroma feature extraction, clustering, confidence scoring, and iterative model retraining. |
| **`autonew`** | Vision / LLM & Rule Derivation | 🟢 **Working** | Full production pipeline: CLIP/DINOv2 embeddings, HDBSCAN clustering, mid-region sampling, GPT-4o/Gemini batch auto-labeling, decision tree rule derivation, and COCO/YOLO dataset export. |
| **`Data_automation`** | Vision / Eye State Classification | 🟢 **Working** | Keras CNN model (`eye_open_close_model.keras` & `.h5`), training scripts (`train_model.py`), single-image inference (`predict_one.py`), auto-labeling, and resizing utilities. |
| **`auto_annotation_uni`** | Vision / Modular Pipeline | 🟢 **Working** | Standalone modular image auto-labeling engine (`src/` with `embed_images.py`, `clustering_hdbscan.py`, `llm_annotate_batch.py`, `derive_learning_rule.py`). |
| **`annotation_auto`** | Vision / Vector Indexing | 🟢 **Working** | Vector indexing and embedding pipeline (`pipeline.py`, pre-computed `embeddings/`, FAISS `indexes/`, and stored `results.pkl`). |
| **`website`** | Frontend Presentation | 🟢 **Working** | Vite web interface (`index.html`, `script.js`, `styles.css`, `events-data.js`) for project presentation and event logging. |
| **`Automation2`** | Framework Skeleton | 🔴 **Empty / Skeleton** | Contains folder structure (`src/`, `configs/`, `prompts/`), but files (`main.py`, scripts) are 0-byte placeholders. |

## Recent Changes Log

- **[2026-10-05]** – Completed comprehensive folder inventory and working status analysis across all 11 subdirectories.
- **[2026-10-05]** – Created official system architecture, project state, execution plan, and governance documentation (`current.md`, `architecture.md`, `plan.md`, `rules.md`).
- **[2026-10-04]** – Integrated DINOv2 ViT-B/14 and SAM ViT-B foundation model checkpoints into `learning_loop_T3`.
- **[2026-10-02]** – Integrated FastAPI video annotation service in `learning_loop_video`.
- **[2026-09-28]** – Implemented Web UI active learning server and interface in `learning_loop_T1`.
- **[2026-09-20]** – Built `Audio_learning_loop` automated feature extraction and active learning pipeline.
- **[2026-09-10]** – Completed `autonew` LLM-driven rule derivation and YOLO/COCO export tools.

## Known Issues / Blockers

- Large model weights in `learning_loop_T3` (`sam_vit_b_01ec64.pth` and `dinov2_vitb14_pretrain.pth`) require GPU acceleration for real-time video batch processing.

## Next Immediate Actions

- Consolidate working modules (`learning_loop_T1`, `learning_loop_video`, `Audio_learning_loop`, `autonew`) under a unified API gateway.
- Clean up or populate the skeleton boilerplate in `Automation2`.

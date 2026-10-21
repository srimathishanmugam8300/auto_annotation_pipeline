# 🎨 Visual Clustering Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Unlabeled Images                  │
│                                                              │
│  🖼️  🖼️  🖼️  🖼️  🖼️  🖼️  🖼️  🖼️  🖼️  🖼️                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 STEP 1: Visual Feature Extraction            │
│                                                              │
│  Each image → Vector (512-768 dimensions)                   │
│                                                              │
│  Using: DINOv2 (deep) or Classical CV (fast)               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  STEP 2: Clustering by Similarity            │
│                                                              │
│  K-Means: Group similar vectors together                    │
│                                                              │
│  Result: K clusters of visually similar images              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                STEP 3: Learning Loop Refinement              │
│                                                              │
│  Iteration 1: Reassign borderline images                    │
│  Iteration 2: Update cluster prototypes                     │
│  Iteration 3: Converge to stable groupings                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 STEP 4: Class Name Assignment                │
│                                                              │
│  Mode 1: User provides names → "two", "three", "four"       │
│  Mode 2: Auto-generate → "class_1", "class_2", "class_3"   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  STEP 5: Copy to Class Folders               │
│                                                              │
│  Create folder for each class                               │
│  Copy images to respective folders                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT: Organized Classes                 │
│                                                              │
│  output/                                                     │
│    ├── two/    🖼️  🖼️  🖼️                                  │
│    ├── three/  🖼️  🖼️  🖼️                                  │
│    ├── four/   🖼️  🖼️  🖼️                                  │
│    ├── six/    🖼️  🖼️  🖼️                                  │
│    └── seven/  🖼️  🖼️  🖼️                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 The Key Difference

### ❌ Text-Based Approach (CLIP)
```
Image → "What is this called?" → Text Match → Label
         ↑
    Requires language understanding
    Fails for non-semantic domains
```

### ✅ Visual Clustering Approach
```
Image → "What does this look like?" → Visual Similarity → Group
         ↑
    Pure appearance
    Works for everything
```

---

## 🎯 Two Modes

### Mode 1: With Names
```
Command: python auto_pipeline.py data output A B C D

Input:          Processing:         Output:
🖼️  🖼️          Cluster 0 → A       A/
🖼️  🖼️    →     Cluster 1 → B  →    B/
🖼️  🖼️          Cluster 2 → C       C/
🖼️  🖼️          Cluster 3 → D       D/
```

### Mode 2: With Number
```
Command: python auto_pipeline.py data output 4

Input:          Processing:         Output:
🖼️  🖼️          Cluster 0 → 1       class_1/
🖼️  🖼️    →     Cluster 1 → 2  →    class_2/
🖼️  🖼️          Cluster 2 → 3       class_3/
🖼️  🖼️          Cluster 3 → 4       class_4/
```

---

## 🧠 Learning Loop Detail

```
Initial Clustering
    ↓
┌─────────────────────────┐
│  Compute Prototypes     │  ← Average position of each cluster
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Reassign Images        │  ← Move to nearest prototype
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Update Prototypes      │  ← Recalculate with momentum
└─────────────────────────┘
    ↓
    Repeat until stable (< 1% change)
    ↓
Refined Clusters
```

---

## 📊 Data Flow

```
                  ┌─────────────────┐
                  │  Unlabeled      │
                  │  Images         │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Feature        │
                  │  Extractor      │
                  └────────┬────────┘
                           │
                     [Embeddings]
                           │
                  ┌────────▼────────┐
                  │  Clusterer      │
                  └────────┬────────┘
                           │
                     [Cluster IDs]
                           │
                  ┌────────▼────────┐
                  │  Refinement     │
                  │  Loop           │
                  └────────┬────────┘
                           │
                  [Refined Cluster IDs]
                           │
                  ┌────────▼────────┐
                  │  Name Mapper    │
                  └────────┬────────┘
                           │
                    [Class Names]
                           │
                  ┌────────▼────────┐
                  │  File Organizer │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Class Folders  │
                  └─────────────────┘
```

---

## 🎨 Quality Improvement

```
Before Refinement:          After Refinement:

Class A:  ●●●○●●            Class A:  ●●●●●●
Class B:  ●○●●●●            Class B:  ●●●●●●
Class C:  ○●●●●●            Class C:  ●●●●●●

● = Correctly clustered
○ = Misplaced (gets moved during refinement)
```

---

## 🚀 Performance

```
Speed Comparison:

Lightweight CV:     ████░░░░░░ (Fast)
DINOv2 (CPU):       ████████░░ (Moderate)
DINOv2 (GPU):       ██████████ (Very Fast)

Quality Comparison:

Lightweight CV:     ██████░░░░ (Good)
DINOv2:             ██████████ (Excellent)
```

---

## 💡 When to Use Each Mode

### Mode 1 (Named Classes)
```
✅ You know the class names
✅ You want meaningful folder names
✅ Examples: sign language, defect types, organs

Command:
python auto_pipeline.py data output heart lung kidney liver
```

### Mode 2 (Numbered Classes)
```
✅ You don't know what classes exist
✅ You want to discover structure
✅ You'll label manually later

Command:
python auto_pipeline.py data output 5
```

---

## 🔧 Under the Hood

```python
# Simplified pseudocode

# 1. Extract features
for image in images:
    features[i] = extract_visual_features(image)

# 2. Cluster
clusters = kmeans(features, n_clusters=K)

# 3. Refine
for iteration in range(max_iterations):
    prototypes = compute_centroids(features, clusters)
    clusters = assign_to_nearest(features, prototypes)
    if converged:
        break

# 4. Organize
for image, cluster_id in zip(images, clusters):
    class_name = cluster_to_name[cluster_id]
    copy(image, f"output/{class_name}/")
```

---

## 🎯 Success Metrics

Pipeline saves quality metrics:

```json
{
  "silhouette_score": 0.73,      // Higher = better separation
  "davies_bouldin": 0.52,        // Lower = tighter clusters
  "calinski_harabasz": 487.3,    // Higher = better defined
  "n_clusters": 7,
  "n_iterations": 3
}
```

---

**Visual clustering: Simple, universal, effective.**

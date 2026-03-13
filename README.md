# Temporal Tree-sliced Wasserstein (TTSW) Distance for Time Series

This project implements **TTSWD**, a Tree-Wasserstein distance method for time series, supporting classification, clustering, and visualization. The output is a pairwise distance matrix that can be used directly with k-NN, k-Medoids, and t-SNE.

---

## Algorithm

TTSWD computes optimal-transport distances between time series through four steps:

1. **Embed series as point clouds** — each series is represented as a set of points in a joint feature–time space. A normalized time index is prepended to each point, allowing the partitioning tree to split along both the time and feature dimensions simultaneously.

2. **Build a random FRT forest** — each tree is a 2-Hierarchically Separated Tree (2-HST) constructed using the Fakcharoenphol–Rao–Talwar 2004 algorithm. Ball-partition with a random center and level weights that scale as powers of two guarantee the FRT embedding property. Using a forest of multiple trees reduces variance.

3. **Compute sparse Tree-Wasserstein distances** — for each tree, the distance between two series is:

   ```
   TW(A, B) = Σ_v  weight(v) · |m_A(v) − m_B(v)|
   ```

   where `m_A(v)` is the mass of series A at node v. A sparse representation (storing only nodes with non-zero mass) combined with merge-sort makes each pair computable in O(k_A + k_B).

4. **Aggregate across the forest** — the final distance is the average over all trees in the forest, yielding a stable estimate of the Wasserstein distance.

---

## Directory Structure

```
TTSW-main/
├── requirements.txt
├── data/
│   └── UCR/                        # UCR Time Series datasets (.ts)
│       ├── ArrowHead/
│       └── ...
├── experiment/
│   ├── knn.ipynb                   # k-NN classification
│   ├── kmedoid.ipynb               # k-Medoids clustering
│   └── tsne.ipynb                  # t-SNE visualization
├── result/                         # Experiment results (auto-created)
└── src/
    ├── _common/
    │   ├── _metric_base.py         # Pairwise distance computation, streaming cache
    │   └── _time_utils.py          # Time index injection, series stacking
    ├── ttswd/
    │   ├── __init__.py
    │   ├── _tree.py                # TTSWDTree — 2-HST tree construction
    │   ├── _forest.py              # build_global_ttswd_forest
    │   ├── _metric.py              # Re-export from _common._metric_base
    │   ├── _time_utils.py          # Re-export from _common._time_utils
    │   └── _wasserstein.py         # Sparse TW distance computation
    ├── experiments/
    │   ├── io.py                   # load_ucr_dataset_tsl, read/write result tables
    │   ├── metrics.py              # MAP, clustering accuracy, embedding quality
    │   ├── knn.py                  # run_knn
    │   ├── kmedoid.py              # run_kmedoid
    │   └── tsne.py                 # run_tsne
    └── utilities.py                # Re-export shim (from experiments/)
```

---

## Dependencies

| Package | Role |
|---------|------|
| numpy >= 1.26 | Numerical computation |
| scipy | Scientific computation |
| scikit-learn | k-NN, t-SNE, k-Means, metrics |
| scikit-learn-extra | k-Medoids |
| pandas | Read/write result tables (Excel / CSV) |
| joblib | Parallel tree building |
| tqdm | Progress bars |
| tslearn | Load UCR datasets |
| numba | JIT for mass accumulation loops (~10–50× faster, optional) |
| matplotlib | t-SNE scatter plots |
| openpyxl | Excel export (falls back to CSV if missing) |

---

## Installation

```bash
pip install -r requirements.txt
```

The notebooks automatically add `src/` to `sys.path`.

---

## Data

### Directory format

```
data/UCR/<DatasetName>/<DatasetName>_TRAIN.ts
data/UCR/<DatasetName>/<DatasetName>_TEST.ts
```

### Datasets used in experiments (UCR)

| Abbreviation | Dataset |
|--------------|---------|
| AH | ArrowHead |
| BM | BasicMotions |
| BF | BeetleFly |
| CBF | CBF |
| CT | Chinatown |
| CET | CinCECGTorso |
| DSR | DiatomSizeReduction |
| GPA | GunPointAgeSpan |
| GPM | GunPointMaleVersusFemale |
| GPO | GunPointOldVersusYoung |
| Ham | Ham |
| IERT | InsectEPGRegularTrain |
| IPD | ItalyPowerDemand |
| Meat | Meat |
| MP | MelbournePedestrian |
| MS2T | MixedShapesSmallTrain |
| MS | MoteStrain |
| O2 | OliveOil |
| Plane | Plane |
| S2 | SmoothSubspace |

---

## Quick Start

### Running from a notebook

Open a notebook in `experiment/`, configure the first cell (`alg_kwargs`, `num_runs`), then run all cells. Results are written to `result/`.

### Running from Python

```python
import sys
sys.path.insert(0, "src")
from utilities import run_knn, run_kmedoid, run_tsne

dataset  = "BeetleFly"
datatype = "UCR_TSL"

# k-NN classification  → result/TTSWD_knn.xlsx
run_knn(dataset, datatype, "TTSWD", num_neighbor_list=[1], num_runs=5)

# k-Medoids clustering → result/TTSWD_kmedoid.xlsx
run_kmedoid(dataset, datatype, "TTSWD", n_clusters=None, num_runs=5)

# t-SNE visualization  → result/TTSWD_tsne.xlsx + result/<dataset>/TTSWD_tsne_seed0.png
run_tsne(dataset, datatype, "TTSWD", num_runs=5, n_components=2)
```

Pass a list to run multiple datasets sequentially:

```python
run_knn(["ArrowHead", "BeetleFly", "CBF"], datatype, "TTSWD", num_runs=3)
```

### Low-level API

```python
import sys, numpy as np
sys.path.insert(0, "src")
from ttswd import (
    stack_union_points_with_time,
    build_global_ttswd_forest,
    compute_distance_matrices_ttswd,
)

X_train = [np.random.randn(50, 3) for _ in range(100)]
X_test  = [np.random.randn(50, 3) for _ in range(20)]

# Stack all points and inject time index
coords, _, boundaries = stack_union_points_with_time(X_train, X_test)

# Build the forest
trees, meta = build_global_ttswd_forest(
    coords,
    n_trees=64,
    time_weight=64.0,
    random_state=0,
)

# Compute distance matrices
D_train, D_test = compute_distance_matrices_ttswd(
    trees, boundaries, n_tr=100, n_te=20
)
# D_train: (100, 100) symmetric; D_test: (20, 100)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_trees` | 112 | Number of trees — more trees reduce variance but increase runtime |
| `time_weight` | 64.0 | Weight of the time dimension in the embedding space |
| `distance_mode` | `"auto"` | `"precompute"` when N ≤ 24 444, `"streaming"` for larger datasets |
| `streaming_block_size` | 8192 | Block size for exact diameter computation in streaming mode |
| `distance_scale` | 1.0 | Global scaling factor applied to all distances |
| `random_state` | `None` | Seed for reproducibility |

> N here refers to the **total number of points** across all series (sum of all series lengths), not the number of series.

### Choosing `distance_mode`

- **`"precompute"`** — before building any tree, computes the full N×N pairwise Euclidean distance matrix of all points in RAM. Each ball query during tree construction then costs O(n) via a simple array comparison. This is the fastest option when you have enough RAM. **If your machine has ample memory (rule of thumb: total points N ≤ 24 444, or if N² floats fit comfortably in RAM), prefer `"precompute"` — it is noticeably faster than streaming.**

- **`"streaming"`** — instead of precomputing the N×N matrix, builds a `cKDTree` (low-dimensional data) or `BallTree` (high-dimensional data) spatial index. Each ball query during tree construction costs O(log N + k) where k is the number of points in the ball — no N×N matrix is ever allocated. `streaming_block_size` controls the block size used only for the exact diameter computation step. Use this when N is large and the full point-distance matrix would exhaust RAM.

- **`"auto"`** (default) — automatically selects `"precompute"` for N ≤ 24 444 and `"streaming"` otherwise.

---

## Experiment Notebooks

| Notebook | Task | Output |
|----------|------|--------|
| [experiment/knn.ipynb](experiment/knn.ipynb) | k-NN classification (Accuracy + MAP) | `result/TTSWD_knn.xlsx` |
| [experiment/kmedoid.ipynb](experiment/kmedoid.ipynb) | k-Medoids clustering (NMI + ACC) | `result/TTSWD_kmedoid.xlsx` |
| [experiment/tsne.ipynb](experiment/tsne.ipynb) | t-SNE embedding (Trustworthiness, Continuity, LCMC) | `result/TTSWD_tsne.xlsx` + scatter plots |

---

## Evaluation Metrics

### Classification (k-NN)

- **Accuracy** — fraction of correctly classified test samples (%).
- **MAP** (Mean Average Precision) — mean of the average precision at each of the k nearest neighbors.

### Clustering (k-Medoids)

- **NMI** (Normalized Mutual Information) — measures agreement between predicted cluster labels and ground-truth labels, normalized to [0, 1].
- **ACC** (Clustering Accuracy) — fraction of correctly assigned samples under the optimal label permutation (Hungarian algorithm).

### Embedding (t-SNE)

- **Trustworthiness** — degree to which neighbors in the embedding are true neighbors in the original space.
- **Continuity** — degree to which neighbors in the original space are preserved in the embedding.
- **LCMC** (Local Continuity Meta-Criterion) — composite measure combining trustworthiness and continuity.

---

## Output Files

```
result/
├── TTSWD_knn.xlsx          # Accuracy + MAP table across 
├── TTSWD_kmedoid.xlsx      # NMI + ACC table across 
├── TTSWD_tsne.xlsx         # Trust + Cont + LCMC table across datasets
└── <DatasetName>/
    └── TTSWD_tsne_seed0.png    # t-SNE scatter plot for the first run
```

---

## Performance Tips

1. **Install numba** — accelerates mass accumulation loops by ~10–50×.
2. **Use `"precompute"` mode on capable machines** — if your machine has sufficient RAM (e.g., 16 GB+), set `distance_mode="precompute"` explicitly even for datasets where `"auto"` would choose `"streaming"`. Precomputing the full N×N point-distance matrix once and reusing it for all trees is significantly faster than issuing spatial-index ball queries tree by tree.
3. **Use `"streaming"` for large datasets** — when the total number of points N > 24 444 or RAM is limited, `distance_mode="streaming"` avoids the O(N²) allocation. The spatial index (KDTree/BallTree) handles ball queries in O(log N + k) per query.
4. **Reduce `n_trees`** — 16 trees already yield stable distances for quick exploration.
5. **Parallelism** — the forest builder uses joblib threading automatically (up to `min(cpu_count, n_trees, 8)` threads).

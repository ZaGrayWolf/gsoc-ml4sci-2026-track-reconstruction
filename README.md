# GSoC 2026 – ML4SCI E2E6: Track Reconstruction (Task 2i)

**Author:** Kunwar Abhuday Singh  
**Organization:** ML4SCI  
**Project:** E2E6 — End-to-End Deep Learning Track Reconstruction at CMS  
**Kaggle Notebook:** [gsoc-2026-ml4sci-task-2i](https://www.kaggle.com/code/abhuday7/gsoc-2026-ml4sci-task-2i)

---

## Overview

This repository contains my implementation for the **GSoC 2026 ML4SCI evaluation task (Task 2i)**: quark/gluon jet classification on CMS calorimeter data using graph neural networks.

Two architectures are implemented and compared:

- **EdgeConv (DGCNN)** — dynamic graph recomputation per layer in feature space
- **Graph Attention Network (GAT)** — static graph with learned per-edge attention weights

The central question: does dynamic graph construction improve performance over a static graph baseline? The answer directly motivates the hypothesis in the E2E6 GSoC proposal — that EdgeConv may reduce fake rate under high pile-up conditions where static geometric graphs increasingly connect hits from unrelated tracks.

---

## Method

### Graph Construction

Each `125×125` CMS calorimeter image is converted to a graph:

- **Nodes** — non-zero energy pixels
- **Node features** — `[η_rel, φ_rel, p_T^track, E_ECAL, E_HCAL]`
- **Edges** — k-nearest neighbours (`k=8`) using ΔR distance:

$$\Delta R = \sqrt{\Delta\eta^2 + \Delta\phi^2}$$

ΔR is used over Euclidean pixel distance because the pixel-to-η,φ mapping is not isotropic — pixel distance is not a physically meaningful proximity measure in this context.

### Architecture Summary

| Model | Mechanism | Parameters |
|-------|-----------|------------|
| **EdgeConv (DGCNN)** | Dynamic graph recomputed per layer in feature space | 38,785 |
| **GAT** | Fixed ΔR graph, learned per-edge attention weights | 94,401 |

---

## Results

| Metric | EdgeConv | GAT |
|--------|----------|-----|
| Best Val AUC | 0.7954 | 0.7863 |
| **Test AUC** | **0.7885** | **0.7811** |
| Test Accuracy | 72.0% | 71.7% |
| Inference (ms/batch) | 62.6 ± 40.1 | 76.9 ± 19.6 |
| Parameters | 38,785 | 94,401 |

### Gluon Rejection at Fixed Quark Efficiency

| Quark Efficiency | EdgeConv | GAT |
|-----------------|----------|-----|
| 20% | 34.5× | 34.5× |
| 30% | 19.8× | 19.0× |
| 50% | **8.3×** | 7.8× |
| 70% | 3.9× | 3.7× |
| 80% | 2.7× | 2.6× |

### k-NN Ablation (EdgeConv)

| k | Val AUC |
|---|---------|
| 4 | 0.7832 |
| **8** | **0.7885** |
| 12 | 0.7892 |
| 16 | 0.7885 |

`k=8` chosen over `k=12`: the +0.0007 AUC difference is within noise at 100 validation batches (~6k jets), and `k=12` increases edge count by 50% with no meaningful physics gain.

---

## Key Finding

EdgeConv outperforms GAT by **+0.007 AUC** using **2.4× fewer parameters** and running **19% faster**. The gap is consistent across all 20 training epochs, ruling out statistical fluctuation.

The structural reason: EdgeConv rebuilds its graph in feature space at each layer, allowing it to cluster kinematically similar particles regardless of their fixed angular separation — whereas GAT is constrained to the initial ΔR topology. This result directly motivates testing whether dynamic graph topology reduces fake rate under HL-LHC pile-up conditions ($\mu \sim 200$), where static geometric graphs increasingly connect hits from unrelated interactions that are spatially nearby but kinematically distinct.

---

## Repository Structure

```
.
├── gsoc-2026-ml4sci-task-2i.ipynb   # Main implementation notebook
├── requirements.txt
├── README.md
└── LICENSE
```

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook gsoc-2026-ml4sci-task-2i.ipynb
```

The full notebook including training runs and plots is also available on Kaggle:  
https://www.kaggle.com/code/abhuday7/gsoc-2026-ml4sci-task-2i

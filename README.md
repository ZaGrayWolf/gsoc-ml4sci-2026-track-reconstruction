# GSoC 2026 – ML4SCI Track Reconstruction (E2E6)

This repository contains my implementation for the **GSoC 2026 ML4SCI evaluation task (Task 2i)** and preparatory work for the **E2E6 Track Reconstruction project at CMS**.

---

## Overview

I implemented and compared two graph neural network architectures for **quark/gluon jet classification** using CMS calorimeter data:

- **EdgeConv (DGCNN)** – dynamic graph recomputation
- **Graph Attention Network (GAT)** – static graph with attention

The goal was to evaluate whether **dynamic graph construction improves performance over static graph methods**, which directly motivates the hypothesis in my GSoC proposal.

---

## Method

### Graph Construction
- Converted **125×125 calorimeter images** into graphs
- Nodes = non-zero energy pixels
- Edges = **k-nearest neighbours (k=8)** using ΔR distance:
  
  ΔR = √(Δη² + Δφ²)

### Node Features
- Relative η
- Relative φ
- Track pT
- ECAL energy
- HCAL energy

---

## Models

| Model | Description |
|------|------------|
| **EdgeConv (DGCNN)** | Dynamic graph recomputed at each layer in feature space |
| **GAT** | Static graph with learned attention weights |

---

## Results

| Metric | EdgeConv | GAT |
|--------|----------|-----|
| **Test AUC** | **0.7885** | 0.7811 |
| Accuracy | 72.0% | 71.7% |
| Parameters | 38,785 | 94,401 |

### Key Observations
- EdgeConv achieves **+0.007 AUC improvement**
- Uses **2.4× fewer parameters**
- ~19% faster inference
- Performance improvement is consistent across epochs

---

## Key Insight

Dynamic graph recomputation allows the model to learn **feature-space neighbourhoods**, enabling better clustering of physically related particles compared to static graph structures.

This result motivates applying EdgeConv to **track reconstruction under high pile-up conditions**, where static geometric graphs may introduce large numbers of false edges.

---

## Repository Structure

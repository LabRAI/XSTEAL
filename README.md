# GNN-MEA: Decision-Boundary Sampling for Model Extraction Attacks on Graph Neural Networks

This repository contains the code, trained checkpoints, and experimental results for a study of **black-box model extraction attacks (MEA) against graph-classification GNNs**. The central contribution is a query strategy that builds the surrogate's training set from **graph pairs that straddle the victim's decision boundary**, identified by a two-phase search that combines exhaustive single-edge perturbation with explainer-guided Monte-Carlo edge sensitivity.

## Threat model

- **Black-box access only.** The attacker observes hard labels from the victim (`argmax` of the predicted distribution); no gradients, logits, weights, or architecture details are revealed.
- **Shadow set.** The original dataset is split 60% victim-train / 20% shadow / 20% held-out test. The attacker may only query the shadow partition.
- **Query budget.** Reported as a fraction of the shadow set, swept from 5% to 70%.
- **Local explainer.** The attacker is allowed to train a PGExplainer or GNNExplainer against the victim and use it to guide edge perturbations.
- **Surrogate architecture** is assumed to match the victim's family (GCN / GAT / GraphSAGE).

## Method

The proposed attack constructs a training set of `(G, G')` pairs in which `G'` is a minimal perturbation of `G` whose victim prediction differs from `G`'s. Such pairs lie on opposite sides of the decision boundary, so a surrogate trained on them learns the *shape* of the boundary rather than only the bulk of the input distribution.

The search proceeds in two phases:

- **Phase 1 — exhaustive single-edge deletion.** For every shadow graph, each edge is deleted in turn and the victim is queried. If any single deletion flips the predicted class, the resulting `(G, G')` pair is stored. Graphs whose prediction survives every single-edge deletion are bucketed for Phase 2.
- **Phase 2 — explainer-guided MC edge sensitivity.** For each non-boundary graph, an explainer (PGExplainer or GNNExplainer) restricts the candidate edge set to the subgraph it flags as important. For each candidate edge, a Monte-Carlo sensitivity score is estimated (`n_mc = 5` perturbations at flip probability `p_flip = 0.05`), and the top-3 edges by sensitivity are flipped. The procedure iterates up to 10 rounds or until `delta_max = 20` edges have been changed.

A post-hoc transition-balancing step keeps the `(class i → class j)` flip distribution uniform across class pairs, which prevents surrogate collapse on imbalanced datasets.

## Repository layout

```
GNN-MEA/
├── Colab Notebooks/                       # all code
│   ├── Datasets,_Victim,_Models,_and_Explainer.ipynb
│   ├── Decision_Boundary_Sampling_Empirical_Motivation.ipynb
│   ├── Main_Experiment_(GCN).ipynb
│   ├── Main_Experiment_(GAT_&_GraphSAGE).ipynb
│   ├── EGSteal-BB-Fidelity.ipynb
│   ├── Ablations.ipynb
│   ├── Phase_Ablation.ipynb
│   ├── Bar_Chart_Plot.ipynb
│   └── Fidelity_Trajectory_Plot.ipynb
│
├── Datasets, Victim Models, Explainers/   # trained checkpoints
│   ├── victims/{GCN,GAT,GraphSAGE}/<dataset>_victim.pt
│   └── explainers/{GCN,GAT,GraphSAGE}/<dataset>_explainer.pt
│
├── Boundary Results/                      # cached Phase-1 boundary pairs
│   └── <dataset>_boundary.pt
│
├── Results/                               # raw experiment outputs
│   ├── gcn_results.json
│   ├── gat_graphsage_results.json
│   ├── p1_vs_p2_ablation_results.json
│   └── experiment_results_for_plots.json
│
└── Visuals/                               # paper figures
    ├── Grouped_Bar/bar_fidelity_*.pdf
    └── Fidelity_Query/fidelity_ntrain_*.pdf
```

## Experimental setup

| | |
|---|---|
| **Architectures** | GCN, GAT (8 heads), GraphSAGE — 3 layers, hidden size 64, global mean pool |
| **Datasets** | AIDS, MUTAG, NCI1, PTC_FM, Tox21_AhR_training, Letter-low, Synthie, MNIST |
| **Explainers** | PGExplainer, GNNExplainer |
| **Baselines** | Random, Non-boundary, EGSteal-BB, Hybrid (50/50 Boundary + Random) |
| **Metric** | Fidelity (surrogate ↔ victim agreement on class-balanced test split); accuracy reported as secondary |
| **Seeds** | GCN: 3 seeds (42, 123, 456); GAT / GraphSAGE: single seed |

Victims are trained with Adam (`lr = 1e-3`, `wd = 5e-4`) under a grid of `hidden ∈ {64, 128} × epochs ∈ {300, 500, 700, 1000}`.

## Headline results

Fidelity at a 70% query budget on GCN (mean over 3 seeds):

| Dataset      | Random | EGSteal-BB | **Boundary** | Hybrid |
|--------------|--------|------------|--------------|--------|
| AIDS         | 0.868  | 0.658      | **0.911**    | 0.851  |
| MUTAG        | 0.818  | 0.909      | **1.000**    | 0.970  |
| NCI1         | 0.943  | 0.889      | **0.960**    | 0.922  |
| Tox21_AhR    | 0.607  | 0.548      | **0.893**    | 0.774  |
| PTC_FM       | 0.774  | 0.595      | 0.857        | **0.904** |
| Letter-low   | 0.686  | 0.689      | 0.630        | **0.721** |
| Synthie      | 0.397  | 0.430      | **0.513**    | 0.468  |
| MNIST        | 0.534  | 0.538      | 0.549        | **0.551** |

Boundary or Hybrid wins on every dataset; the largest absolute gains are on Tox21_AhR (+0.29 fidelity over Random) and MUTAG (+0.18). EGSteal-BB underperforms even Random on AIDS, PTC_FM, and Tox21. GAT and GraphSAGE exhibit the same ordering on binary tasks. On the multi-class, low-signal datasets (Letter-low, Synthie, MNIST) all methods perform near chance and Hybrid is the safer choice.

## Ablations

- **`Ablations.ipynb`** — runs **NoExp** (Phase 2 with random candidate edges, removing the explainer) and **NoMC** (`n_mc = 1`, removing the MC sensitivity estimator) on GCN × {AIDS, NCI1, Tox21_AhR}.
- **`Phase_Ablation.ipynb`** — runs **P1-only** vs **P2-only** for Boundary and Hybrid on the same three datasets. Phases are complementary: P2 dominates on Tox21 (P1-only 0.71 → P2-only 0.96) where single-edge flips are rare; P1 alone is sufficient on NCI1 (0.95) where graphs are small and dense.

## Reproducing the experiments

The notebooks are self-contained Colab notebooks. Required packages: `torch`, `torch-geometric`, `numpy`, `scikit-learn`, `matplotlib`. Run order:

1. `Datasets,_Victim,_Models,_and_Explainer.ipynb` — trains victims and explainers, writes checkpoints into `Datasets, Victim Models, Explainers/`. (Skip if using the bundled checkpoints.)
2. `Main_Experiment_(GCN).ipynb` and `Main_Experiment_(GAT_&_GraphSAGE).ipynb` — the full method-vs-baseline sweep across all datasets and query budgets.
3. `EGSteal-BB-Fidelity.ipynb` — the EGSteal-BB baseline.
4. `Ablations.ipynb` and `Phase_Ablation.ipynb` — component and phase ablations.
5. `Bar_Chart_Plot.ipynb` and `Fidelity_Trajectory_Plot.ipynb` — consume the JSON result files in `Results/` and emit the PDF figures in `Visuals/`.

The notebook `Decision_Boundary_Sampling_Empirical_Motivation.ipynb` is standalone and provides the empirical motivation for boundary-pair sampling.

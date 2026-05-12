# GNN-MEA

Code, trained checkpoints, and experimental results for a study of black-box model extraction attacks against graph-classification GNNs.

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

Victims are trained with Adam (`lr = 1e-3`, `wd = 5e-4`) under a grid of `hidden ∈ {64, 128} × epochs ∈ {300, 500, 700, 1000}`.

## Headline results

Fidelity at a 70% query budget on GCN:

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

## Reproducing the experiments

The notebooks are self-contained Colab notebooks. Required packages: `torch`, `torch-geometric`, `numpy`, `scikit-learn`, `matplotlib`. Run order:

1. `Datasets,_Victim,_Models,_and_Explainer.ipynb` — trains victims and explainers, writes checkpoints into `Datasets, Victim Models, Explainers/`. (Skip if using the bundled checkpoints.)
2. `Main_Experiment_(GCN).ipynb` and `Main_Experiment_(GAT_&_GraphSAGE).ipynb` — the full method-vs-baseline sweep across all datasets and query budgets.
3. `EGSteal-BB-Fidelity.ipynb` — the EGSteal-BB baseline.
4. `Ablations.ipynb` and `Phase_Ablation.ipynb` — component and phase ablations.
5. `Bar_Chart_Plot.ipynb` and `Fidelity_Trajectory_Plot.ipynb` — consume the JSON result files in `Results/` and emit the PDF figures in `Visuals/`.

The notebook `Decision_Boundary_Sampling_Empirical_Motivation.ipynb` is standalone and provides the empirical motivation for boundary-pair sampling.

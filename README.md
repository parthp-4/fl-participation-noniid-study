# Impact of Client Participation Rate on FL Algorithm Performance Under Varying Statistical Heterogeneity

A systematic experimental study comparing FedAvg, FedProx, and SCAFFOLD across a 2D grid of non-IID severity (Dirichlet α) and client participation rates.

## Research Gap Addressed

Li et al. (NIID-Bench, ICDE 2022) identified that SCAFFOLD performs poorly under partial participation (Finding 6), but tested only a single participation rate of 10% with 100 clients and a default α=0.5. No prior study has systematically quantified the **joint effect** of participation rate and non-IID severity on algorithm performance. This study presents the first 2D analysis of this interaction.

## Experimental Setup

| Parameter | Values |
|---|---|
| **Dataset** | MNIST |
| **Clients** | 10 |
| **Algorithms** | FedAvg, FedProx (μ=0.01), SCAFFOLD |
| **Non-IID settings** | Dirichlet α ∈ {0.05, 0.1, 0.5, 1.0} |
| **Participation rates** | {20%, 50%, 80%, 100%} |
| **Rounds** | 50 per experiment |
| **Local epochs** | 5 |
| **Total configurations** | 48 |

## Repository Structure

```
├── experiments/
│   ├── run_experiments.ipynb     # Colab notebook (run all 48 experiments)
│   ├── models.py                 # SimpleCNN definition
│   ├── data_utils.py             # Dirichlet partitioner
│   ├── fl_algorithms.py          # FedAvg, FedProx, SCAFFOLD
│   └── plot_results.py           # Figure generation
├── results/
│   └── .gitkeep
├── paper/
│   └── figures/                  # Exported publication figures
├── requirements.txt
└── .gitignore
```

## Running the Experiments

1. Open `experiments/run_experiments.ipynb` in Google Colab
2. Mount Google Drive (results are saved incrementally — crash-safe)
3. Run all cells in order
4. After completion, run `plot_results.py` or use the plotting cell in the notebook

## Citation

```
@inproceedings{porwal2026participation,
  title={Impact of Client Participation Rate on Federated Learning Algorithm 
         Performance Under Varying Statistical Heterogeneity},
  author={Porwal, Parth},
  year={2026}
}
```

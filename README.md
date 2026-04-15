# Impact of Client Participation Rate on Federated Learning Algorithm Performance Under Varying Statistical Heterogeneity

> A systematic experimental study comparing FedAvg, FedProx, and SCAFFOLD across a 2D grid of non-IID severity (Dirichlet α) and client participation rates.

---

## Overview

This repository accompanies our conference paper investigating the **joint effect** of client participation rate and statistical heterogeneity on federated learning algorithm performance. While prior benchmarking studies — most notably NIID-Bench (Li et al., ICDE 2022) — identified SCAFFOLD's failure under partial participation, their experiments were limited to a single operating point (10% participation, α = 0.5, 100 clients). No existing work has systematically mapped how participation rate and data heterogeneity **interact** across a comprehensive parameter grid.

We address this gap by evaluating three widely adopted aggregation strategies across a **4 × 4 experimental grid** (4 heterogeneity levels × 4 participation rates), yielding **48 distinct training configurations**.

## Key Findings

| Finding | Detail |
|---------|--------|
| **SCAFFOLD ceiling** | Peaked at 93.39% even under ideal conditions (α=1.0, p=1.0), vs. 98.05% FedAvg and 98.10% FedProx |
| **Model collapse** | At α=0.1 with 20% participation, all algorithms are vulnerable to catastrophic collapse; which one fails depends on the data partition (see paper Section V) |
| **Moderate heterogeneity** | At α ≥ 0.5, FedAvg and FedProx sustained 95–98% regardless of participation; SCAFFOLD remained at 86–93% |
| **FedProx dominance** | Best performer across the majority of the parameter space, particularly at moderate heterogeneity |

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
| Clients | 10 |
| Algorithms | FedAvg, FedProx (μ=0.01), SCAFFOLD |
| Non-IID settings | Dirichlet α ∈ {0.05, 0.1, 0.5, 1.0} |
| Participation rates | {20%, 50%, 80%, 100%} |
| Communication rounds | 30 |
| Local epochs | 1 |
| Total configurations | 48 |

## Repository Structure

```
├── paper/
│   ├── main.tex                    # LaTeX source (IEEE format)
│   └── figures/                    # Publication figures (5 PNGs)
│       ├── fig1_convergence_alpha01.png
│       ├── fig2_heatmap_all.png
│       ├── fig3_rounds_to_80.png
│       ├── fig4_participation_degradation.png
│       └── fig5_per_client_variance.png
├── experiments/
│   ├── run_experiments.ipynb       # Google Colab notebook (primary runner)
│   ├── run_fast.py                 # Local experiment runner (CPU/GPU)
│   ├── generate_figures.py         # Publication figure generation
│   ├── models.py                   # SimpleCNN architecture (unused legacy)
│   ├── data_utils.py               # Dirichlet-based data partitioner
│   └── fl_algorithms.py            # FedAvg, FedProx, SCAFFOLD implementations
├── results/
│   └── all_results.csv             # Complete results for all 48 experiments
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducing the Experiments

### Option A: Google Colab (Recommended)

1. Open `experiments/run_experiments.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Mount Google Drive — results are saved incrementally for crash resilience
3. Run all cells in order
4. After completion, run `experiments/generate_figures.py` to produce publication figures

### Option B: Local Execution

```bash
# Clone the repository
git clone https://github.com/parthp-4/fl-participation-noniid-study.git
cd fl-participation-noniid-study

# Install dependencies
pip install -r requirements.txt

# Run all 48 experiments (~3.5 hours on CPU)
python experiments/run_fast.py

# Generate publication figures
python experiments/generate_figures.py
```

> **Note:** The runner automatically skips completed experiments by checking `results/all_results.csv`, so partial runs resume seamlessly.

## Compiling the Paper

1. Create a new project on [Overleaf](https://www.overleaf.com/)
2. Upload `paper/main.tex` and the contents of `paper/figures/`
3. Set the compiler to **pdflatex**
4. Compile — Overleaf handles the `pdflatex → bibtex → pdflatex → pdflatex` cycle automatically

## Citation

```bibtex
@inproceedings{porwal2026participation,
  title     = {Impact of Client Participation Rate on Federated Learning Algorithm 
               Performance Under Varying Statistical Heterogeneity},
  author    = {Porwal, Parth},
  year      = {2026},
  note      = {Manipal University Jaipur}
}
```

## License

This project is released for academic and research purposes.

## Author

**Parth Porwal**  
Department of Information Technology, Manipal University Jaipur  
📧 parthporwal4@gmail.com

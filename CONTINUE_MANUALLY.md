# Manual Continuation Guide — FL Study

## ✅ STATUS: Experiments & Paper COMPLETE

All 48 experiments have been completed and the LaTeX paper has been fully populated with actual results. The only remaining steps are GitHub setup and Overleaf compilation.

---

## Step 1: Push to GitHub

```bash
# 1. Create a new repository on github.com named: fl-participation-noniid-study
# 2. Then push:
cd /Users/parthporwal4/Desktop/work/FL/ag
git init
git add .
git commit -m "Initial commit: FL participation x heterogeneity study"
git branch -M main
git remote add origin https://github.com/parthporwal04/fl-participation-noniid-study.git
git push -u origin main
```

> **Note**: The `results/` directory is in `.gitignore`. If you want to include the CSV in the repo, remove the `results/` line from `.gitignore` before pushing.

---

## Step 2: Compile Paper in Overleaf

1. **Create a new Overleaf project** (Blank Project)
2. **Upload files**:
   - `paper/main.tex` → rename to `main.tex` in Overleaf root
   - Create a `figures/` folder in Overleaf
   - Upload all 5 PNG files from `paper/figures/` into the `figures/` folder
3. **Compile** using the following sequence:
   - Set compiler to `pdflatex` in Overleaf settings
   - Overleaf handles the `pdflatex → bibtex → pdflatex → pdflatex` cycle automatically
4. **Verify**: All 5 figures should render, all tables should populate

---

## Step 3: Final Checks

### Paper Quality Checklist
- [ ] Abstract has specific numbers (92.32% → 42.29% SCAFFOLD drop, etc.)
- [ ] All 5 figures render correctly
- [ ] Decision matrix (Table II) reflects actual best performers
- [ ] Author email is correct: 23fe10ite00030@muj.manipal.edu
- [ ] GitHub URL in footnote matches your actual repo
- [ ] References section compiles without warnings

### Key Data Points for Quick Reference
| Metric | FedAvg | FedProx | SCAFFOLD |
|--------|--------|---------|----------|
| Best accuracy | 98.05% | 98.10% | 93.39% |
| Worst accuracy | 11.95% | 14.41% | 42.29% |
| Best at α=0.05, p=0.2 | 57.76% | 27.63% | 42.29% |
| Best at α=1.0, p=1.0 | 98.05% | 97.98% | 93.39% |

---

## File Locations

| File | Path |
|------|------|
| LaTeX paper | `paper/main.tex` |
| Figures | `paper/figures/fig{1-5}_*.png` |
| Raw results | `results/all_results.csv` |
| Experiment runner | `experiments/run_fast.py` |
| Figure generator | `experiments/generate_figures.py` |
| Colab notebook | `experiments/run_experiments.ipynb` |
| Conda env | `fl_study` (Python 3.11 + PyTorch) |

---

## If You Need to Re-run Experiments

```bash
# Activate the conda environment
conda activate fl_study

# Re-run all 48 experiments (~3.5 hours on CPU)
cd /Users/parthporwal4/Desktop/work/FL/ag
python experiments/run_fast.py

# Re-generate figures
python experiments/generate_figures.py
```

The runner auto-skips completed experiments (checks `results/all_results.csv`), so partial runs resume automatically.

# ML741 Assignment 4

Code for the self-organising map experiments in the assignment 4 report (Machine Learning 741, Stellenbosch University).

- `data.py` — dataset in the assignment 2 k-NN representation, min-max scaling, distinct vectors, scoring
- `som.py` — batch self-organising map, topographic error, adjacency-constrained Ward clustering, cluster labelling and prediction
- `tune.py <stage>` — staged tuning on the fold-0 inner split (`lattice`, `sigma`, `epochs`, `density`, `size`, `labelling`); writes `tuning/<stage>.csv` and `params.json`
- `experiment.py` — ten-fold evaluation of the tuned map; writes `results/folds.csv` and `results/confusion.npz`
- `analysis.py` — Wilcoxon tests against the assignment 2 tree and k-NN, per-class scores and summary statistics; writes `results/tests.csv`, `results/per_class.csv` and `results/stats.csv`

The dataset (`networkTraffic.csv`, from STEMlearn) goes in the directory above this one. The assignment 2 result files (`ct_results.csv`, `knn_results.csv` and the confusion npz files) are read from `../../assignment2/code`. Run the tuning stages in order, then `experiment.py` and `analysis.py`.

Python 3.14.3 with NumPy 2.4.2, SciPy 1.17.0, pandas 3.0.5 and scikit-learn 1.8.0.

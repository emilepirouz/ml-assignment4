"""Statistics from the ten-fold results, the assignment 2 baselines and the
tuning records."""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import data

folds = pd.read_csv("results/folds.csv")
som = folds[folds["variant"] == "chosen"].reset_index(drop=True)
variant = {v: d.reset_index(drop=True) for v, d in folds.groupby("variant")}

# best assignment 2 tree and k-NN by mean macro F1, on the same folds
a2 = "../../assignment2/code"
models = {"som": som}
confusion = {"som": np.load("results/confusion.npz")["confusion"]}
for name, keys in [
    ("ct", ["criterion", "control", "value"]),
    ("knn", ["metric", "weighting", "k"]),
]:
    d = pd.read_csv(f"{a2}/{name}_results.csv")
    best = d.groupby(keys)["macro_f1"].mean().idxmax()
    rows = (d[keys] == list(best)).all(axis=1)
    models[name] = d[rows].sort_values("fold").reset_index(drop=True)
    npz = np.load(f"{a2}/{name}_confusion.npz")
    confusion[name] = npz["matrices"][
        np.all([npz[k] == v for k, v in zip(keys, best)], axis=0)
    ][0]

# Wilcoxon signed-rank tests on per-fold macro F1, Holm-adjusted within each
# family
families = [
    [("SOM - CT", models["ct"]), ("SOM - kNN", models["knn"])],
    [
        ("chosen - majority", variant["majority"]),
        ("chosen - neuron level", variant["neuron_chosen"]),
    ],
]
tests = []
for family in families:
    rows = []
    for name, other in family:
        a, b = som["macro_f1"].to_numpy(), other["macro_f1"].to_numpy()
        stat, pval = wilcoxon(a, b)
        rows.append({"pair": name, "difference": (a - b).mean(), "W": stat, "p": pval})
    fam = pd.DataFrame(rows)
    pvals = fam["p"].to_numpy()
    holm = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, (len(pvals) - rank) * pvals[i])
        holm[i] = min(1.0, running)
    fam["holm"] = holm
    tests.append(fam)
tests = pd.concat(tests)
tests.to_csv("results/tests.csv", index=False)
print(tests.round(4).to_string(index=False))

# per-class F1 and recall of the three models, and the recall of each class
# when every duplicated feature vector takes its majority class
per_class = pd.DataFrame({"class": data.CLASSES})
for m in ["ct", "knn", "som"]:
    per_class[f"f1_{m}"] = models[m][[f"f1_c{c}" for c in range(10)]].mean().to_numpy()
    per_class[f"recall_{m}"] = np.diag(confusion[m]) / confusion[m].sum(axis=1)
x, y = data.load()
keys = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
table = pd.crosstab(keys, y)
major = table.idxmax(axis=1)
per_class["recall_majority"] = [
    table.loc[major == c, c].sum() / table[c].sum() for c in range(10)
]
per_class.to_csv("results/per_class.csv", index=False)

stats = {
    "accuracy_ceiling": table.max(axis=1).sum() / len(y),
    "som_seconds_mean": som["seconds"].mean(),
    "som_occupied_mean": som["occupied"].mean(),
    "som_test_qe_mean": som["test_qe"].mean(),
    "som_test_te_mean": som["test_te"].mean(),
}
for m, d in models.items():
    stats[f"{m}_macro_f1_mean"] = d["macro_f1"].mean()
    stats[f"{m}_macro_f1_std"] = d["macro_f1"].std()
    stats[f"{m}_accuracy_mean"] = d["accuracy"].mean()
    stats[f"{m}_accuracy_std"] = d["accuracy"].std()
for v, d in variant.items():
    stats[f"{v}_macro_f1_mean"] = d["macro_f1"].mean()
    stats[f"{v}_accuracy_mean"] = d["accuracy"].mean()
pd.Series(stats).to_csv("results/stats.csv", header=False)

# mean over seeds of every setting in the tuning stages
for stage, keys in [
    ("lattice", ["lattice", "init"]),
    ("sigma", ["ratio", "sigma_t"]),
    ("epochs", ["epochs"]),
    ("density", ["density"]),
    ("size", ["side"]),
]:
    d = pd.read_csv(f"tuning/{stage}.csv")
    print(d.groupby(keys)[["f1_majority", "val_qe", "val_te", "empty"]].mean().round(4))

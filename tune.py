"""Staged tuning on the fold-0 inner split: python tune.py <stage>, stages in
the order lattice, sigma, epochs, density, size, labelling."""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import data
import som

SEEDS = range(5)
GRIDS = {
    "lattice": [
        {"lattice": g, "init": i} for g in ["rect", "hex"] for i in ["sample", "pca"]
    ],
    "sigma": [{"ratio": r, "sigma_t": t} for r in [0.5, 0.25] for t in [0.5, 1.0, 2.0]],
    "epochs": [{"epochs": e} for e in [50, 100, 200]],
    "density": [{"density": d} for d in ["weighted", "distinct"]],
    "size": [{"side": s} for s in range(10, 101, 10)],
}
COST = {"epochs": "epochs", "size": "side"}
CLUSTERS = [
    10,
    20,
    30,
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    5000,
]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def select(results: list[tuple[dict, list[float]]], cost: str | None = None) -> dict:
    # results holds (setting, scores over seeds); the best mean wins, or with a
    # cost key the cheapest setting within one standard error of the best
    means = [np.mean(f1) for _, f1 in results]
    best = int(np.argmax(means))
    if cost is None:
        return results[best][0]
    se = pd.Series(results[best][1]).sem() if len(results[best][1]) > 1 else 0.0
    within = [r for r, m in zip(results, means) if m >= means[best] - se]
    return min(within, key=lambda r: r[0][cost])[0]


stage = sys.argv[1]
params = {
    "lattice": "rect",
    "init": "sample",
    "ratio": 0.5,
    "sigma_t": 1.0,
    "epochs": 100,
    "density": "weighted",
    "side": 30,
}
if os.path.exists("params.json"):
    params |= json.load(open("params.json"))

x, y = data.load()
train_idx, _ = next(StratifiedKFold(10, shuffle=True, random_state=441).split(y, y))
fit_idx, val_idx = train_test_split(
    train_idx, test_size=0.2, stratify=y[train_idx], random_state=441
)
x_fit, x_val, xd, cnt = data.prepare(x[fit_idx], x[val_idx])
y_fit, y_val = y[fit_idx], y[val_idx]

if stage == "labelling":
    rows = []
    weight = cnt if params["density"] == "weighted" else np.ones(len(xd))
    for seed in SEEDS if params["init"] == "sample" else [0]:
        w, adj, _ = som.train(params, xd, weight, seed)
        winner, _ = som.bmn(x_fit, w)
        hits = np.bincount(winner, minlength=len(w))
        merges, unconstrained = som.ward(w, adj, hits)
        occupied = np.flatnonzero(hits)
        val_winner = occupied[som.bmn(x_val, w[occupied])[0]]
        for c in [c for c in CLUSTERS if c < len(occupied)] + [len(occupied)]:
            for alpha in ALPHAS:
                pred = som.predict(merges, hits, c, winner, y_fit, alpha, val_winner)
                rows.append(
                    {
                        "seed": seed,
                        "occupied": len(occupied),
                        "unconstrained": unconstrained,
                        "clusters": c,
                        "alpha": alpha,
                    }
                    | data.score(y_val, pred)
                )
        print(f"seed {seed}: {len(occupied)} occupied neurons", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("tuning/labelling.csv", index=False)
    neuron = df[df["clusters"] == df["occupied"]]
    params |= select(
        [({"alpha": a}, d["macro_f1"].tolist()) for a, d in neuron.groupby("alpha")]
    )
    grid = df[(df["alpha"] == params["alpha"]) & df["clusters"].isin(CLUSTERS)]
    params |= select(
        [
            ({"clusters": c}, d["macro_f1"].tolist())
            for c, d in grid.groupby("clusters")
        ],
        "clusters",
    )
else:
    rows, results = [], []
    for setting in GRIDS[stage]:
        p = params | setting
        weight = cnt if p["density"] == "weighted" else np.ones(len(xd))
        f1 = []
        for seed in SEEDS if p["init"] == "sample" else [0]:
            start = time.perf_counter()
            w, adj, train_qe = som.train(p, xd, weight, seed)
            seconds = time.perf_counter() - start
            winner, _ = som.bmn(x_fit, w)
            hits = np.bincount(winner, minlength=len(w))
            occupied = np.flatnonzero(hits)
            val_winner = occupied[som.bmn(x_val, w[occupied])[0]]
            row = {
                **p,
                "seed": seed,
                "seconds": seconds,
                "train_qe": train_qe,
                "val_qe": som.bmn(x_val, w)[1].mean(),
                "val_te": som.topographic_error(x_val, w, adj),
                "empty": 1.0 - (hits > 0).mean(),
            }
            for alpha, name in ((0.0, "majority"), (1.0, "normalised")):
                pred = som.predict(
                    [], hits, len(occupied), winner, y_fit, alpha, val_winner
                )
                s = data.score(y_val, pred)
                row[f"f1_{name}"] = s["macro_f1"]
                row[f"acc_{name}"] = s["accuracy"]
            rows.append(row)
            f1.append(row["f1_majority"])
            print(
                f"{setting} seed {seed}: f1 {row['f1_majority']:.4f} "
                f"qe {row['val_qe']:.4f} te {row['val_te']:.4f} "
                f"empty {row['empty']:.3f} {seconds:.0f}s",
                flush=True,
            )
        results.append((setting, f1))
    pd.DataFrame(rows).to_csv(f"tuning/{stage}.csv", index=False)
    params |= select(results, COST.get(stage))

json.dump(params, open("params.json", "w"), indent=2)
print(params)

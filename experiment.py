"""Ten-fold evaluation of the tuned map on the assignment 2 folds."""

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import data
import som

p = json.load(open("params.json"))
x, y = data.load()
rows = []
confusion = np.zeros((10, 10), dtype=np.int64)
folds = StratifiedKFold(10, shuffle=True, random_state=441).split(y, y)
for fold, (train_idx, test_idx) in enumerate(folds):
    x_train, x_test, xd, cnt = data.prepare(x[train_idx], x[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]
    weight = cnt if p["density"] == "weighted" else np.ones(len(xd))
    start = time.perf_counter()
    w, adj, train_qe = som.train(p, xd, weight, fold)
    seconds = time.perf_counter() - start

    winner, _ = som.bmn(x_train, w)
    hits = np.bincount(winner, minlength=len(w))
    merges, unconstrained = som.ward(w, adj, hits)
    occupied = np.flatnonzero(hits)
    test_winner = occupied[som.bmn(x_test, w[occupied])[0]]
    test_qe = som.bmn(x_test, w)[1].mean()
    test_te = som.topographic_error(x_test, w, adj)

    # the chosen labelling, the majority rule at the same cluster count, and
    # both rules at neuron level
    variants = {
        "chosen": (p["clusters"], p["alpha"]),
        "majority": (p["clusters"], 0.0),
        "neuron_chosen": (len(occupied), p["alpha"]),
        "neuron_majority": (len(occupied), 0.0),
    }
    for variant, (c, alpha) in variants.items():
        pred = som.predict(merges, hits, c, winner, y_train, alpha, test_winner)
        rows.append(
            {
                "fold": fold,
                "variant": variant,
                "clusters": c,
                "alpha": alpha,
                "seconds": seconds,
                "occupied": len(occupied),
                "unconstrained": unconstrained,
                "train_qe": train_qe,
                "test_qe": test_qe,
                "test_te": test_te,
            }
            | data.score(y_test, pred)
        )
        if variant == "chosen":
            confusion += confusion_matrix(y_test, pred, labels=list(range(10)))
    chosen = rows[-4]
    print(
        f"fold {fold}: acc {chosen['accuracy']:.4f} f1 {chosen['macro_f1']:.4f} "
        f"occupied {len(occupied)} {seconds:.0f}s",
        flush=True,
    )
pd.DataFrame(rows).to_csv("results/folds.csv", index=False)
np.savez_compressed("results/confusion.npz", confusion=confusion)

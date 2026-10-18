"""Dataset in the assignment 2 k-NN representation, scaling and scoring."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASSES = [
    "Normal",
    "Reconnaissance",
    "Backdoor",
    "DoS",
    "Exploits",
    "Analysis",
    "Fuzzers",
    "Worms",
    "Shellcode",
    "Generic",
]


def load() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("../networkTraffic.csv")
    y = df.pop("attack_cat").to_numpy(dtype=np.int8)
    df = df.drop(columns=["id", "is_ftp_login", "sloss", "dloss", "tcprtt"])
    nominal = ["proto", "service", "state"]
    numeric = df.drop(columns=nominal)
    skewed = numeric.columns[numeric.skew() > 2]
    numeric[skewed] = np.log1p(numeric[skewed])
    dummies = pd.get_dummies(df[nominal], dtype=np.float32)
    x = pd.concat([numeric.astype(np.float32), dummies], axis=1).to_numpy()
    return x, y


def prepare(
    x_train: np.ndarray, x_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # min-max scaling fitted on the training part, then the distinct training
    # vectors and how many times each occurs
    lo = x_train.min(axis=0)
    span = x_train.max(axis=0) - lo
    span[span == 0] = 1.0
    x_train = (x_train - lo) / span
    x_other = (x_other - lo) / span
    keys = pd.util.hash_pandas_object(pd.DataFrame(x_train), index=False).to_numpy()
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    return x_train, x_other, x_train[first], counts.astype(np.float64)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    f1 = f1_score(y_true, y_pred, average=None, labels=list(range(10)), zero_division=0)
    row = {"accuracy": float((y_true == y_pred).mean()), "macro_f1": f1.mean()}
    return row | {f"f1_c{c}": f1[c] for c in range(10)}

import json, numpy as np, pandas as pd, imagehash
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix, classification_report)

OUT = "C:/Users/lokes/Downloads/citrus_cv_project/code/out"
CLASSES = ["fresh", "grenning", "canker", "blackspot"]


def load():
    d = np.load(f"{OUT}/arrays.npz")
    df = pd.read_csv(f"{OUT}/index.csv")
    return d["X1"], d["X2"], df


def make_splits(df, seed=0, thr=4):
    """Leak-safe splits.
    1) drop train images that are near-duplicates (pHash<=thr) of any test image
    2) group near-duplicate train images (augmented copies) so they never straddle train/val
    3) stratified-group split -> ~1/6 of train as validation
    Returns index arrays (into df) for train, val, test and a dict of audit numbers."""
    hs = [imagehash.hex_to_hash(h) for h in df.phash]
    tr = df.index[df.split == "train"].to_numpy()
    te = df.index[df.split == "test"].to_numpy()
    leak = {j for j in tr for i in te if hs[i] - hs[j] <= thr}
    tr = np.array([j for j in tr if j not in leak])
    # union-find over train near-dupes
    parent = {j: j for j in tr}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a_i, a in enumerate(tr):
        for b in tr[a_i + 1:]:
            if hs[a] - hs[b] <= thr:
                parent[find(a)] = find(b)
    groups = np.array([find(j) for j in tr])
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    fit, val = next(sgkf.split(tr, df.y.values[tr], groups))
    audit = dict(train_dropped_for_test_leak=len(leak), n_groups=len(set(groups)),
                 n_fit=len(fit), n_val=len(val), n_test=len(te))
    return tr[fit], tr[val], te, audit


def metrics(y, p, n_boot=1000, seed=0):
    out = dict(
        accuracy=float(accuracy_score(y, p)),
        macro_f1=float(f1_score(y, p, average="macro")),
        weighted_f1=float(f1_score(y, p, average="weighted")),
        macro_recall=float(recall_score(y, p, average="macro")),
        macro_precision=float(precision_score(y, p, average="macro", zero_division=0)),
        per_class_recall=dict(zip(CLASSES, recall_score(y, p, average=None, labels=range(4)).round(3).tolist())),
        per_class_precision=dict(zip(CLASSES, precision_score(y, p, average=None, labels=range(4), zero_division=0).round(3).tolist())),
        per_class_f1=dict(zip(CLASSES, f1_score(y, p, average=None, labels=range(4)).round(3).tolist())),
        confusion=confusion_matrix(y, p, labels=range(4)).tolist(),
    )
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        bs.append(f1_score(y[i], p[i], average="macro"))
    out["macro_f1_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out

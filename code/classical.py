"""Approach A: hand-crafted colour/texture/lesion features + SVM / Random Forest."""
import sys; sys.path.insert(0, "C:/Users/lokes/Downloads/citrus_cv_project/code")
import time, json, pickle, numpy as np, cv2
from skimage.feature import local_binary_pattern
from scipy import ndimage as ndi
from scipy.stats import skew
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from common import *

GRAY = 124


def features(img):
    rgb = img
    fg = ~np.all(rgb == GRAY, axis=2)           # S2 fills background with exact gray; S1 -> all True
    if fg.sum() < 200:
        fg = np.ones(fg.shape, bool)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    f = []
    # colour histograms (fruit pixels only if isolated)
    h = hsv[..., 0][fg]; s = hsv[..., 1][fg]; v = hsv[..., 2][fg]
    f += list(np.histogram(h, 18, (0, 180))[0] / len(h))
    f += list(np.histogram(s, 8, (0, 256))[0] / len(s))
    f += list(np.histogram(v, 8, (0, 256))[0] / len(v))
    # colour moments in HSV and Lab
    for im in (hsv, lab):
        for c in range(3):
            x = im[..., c][fg].astype(np.float32)
            f += [x.mean(), x.std(), float(skew(x))]
    # texture: uniform LBP at two scales on V
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(g, P, R, "uniform")
        hist = np.bincount(lbp[fg].astype(int), minlength=P + 2) / fg.sum()
        f += list(hist)
    # lesion cues: dark blobs relative to the fruit's own median brightness
    gs = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
    med = np.median(gs[fg])
    fg_er = cv2.erode(fg.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)  # ignore silhouette edge
    if fg_er.sum() < 100:
        fg_er = fg
    for ratio in (0.45, 0.6, 0.75):
        dark = (gs < ratio * med) & fg_er
        lab_, n = ndi.label(dark)
        areas = ndi.sum(dark, lab_, range(1, n + 1)) if n else np.array([0.0])
        f += [dark.sum() / fg_er.sum(), n, float(areas.mean()), float(areas.max())]
    # green fraction (greening/unripe cue) and orange fraction
    f += [((h > 35) & (h < 85) & (s > 60)).mean(), ((h > 5) & (h < 25) & (s > 120)).mean()]
    return np.array(f, np.float32)


def featurize(X):
    return np.stack(Parallel(n_jobs=2)(delayed(features)(x) for x in X))


if __name__ == "__main__":
    X1, X2, df = load()
    fit, val, te, audit = make_splits(df)
    print("split audit", audit)
    y = df.y.values
    res = {"audit": audit}
    F = {}
    for name, X in (("S1_asis", X1), ("S2_isolated", X2)):
        t0 = time.time(); F[name] = featurize(X); print(name, F[name].shape, f"{time.time()-t0:.1f}s")
    pickle.dump(F, open(f"{OUT}/classical_feats.pkl", "wb"))

    for name in F:
        Fx = F[name]
        cands = {}
        for C in (1, 10, 100):
            for g in ("scale", 0.01, 0.001):
                cands[f"svm_C{C}_g{g}"] = make_pipeline(StandardScaler(), SVC(C=C, gamma=g, class_weight="balanced"))
        for n in (300,):
            cands[f"rf_{n}"] = RandomForestClassifier(n_estimators=n, class_weight="balanced_subsample", n_jobs=2, random_state=0)
        scores = {}
        for k, m in cands.items():
            m.fit(Fx[fit], y[fit]); scores[k] = f1_score(y[val], m.predict(Fx[val]), average="macro")
        best_svm = max((k for k in scores if k.startswith("svm")), key=scores.get)
        best_rf = "rf_300"
        res[name] = {"val_macro_f1": {k: round(v, 4) for k, v in scores.items()}, "models": {}}
        for tag, k in (("SVM", best_svm), ("RandomForest", best_rf)):
            m = cands[k]  # already fit on fit split
            t0 = time.time(); p = m.predict(Fx[te]); dt = (time.time() - t0) / len(te)
            res[name]["models"][tag] = dict(config=k, val_macro_f1=round(scores[k], 4),
                                            test=metrics(y[te], p), predict_ms_per_img_excl_features=dt * 1000)
            pickle.dump(m, open(f"{OUT}/clf_{name}_{tag}.pkl", "wb"))
            np.save(f"{OUT}/pred_classical_{name}_{tag}.npy", p)
            print(name, tag, k, "val", round(scores[k], 3), "test acc", round(res[name]["models"][tag]["test"]["accuracy"], 3),
                  "macroF1", round(res[name]["models"][tag]["test"]["macro_f1"], 3))

    # shortcut check: train on as-is features, test on isolated-fruit features (and vice versa)
    res["cross"] = {}
    for tr_name, te_name in (("S1_asis", "S2_isolated"), ("S2_isolated", "S1_asis")):
        for tag in ("SVM", "RandomForest"):
            m = pickle.load(open(f"{OUT}/clf_{tr_name}_{tag}.pkl", "rb"))
            p = m.predict(F[te_name][te])
            res["cross"][f"train_{tr_name}__test_{te_name}__{tag}"] = metrics(y[te], p)
            print("cross", tr_name, "->", te_name, tag, round(res["cross"][f"train_{tr_name}__test_{te_name}__{tag}"]["macro_f1"], 3))
    # single-image end-to-end latency (features + SVM), S1
    m = pickle.load(open(f"{OUT}/clf_S1_asis_SVM.pkl", "rb"))
    t0 = time.time()
    for i in te[:30]:
        m.predict(features(X1[i])[None])
    res["e2e_ms_per_img_S1_SVM"] = (time.time() - t0) / 30 * 1000
    json.dump(res, open(f"{OUT}/classical_results.json", "w"), indent=1)

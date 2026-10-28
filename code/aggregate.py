import sys; sys.path.insert(0, "C:/Users/lokes/Downloads/citrus_cv_project/code")
import json, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *

C = json.load(open(f"{OUT}/classical_results.json"))
X1, X2, df = load()
fit, val, te, audit = make_splits(df)
y = df.y.values[te]
rows, summ = [], {}

# classical rows
for scn in ("S1_asis", "S2_isolated"):
    for tag in ("SVM", "RandomForest"):
        m = C[scn]["models"][tag]["test"]
        rows.append(dict(model=f"A: {tag}", input=scn, seeds=1, acc=m["accuracy"], macro_f1=m["macro_f1"], f1_lo=m["macro_f1_ci95"][0],
                         f1_hi=m["macro_f1_ci95"][1], f1_sd=0.0, recall_min=min(m["per_class_recall"].values()),
                         cross_f1=C["cross"][f"train_{scn}__test_{'S2_isolated' if scn=='S1_asis' else 'S1_asis'}__{tag}"]["macro_f1"]))
# cnn rows
cnn = {}
for scn in ("S1_asis", "S2_isolated"):
    R = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/cnn_{scn}_s*.json"))]
    cnn[scn] = R
    if not R: continue
    accs = [r["test"]["accuracy"] for r in R]; f1s = [r["test"]["macro_f1"] for r in R]
    rows.append(dict(model="B: SmallCNN (mean of seeds)", input=scn, seeds=len(R), acc=np.mean(accs), macro_f1=np.mean(f1s),
                     f1_lo=np.mean([r["test"]["macro_f1_ci95"][0] for r in R]), f1_hi=np.mean([r["test"]["macro_f1_ci95"][1] for r in R]),
                     f1_sd=np.std(f1s), recall_min=np.mean([min(r["test"]["per_class_recall"].values()) for r in R]),
                     cross_f1=np.mean([r["cross_test_other_input"]["macro_f1"] for r in R])))
T = pd.DataFrame(rows)
T.round(3).to_csv(f"{OUT}/comparison_table.csv", index=False)
print(T.round(3).to_string())

# CNN ensemble (mean of seed probs) on test
ens = {}
for scn in ("S1_asis", "S2_isolated"):
    ps = [np.load(p) for p in sorted(glob.glob(f"{OUT}/prob_cnn_{scn}_s*.npy"))]
    if ps:
        ens[scn] = metrics(y, np.mean(ps, 0).argmax(1)); print("ENS", scn, round(ens[scn]["accuracy"], 3), round(ens[scn]["macro_f1"], 3))
summ["cnn_ensemble"] = ens
summ["cnn_params"] = cnn["S2_isolated"][0]["params"] if cnn["S2_isolated"] else None
summ["cnn_latency_ms"] = {s: float(np.mean([r["latency_ms_per_img"] for r in cnn[s]])) for s in cnn if cnn[s]}
summ["cnn_seed_f1"] = {s: [round(r["test"]["macro_f1"], 4) for r in cnn[s]] for s in cnn}
summ["cnn_seed_acc"] = {s: [round(r["test"]["accuracy"], 4) for r in cnn[s]] for s in cnn}
summ["classical_e2e_ms"] = C["e2e_ms_per_img_S1_SVM"]
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)

# ---- figure: confusion matrices (best per approach on isolated input + as-is)
def cm_ax(ax, cm, title):
    cm = np.array(cm); n = cm.sum(1, keepdims=True)
    ax.imshow(cm / n, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] / n[i, 0] > .5 else "black", fontsize=11)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_title(title, fontsize=9); ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)

if ens:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for r_i, scn in enumerate(("S1_asis", "S2_isolated")):
        cm_ax(axes[r_i, 0], C[scn]["models"]["SVM"]["test"]["confusion"], f"A: SVM · {scn}")
        cm_ax(axes[r_i, 1], C[scn]["models"]["RandomForest"]["test"]["confusion"], f"A: Random Forest · {scn}")
        cm_ax(axes[r_i, 2], ens[scn]["confusion"], f"B: CNN (3-seed ensemble) · {scn}")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_confusion.png", dpi=130); plt.close()

    # ---- figure: macro-F1 with bootstrap CI + shortcut check
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    lab = [f"{r.model.split(' (')[0]}\n{r.input.replace('_', ' ')}" for r in T.itertuples()]
    xx = np.arange(len(T))
    col = ["#2a9d8f" if r.model.startswith("A") else "#e76f51" for r in T.itertuples()]
    ax[0].bar(xx, T.macro_f1, color=col, yerr=[T.macro_f1 - T.f1_lo, T.f1_hi - T.macro_f1], capsize=3)
    ax[0].set_xticks(xx); ax[0].set_xticklabels(lab, fontsize=7); ax[0].set_ylim(.5, 1.02); ax[0].set_title("Test macro-F1 (bars: bootstrap 95% CI, n=99)", fontsize=10)
    ax[1].bar(xx - .2, T.macro_f1, .4, label="same input type as training", color=col)
    ax[1].bar(xx + .2, T.cross_f1, .4, label="other input type (background changed)", color=col, alpha=.4, hatch="//")
    ax[1].set_xticks(xx); ax[1].set_xticklabels(lab, fontsize=7); ax[1].set_title("Shortcut check: macro-F1 when the background cue is swapped", fontsize=10); ax[1].legend(fontsize=7)
    for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_compare.png", dpi=130); plt.close()

    # ---- training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for k, scn in enumerate(("S1_asis", "S2_isolated")):
        for r in cnn[scn]:
            ax[k].plot([h["epoch"] for h in r["history"]], [h["val_macro_f1"] for h in r["history"]], label=f"seed {r['seed']}")
        ax[k].set_title(f"CNN validation macro-F1 · {scn}", fontsize=10); ax[k].set_xlabel("epoch"); ax[k].legend(fontsize=7); ax[k].spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_curves.png", dpi=130); plt.close()

    # ---- errors gallery (CNN ensemble, isolated input)
    ps = np.mean([np.load(p) for p in sorted(glob.glob(f"{OUT}/prob_cnn_S2_isolated_s*.npy"))], 0)
    wrong = np.where(ps.argmax(1) != y)[0]
    print("CNN-ens S2 errors:", [(df.cls[te[i]], CLASSES[ps[i].argmax()], df.path[te[i]].split('/')[-1]) for i in wrong])
    if len(wrong):
        n = min(len(wrong), 8); fig, ax = plt.subplots(2, 4, figsize=(11, 6))
        for a_ in ax.ravel(): a_.axis("off")
        for k, i in enumerate(wrong[:n]):
            ax.ravel()[k].imshow(X1[te[i]]); ax.ravel()[k].set_title(f"true {df.cls[te[i]]}\npred {CLASSES[ps[i].argmax()]} ({ps[i].max():.2f})", fontsize=8)
        plt.tight_layout(); plt.savefig(f"{OUT}/fig_errors.png", dpi=110); plt.close()

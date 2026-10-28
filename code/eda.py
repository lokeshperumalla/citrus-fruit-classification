import os, sys, json, hashlib
import numpy as np, pandas as pd
from PIL import Image, ImageFile
import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "C:/Users/lokes/Downloads/Orange Dataset"
OUT = "C:/Users/lokes/Downloads/citrus_cv_project/code/out"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["fresh", "grenning", "canker", "blackspot"]

rows = []
for split in ["train", "test"]:
    for c in CLASSES:
        d = f"{ROOT}/{split}/{c}"
        for f in sorted(os.listdir(d)):
            p = f"{d}/{f}"
            r = dict(path=p, split=split, cls=c, file=f, ext=f.rsplit(".", 1)[-1].lower(),
                     bytes=os.path.getsize(p), ok=True)
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    r["mode"] = im.mode
                    r["w"], r["h"] = im.size
                    g = im.convert("RGB")
                    a = np.asarray(g.resize((64, 64)), dtype=np.float32)
                    r["bright"] = a.mean()
                    r["contrast"] = a.std()
                    hsv = np.asarray(g.resize((64, 64)).convert("HSV"), dtype=np.float32)
                    r["hue"], r["sat"] = hsv[..., 0].mean(), hsv[..., 1].mean()
                    # background proxy: mean of border pixels
                    b = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                                        a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
                    r["border_bright"] = b.mean()
                    r["md5"] = hashlib.md5(open(p, "rb").read()).hexdigest()
                    r["phash"] = str(imagehash.phash(g))
                    # blur: variance of laplacian
                    import cv2
                    gray = cv2.cvtColor(np.asarray(g), cv2.COLOR_RGB2GRAY)
                    r["blur"] = cv2.Laplacian(gray, cv2.CV_64F).var()
            except Exception as e:
                r["ok"] = False
                r["err"] = str(e)
            rows.append(r)

df = pd.DataFrame(rows)
df["aspect"] = df["w"] / df["h"]
df["mpix"] = df["w"] * df["h"] / 1e6
df.drop(columns=["phash"]).to_csv(f"{OUT}/meta.csv", index=False)

res = {}
res["n_total"] = len(df)
res["corrupt"] = int((~df.ok).sum())
res["counts"] = df.groupby(["split", "cls"]).size().unstack(fill_value=0).to_dict()
tr = df[(df.split == "train") & df.ok]
cnt = tr.cls.value_counts()
res["train_imbalance_ratio_max_min"] = round(cnt.max() / cnt.min(), 2)
res["size_summary"] = df[["w", "h", "mpix", "bytes"]].describe().round(2).to_dict()
res["unique_sizes"] = int(df.groupby(["w", "h"]).ngroups)
res["top_sizes"] = df.groupby(["w", "h"]).size().sort_values(ascending=False).head(8).reset_index().values.tolist()
res["ext_by_class"] = df.groupby(["cls", "ext"]).size().unstack(fill_value=0).to_dict()
res["mode_counts"] = df["mode"].value_counts().to_dict()
res["per_class_stats"] = df.groupby("cls")[["w", "h", "bright", "border_bright", "sat", "hue", "blur", "bytes"]].mean().round(1).to_dict()

# exact dupes
dup = df[df.duplicated("md5", keep=False)]
res["exact_dup_files"] = int(len(dup))
res["exact_dup_groups"] = int(dup.md5.nunique())
# cross-class / cross-split exact dupes
x = dup.groupby("md5").agg(ncls=("cls", "nunique"), nsplit=("split", "nunique"))
res["exact_dup_cross_class"] = int((x.ncls > 1).sum())
res["exact_dup_cross_split"] = int((x.nsplit > 1).sum())

# near dupes via phash (hamming <= 4), cross-split leakage
hs = [imagehash.hex_to_hash(h) for h in df.phash.fillna("0" * 16)]
idx = df.index.tolist()
near_pairs = []
tr_idx = df.index[df.split == "train"].tolist()
te_idx = df.index[df.split == "test"].tolist()
for i in te_idx:
    for j in tr_idx:
        if hs[i] - hs[j] <= 4:
            near_pairs.append((i, j))
res["near_dup_test_vs_train_pairs"] = len(near_pairs)
res["near_dup_test_images_leaking"] = len({i for i, _ in near_pairs})
# within-train near dupes
w = 0
seen = set()
for a_i, a in enumerate(tr_idx):
    for b in tr_idx[a_i + 1:]:
        if hs[a] - hs[b] <= 4:
            w += 1
            seen.add(a); seen.add(b)
res["near_dup_within_train_pairs"] = w
res["near_dup_within_train_images"] = len(seen)
# label conflicts among near dupes
conf = 0
for i, j in near_pairs:
    if df.loc[i, "cls"] != df.loc[j, "cls"]:
        conf += 1
res["near_dup_test_train_label_conflicts"] = conf

# outliers: size (IQR on mpix), brightness, blur
def iqr_out(s):
    q1, q3 = s.quantile([.25, .75]); k = 1.5 * (q3 - q1)
    return (s < q1 - k) | (s > q3 + k)
for col in ["mpix", "bright", "blur", "aspect"]:
    m = iqr_out(df[col])
    res[f"outliers_{col}"] = int(m.sum())
    res[f"outliers_{col}_by_class"] = df[m].cls.value_counts().to_dict()
df["out_mpix"] = iqr_out(df.mpix); df["out_bright"] = iqr_out(df.bright); df["out_blur"] = iqr_out(df.blur)
df[["path", "cls", "split", "w", "h", "bright", "blur", "out_mpix", "out_bright", "out_blur"]].to_csv(f"{OUT}/outliers.csv", index=False)

# train/test distribution shift
res["test_class_share"] = (df[df.split == "test"].cls.value_counts(normalize=True).round(3)).to_dict()
res["train_class_share"] = (tr.cls.value_counts(normalize=True).round(3)).to_dict()
json.dump(res, open(f"{OUT}/eda.json", "w"), indent=1, default=str)

# ---- figures
colors = {"fresh": "#2a9d8f", "grenning": "#e9c46a", "canker": "#e76f51", "blackspot": "#264653"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for k, s in enumerate(["train", "test"]):
    v = df[df.split == s].cls.value_counts().reindex(CLASSES)
    ax[k].bar(v.index, v.values, color=[colors[c] for c in v.index])
    for i, n in enumerate(v.values): ax[k].text(i, n, str(n), ha="center", va="bottom")
    ax[k].set_title(f"{s} class distribution"); ax[k].spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_class_dist.png", dpi=140); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for c in CLASSES:
    s = df[df.cls == c]
    ax[0].scatter(s.w, s.h, s=8, alpha=.5, label=c, color=colors[c])
ax[0].set_xlabel("width"); ax[0].set_ylabel("height"); ax[0].set_title("Image sizes"); ax[0].legend(fontsize=7)
df.boxplot(column="bright", by="cls", ax=ax[1], grid=False); ax[1].set_title("Mean brightness by class"); ax[1].set_xlabel("")
df.boxplot(column="border_bright", by="cls", ax=ax[2], grid=False); ax[2].set_title("Border (background) brightness by class"); ax[2].set_xlabel("")
plt.suptitle(""); plt.tight_layout(); plt.savefig(f"{OUT}/fig_bias.png", dpi=140); plt.close()

# sample grid
fig, axes = plt.subplots(4, 6, figsize=(13, 8.8))
rng = np.random.RandomState(0)
for r, c in enumerate(CLASSES):
    s = tr[tr.cls == c].sample(6, random_state=0)
    for k, p in enumerate(s.path):
        axes[r, k].imshow(Image.open(p).convert("RGB")); axes[r, k].axis("off")
        if k == 0: axes[r, k].set_title(c, loc="left", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_samples.png", dpi=110); plt.close()
print(json.dumps(res, indent=1, default=str))

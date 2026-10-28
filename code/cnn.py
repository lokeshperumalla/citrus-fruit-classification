"""Approach B: lightweight CNN trained from scratch (no pretrained weights reachable in this sandbox)."""
import sys; sys.path.insert(0, "C:/Users/lokes/Downloads/citrus_cv_project/code")
import time, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from common import *

torch.set_num_threads(2)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_t(X):
    return ((torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255) - MEAN) / STD


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.f = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 96))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(96, n))

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.f(x), 1).flatten(1))


def augment(x):
    """Flips, 90-degree rotations, random crop-and-resize, mild brightness/contrast. Hue is NOT touched (colour is the signal)."""
    B = x.shape[0]
    out = []
    for i in range(B):
        a = x[i]
        if np.random.rand() < .5: a = a.flip(2)
        if np.random.rand() < .5: a = a.flip(1)
        a = torch.rot90(a, np.random.randint(4), (1, 2))
        s = np.random.uniform(0.75, 1.0)
        h = int(128 * s); y0 = np.random.randint(0, 128 - h + 1); x0 = np.random.randint(0, 128 - h + 1)
        a = F.interpolate(a[None, :, y0:y0 + h, x0:x0 + h], size=128, mode="bilinear", align_corners=False)[0]
        a = a * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.15, 0.15)
        out.append(a)
    return torch.stack(out)


def predict(model, X, bs=128):
    model.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ps.append(model(X[i:i + bs]).softmax(1))
    return torch.cat(ps).numpy()


def run(scn, seed, epochs, save=True):
    X1, X2, df = load()
    X = to_t(X1 if scn == "S1_asis" else X2)
    Xo = to_t(X2 if scn == "S1_asis" else X1)       # the "other" input kind, for the shortcut check
    y = torch.from_numpy(df.y.values)
    fit, val, te, audit = make_splits(df)          # same split as classical for a fair comparison
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN()
    nparams = sum(p.numel() for p in model.parameters())
    cnt = np.bincount(df.y.values[fit], minlength=4)
    w = torch.tensor(cnt.sum() / (4 * cnt), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    bs = 32
    steps = epochs * int(np.ceil(len(fit) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(fit); t0 = time.time(); tl = 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb = augment(X[idx]), y[idx]
            loss = F.cross_entropy(model(xb), yb, weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        pv = predict(model, X[val]).argmax(1)
        f1v = f1_score(y[val].numpy(), pv, average="macro")
        hist.append(dict(epoch=ep + 1, loss=tl / len(fit), val_macro_f1=float(f1v), sec=time.time() - t0))
        if f1v >= best:
            best, best_state = f1v, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"{scn} s{seed} ep{ep+1:02d} loss {tl/len(fit):.3f} valF1 {f1v:.3f} {time.time()-t0:.1f}s", flush=True)
    model.load_state_dict(best_state)
    P = predict(model, X[te]); p = P.argmax(1)
    Po = predict(model, Xo[te]).argmax(1)
    r = dict(scenario=scn, seed=seed, params=nparams, best_val_macro_f1=float(best), history=hist,
             test=metrics(y[te].numpy(), p), cross_test_other_input=metrics(y[te].numpy(), Po))
    t0 = time.time()
    for i in te[:40]:
        predict(model, X[i:i + 1])
    r["latency_ms_per_img"] = (time.time() - t0) / 40 * 1000
    if save:
        torch.save(best_state, f"{OUT}/cnn_{scn}_s{seed}.pt"); np.save(f"{OUT}/pred_cnn_{scn}_s{seed}.npy", p)
        np.save(f"{OUT}/prob_cnn_{scn}_s{seed}.npy", P)
        json.dump(r, open(f"{OUT}/cnn_{scn}_s{seed}.json", "w"), indent=1)
    print("DONE", scn, seed, "test acc", round(r["test"]["accuracy"], 3), "macroF1", round(r["test"]["macro_f1"], 3),
          "cross F1", round(r["cross_test_other_input"]["macro_f1"], 3), flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scn", default="S1_asis"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--nosave", action="store_true")
    a = ap.parse_args()
    run(a.scn, a.seed, a.epochs, save=not a.nosave)

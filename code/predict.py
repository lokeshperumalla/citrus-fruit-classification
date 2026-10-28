"""Example run: classify one image with both approaches.
usage: python predict.py <image> [--scn S2_isolated|S1_asis] [--viz out.png]"""
import sys; sys.path.insert(0, "C:/Users/lokes/Downloads/citrus_cv_project/code")
import argparse, glob, pickle, time, numpy as np, torch, torch.nn.functional as F, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from prep import load_rgb, fruit_mask, crop_square, GRAY, S
from classical import features
from cnn import SmallCNN, to_t
from common import CLASSES, OUT


def preprocess(path):
    im = load_rgb(path)
    rgb = np.asarray(im.resize((256, 256), Image.BILINEAR))
    s1 = cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA)
    m = fruit_mask(rgb)
    cr, cm = crop_square(rgb, m)
    s2 = cv2.resize(np.where(cm[..., None], cr, GRAY).astype(np.uint8), (S, S), interpolation=cv2.INTER_AREA)
    return s1, s2


def gradcam(model, x, cls):
    acts = {}
    h = model.f.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    x = x.clone().requires_grad_(True)
    out = model(x); h.remove()
    a = acts["a"]; g = torch.autograd.grad(out[0, cls], a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(S, S), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--scn", default="S2_isolated"); ap.add_argument("--viz", default=None)
    a = ap.parse_args()
    s1, s2 = preprocess(a.image)
    img = s1 if a.scn == "S1_asis" else s2
    res = {}
    # --- Approach A: features + classifier
    t0 = time.time(); f = features(img)[None]
    for tag in ("RandomForest", "SVM"):
        clf = pickle.load(open(f"{OUT}/clf_{a.scn}_{tag}.pkl", "rb"))
        if hasattr(clf, "predict_proba"):
            pr = clf.predict_proba(f)[0]
        else:  # SVC without probability -> softmax over decision scores
            d = clf.decision_function(f)[0]; pr = np.exp(d) / np.exp(d).sum()
        res[f"A:{tag}"] = (CLASSES[int(np.argmax(pr))], float(pr.max()), (time.time() - t0) * 1000)
    # --- Approach B: CNN seed ensemble
    t0 = time.time(); x = to_t(img[None]); probs = []; models = []
    for p in sorted(glob.glob(f"{OUT}/cnn_{a.scn}_s*.pt")):
        m = SmallCNN(); m.load_state_dict(torch.load(p)); m.eval(); models.append(m)
        with torch.no_grad(): probs.append(m(x).softmax(1)[0].numpy())
    pr = np.mean(probs, 0); k = int(pr.argmax())
    res["B:CNN(ensemble of %d)" % len(models)] = (CLASSES[k], float(pr.max()), (time.time() - t0) * 1000)
    print(f"Image: {a.image}  | input mode: {a.scn}")
    for name, (c, conf, ms) in res.items():
        print(f"  {name:<24} -> {c:<10} conf {conf:.2f}   ({ms:.0f} ms)")
    print("  CNN class probabilities:", {c: round(float(v), 3) for c, v in zip(CLASSES, pr)})
    if a.viz:
        cam = gradcam(models[0], x, k)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        over = (0.55 * img + 0.45 * heat).astype(np.uint8)
        fig, ax = plt.subplots(1, 4, figsize=(15, 3.9))
        orig = np.asarray(load_rgb(a.image))
        ax[0].imshow(orig); ax[0].set_title("Original")
        ax[1].imshow(img); ax[1].set_title("Model input (%s)" % a.scn)
        ax[2].imshow(over); ax[2].set_title(f"Grad-CAM for '{CLASSES[k]}'")
        ax[3].barh(CLASSES[::-1], pr[::-1], color=["#264653" if i != 3 - k else "#e76f51" for i in range(4)])
        ax[3].set_xlim(0, 1); ax[3].set_title("CNN probabilities")
        for x_ in ax[:3]: x_.axis("off")
        ax[3].spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); plt.savefig(a.viz, dpi=130); plt.close()


if __name__ == "__main__":
    main()

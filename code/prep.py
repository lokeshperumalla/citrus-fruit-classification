"""Build cached arrays. S1 = images as-is (resized), S2 = fruit isolated on neutral gray."""
import os, numpy as np, pandas as pd, cv2, imagehash
from PIL import Image
from scipy import ndimage as ndi

ROOT = "C:/Users/lokes/Downloads/Orange Dataset"
OUT = "C:/Users/lokes/Downloads/citrus_cv_project/code/out"
CLASSES = ["fresh", "grenning", "canker", "blackspot"]
S = 128
GRAY = np.array([124, 124, 124], np.uint8)


def load_rgb(p):
    im = Image.open(p)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


def fruit_mask(rgb):
    """Fruit = saturated pixels, not black padding, largest blob, holes filled."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (s > 70) & (v > 45)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    lab, n = ndi.label(m)
    if n == 0:
        return np.ones(m.shape, bool)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    if m.mean() < 0.08:  # segmentation failed -> keep everything
        return np.ones(m.shape, bool)
    return m


def crop_square(rgb, m, pad=0.06):
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max(y1 - y0, x1 - x0) / 2 * (1 + pad)
    H, W = m.shape
    ya, yb = int(max(0, cy - half)), int(min(H, cy + half))
    xa, xb = int(max(0, cx - half)), int(min(W, cx + half))
    return rgb[ya:yb, xa:xb], m[ya:yb, xa:xb]


def build():
    rows, X1, X2, fail = [], [], [], 0
    for split in ["train", "test"]:
        for ci, c in enumerate(CLASSES):
            d = f"{ROOT}/{split}/{c}"
            for f in sorted(os.listdir(d)):
                p = f"{d}/{f}"
                im = load_rgb(p)
                rgb = np.asarray(im.resize((256, 256), Image.BILINEAR))
                X1.append(cv2.resize(rgb, (S, S), interpolation=cv2.INTER_AREA))
                m = fruit_mask(rgb)
                if m.all():
                    fail += 1
                cr, cm = crop_square(rgb, m)
                out = np.where(cm[..., None], cr, GRAY).astype(np.uint8)
                X2.append(cv2.resize(out, (S, S), interpolation=cv2.INTER_AREA))
                rows.append(dict(path=p, split=split, cls=c, y=ci, phash=str(imagehash.phash(im)),
                                 mask_frac=float(m.mean())))
    df = pd.DataFrame(rows)
    np.savez_compressed(f"{OUT}/arrays.npz", X1=np.stack(X1), X2=np.stack(X2))
    df.to_csv(f"{OUT}/index.csv", index=False)
    print("images", len(df), "seg_failures", fail)


if __name__ == "__main__":
    build()

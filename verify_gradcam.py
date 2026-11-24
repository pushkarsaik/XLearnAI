"""
verify_gradcam.py - check that the Grad-CAM heatmaps are working and trustworthy.

Run from the XLearnAI folder:
    python verify_gradcam.py                 # 8 fractured + 8 normal test images
    python verify_gradcam.py --n 20          # more images per class

It writes gradcam_check/grid.png (original | heatmap for each image) and prints
three checks:

1. Prediction check  - P(fractured) for each image and whether the AI was right.
2. Deletion test     - grey out the 10% of the image the heatmap marks as most
                       important, and separately a same-sized patch at a random
                       place. If the heatmap is faithful, hiding the "hot" area
                       should lower the fracture score much more.
3. Sanity check      - scramble the model's last-layer weights and recompute the
                       maps. A real explanation must change when the model changes;
                       if the maps stay the same, they only show image edges.
"""
import argparse
import glob
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

import gradcam

HERE = os.path.dirname(os.path.abspath(__file__))


def pick(n, seed):
    rng = random.Random(seed)
    out = []
    for cls in ("fractured", "not_fractured"):
        files = sorted(glob.glob(os.path.join(HERE, "dataset", "test", cls, "*")))
        out += [(f, cls) for f in rng.sample(files, min(n, len(files)))]
    return out


def frac_score(model, img):
    """Raw fracture score (logit) - unlike P(fractured) it doesn't get stuck at 1.00."""
    x = gradcam.preprocess(img)
    feats_unused, logit, _ = gradcam._FN_CACHE.setdefault(id(model), gradcam._fracture_logit_fn(model))(x, _NoTape())
    return float(logit[0])


class _NoTape:
    def watch(self, _):
        pass


def hide(img, mask):
    """Replace masked pixels with the image's mean grey."""
    arr = np.asarray(img, dtype="float32").copy()
    arr[mask] = arr.mean()
    return Image.fromarray(arr.astype("uint8"))


def deletion(model, img, cam, frac=0.10, rng=None):
    """Hide the hottest 10% of the image vs. the same-shaped patch moved to a random place."""
    small = img.resize(gradcam.IMG_SIZE)
    c = gradcam._upsample(cam, small.size)
    k = int(frac * c.size)
    hot = np.zeros(c.size, bool); hot[np.argsort(c.ravel())[-k:]] = True
    hot = hot.reshape(c.shape)
    h, w = c.shape
    rnd = np.roll(hot, (int(rng.integers(h // 4, 3 * h // 4)), int(rng.integers(w // 4, 3 * w // 4))), axis=(0, 1))
    base = frac_score(model, small)
    return base, frac_score(model, hide(small, hot)), frac_score(model, hide(small, rnd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="images per class")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model = load_model(os.path.join(HERE, "fracture_model.h5"), compile=False)
    items = pick(args.n, args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(HERE, "gradcam_check")
    os.makedirs(out_dir, exist_ok=True)

    print("\n1) Predictions + 2) deletion test (fracture score: before -> hot area hidden / same-size random area hidden)")
    print(f"{'image':38} {'true':13} {'P(frac)':>7} {'right':>5}  {'score':>6} {'hot-hid':>7} {'rnd-hid':>7}")
    tiles, cams, drops_hot, drops_rnd, correct = [], [], [], [], 0
    T = 220
    for f, cls in items:
        img = Image.open(f).convert("RGB")
        prob, cam = gradcam.explain(model, img)
        pf = gradcam.fracture_probability(prob)
        right = (pf >= 0.5) == (cls == "fractured")
        correct += right
        cams.append(cam)

        # deletion test only makes sense when the AI says "fracture"
        if pf >= 0.5:
            b, h, r = deletion(model, img, cam, rng=rng)
            drops_hot.append(b - h); drops_rnd.append(b - r)
            extra = f"{b:6.1f} {h:7.1f} {r:7.1f}"
        else:
            extra = f"{'-':>6} {'-':>7} {'-':>7}"
        print(f"{os.path.basename(f)[:38]:38} {cls:13} {pf:7.2f} {'yes' if right else 'NO':>5}  {extra}")

        box = gradcam.focus_box(cam) if pf >= 0.5 else None
        ov = gradcam.overlay_heatmap(img, cam, 0.8, box=box)
        tile = Image.new("RGB", (2 * T, T + 22), "white")
        tile.paste(img.resize((T, T)), (0, 22)); tile.paste(ov.resize((T, T)), (T, 22))
        ImageDraw.Draw(tile).text((4, 5), f"{cls}  P(frac)={pf:.2f}  {'OK' if right else 'WRONG'}", fill="black")
        tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * (2 * T + 10), rows * (T + 32)), "#777")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * (2 * T + 10), (i // cols) * (T + 32)))
    grid.save(os.path.join(out_dir, "grid.png"))

    print(f"\nAccuracy on these {len(items)} images: {correct}/{len(items)}")
    if drops_hot:
        print(f"Average drop in fracture score: hiding HOT area = {np.mean(drops_hot):.1f}, "
              f"hiding a RANDOM same-size area = {np.mean(drops_rnd):.1f}")
        print("   -> PASS: heatmap points at what the AI actually uses" if np.mean(drops_hot) > 1.5 * max(np.mean(drops_rnd), 0.1)
              else "   -> WEAK: heatmap is not clearly better than random")

    # 3) sanity check: scramble last-layer weights
    head = model.layers[-1]
    W = head.kernel.numpy()
    head.kernel.assign(np.random.default_rng(0).permutation(W))
    corrs = []
    for (f, _), cam in zip(items, cams):
        _, cam2 = gradcam.explain(model, Image.open(f).convert("RGB"))
        if cam.std() > 0 and cam2.std() > 0:
            corrs.append(np.corrcoef(cam.ravel(), cam2.ravel())[0, 1])
    head.kernel.assign(W)
    print(f"\n3) Sanity check: similarity of maps after scrambling the model = {np.mean(corrs):.2f} "
          f"(near 0 = good, near 1 = maps don't depend on the model)")
    print(f"\nSaved {os.path.join(out_dir, 'grid.png')}")


if __name__ == "__main__":
    main()

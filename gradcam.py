"""
gradcam.py - Grad-CAM "where did the AI look?" maps for X-Learn AI.

Model in fracture_model.h5 (see model.py):
    Conv(32) -> Pool -> Conv(64) -> Pool -> Conv(128) -> Pool -> Flatten
    -> Dense(128) -> Dropout -> Dense(1, sigmoid)

Training used flow_from_directory on dataset/{fractured,not_fractured}, so the
classes are indexed alphabetically: fractured = 0, not_fractured = 1. The single
sigmoid output is therefore P(NOT fractured), and P(fractured) = 1 - output.

What this module does, and why:
  * The map always answers ONE question: "which parts of the image pushed the AI
    towards FRACTURE?" (not "towards whatever it predicted"), so a student reads
    every map the same way.
  * Gradients are taken from the final layer's raw score (the logit), not from the
    sigmoid. The model is very confident (outputs near 0 or 1), where the sigmoid
    is flat and its gradient is almost zero; the old maps were mostly noise for
    that reason.
  * Edge strips of the image are ignored by default: the conv layers have no
    padding, so the outermost pixels barely reach the map, and the earlier maps
    showed many hot spots on image borders.

Usage (unchanged for app.py):
    from gradcam import explain, overlay_heatmap, fracture_probability, focus_box
    prob, cam = explain(model, pil_image)        # prob = raw sigmoid output
    p_frac    = fracture_probability(prob)
    heat_img  = overlay_heatmap(pil_image, cam, strength=0.8, box=focus_box(cam))
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFilter

IMG_SIZE = (128, 128)          # must match model.py
FRACTURE_IS_LOW = True         # sigmoid output = P(not fractured)

# Which conv layer to explain. conv2d_2 is the last one (28x28 map): best
# balance between "understands bone shapes" and spatial detail.
TARGET_LAYER = None            # None = last Conv2D in the model
EDGE_MARGIN = 0.04             # fraction of each side ignored (image borders)

_RESAMPLE = getattr(Image, "Resampling", Image)


# ------------------------------------------------------------------ helpers
def preprocess(pil_img, size=IMG_SIZE):
    """Same as training: RGB, resized to 128x128, pixel values / 255.
    (flow_from_directory resizes with 'nearest'; we use the same.)"""
    img = pil_img.convert("RGB").resize(size, _RESAMPLE.NEAREST)
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr[None, ...]


def _layers(model):
    return [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]


def _target_layer(model):
    layers = _layers(model)
    if TARGET_LAYER:
        return model.get_layer(TARGET_LAYER)
    for layer in reversed(layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the model.")


def _fracture_logit_fn(model):
    """Return f(batch) -> (feature_map, fracture_logit, sigmoid_output).

    Runs the layers one by one so we can grab the conv feature map, and computes
    the last Dense layer by hand (x @ W + b) to get the logit before the sigmoid.
    """
    layers = _layers(model)
    target = _target_layer(model)
    head = layers[-1]
    if not isinstance(head, tf.keras.layers.Dense):
        raise ValueError("Expected the model to end in a Dense layer.")
    W, b = head.kernel, head.bias

    def run(batch, tape):
        x = tf.convert_to_tensor(batch)
        feats = None
        for layer in layers[:-1]:
            x = layer(x, training=False)
            if layer is target:
                feats = x
                tape.watch(feats)
        z = tf.matmul(x, W) + b                 # logit of "not fractured"
        frac_logit = -z[:, 0] if FRACTURE_IS_LOW else z[:, 0]
        return feats, frac_logit, tf.sigmoid(z[:, 0])

    return run


# cache one compiled function per model object
_FN_CACHE = {}


def make_gradcam(model, batch):
    """Grad-CAM for the FRACTURE class.
    Returns (cam in [0,1] at feature-map resolution, raw sigmoid output)."""
    key = id(model)
    if key not in _FN_CACHE:
        _FN_CACHE[key] = _fracture_logit_fn(model)
    run = _FN_CACHE[key]

    with tf.GradientTape() as tape:
        feats, frac_logit, prob = run(batch, tape)
    grads = tape.gradient(frac_logit, feats)            # (1, H, W, C)

    weights = tf.reduce_mean(grads, axis=(1, 2))        # one weight per channel
    cam = tf.reduce_sum(feats * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()                        # keep "pro-fracture" evidence only

    # ignore a thin strip along every edge
    if EDGE_MARGIN > 0:
        h, w = cam.shape
        mh, mw = max(1, int(round(h * EDGE_MARGIN))), max(1, int(round(w * EDGE_MARGIN)))
        cam[:mh, :] = 0; cam[-mh:, :] = 0; cam[:, :mw] = 0; cam[:, -mw:] = 0

    peak = np.percentile(cam, 99.5)
    cam = np.clip(cam / (peak + 1e-8), 0.0, 1.0) if peak > 0 else np.zeros_like(cam)
    return cam, float(prob[0])


def explain(model, pil_img):
    """One forward/backward pass. Returns (raw sigmoid output, fracture cam)."""
    cam, prob = make_gradcam(model, preprocess(pil_img))
    return prob, cam


def fracture_probability(prob, fracture_is_low=FRACTURE_IS_LOW):
    """Raw sigmoid output -> P(fractured)."""
    return 1.0 - prob if fracture_is_low else prob


# ------------------------------------------------------------------ drawing
def _upsample(cam, size):
    """Smooth, image-sized version of the cam (values 0-1)."""
    w, h = size
    img = Image.fromarray((cam * 255).astype("uint8")).resize((w, h), _RESAMPLE.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=max(w, h) / 120))
    return np.asarray(img, dtype="float32") / 255.0


def focus_box(cam, level=0.6):
    """Bounding box (x0, y0, x1, y1), in 0-1 image coordinates, around the hottest
    region of the cam. Returns None if the map is empty."""
    if cam.max() <= 0:
        return None
    from collections import deque
    mask = cam >= level * cam.max()
    h, w = mask.shape
    seen = np.zeros_like(mask)
    best, best_score = None, -1.0
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            q, pts = deque([(y0, x0)]), []
            seen[y0, x0] = True
            while q:
                y, x = q.popleft(); pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; q.append((ny, nx))
            ys, xs = zip(*pts)
            score = float(cam[list(ys), list(xs)].sum())
            if score > best_score:
                best_score = score
                best = (min(xs) / w, min(ys) / h, (max(xs) + 1) / w, (max(ys) + 1) / h)
    return best


def _jet(v):
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(pil_img, cam, strength=0.8, threshold=0.25, box=None):
    """Blend the fracture heatmap onto the image.
    Below `threshold` the X-ray is untouched; above it the colour fades in, up to
    `strength` opacity. If `box` (from focus_box) is given, a circle is drawn
    around that region."""
    base = pil_img.convert("RGB")
    c = _upsample(cam, base.size)
    heat = _jet(c) * 255.0
    a = (np.clip((c - threshold) / (1.0 - threshold), 0, 1) * strength)[..., None]
    out = np.asarray(base, dtype="float32") * (1 - a) + heat * a
    out = Image.fromarray(out.clip(0, 255).astype("uint8"))

    if box is not None:
        w, h = out.size
        x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        pad = 0.04 * max(w, h)
        d = ImageDraw.Draw(out)
        d.ellipse([x0 - pad, y0 - pad, x1 + pad, y1 + pad],
                  outline=(255, 255, 255), width=max(3, int(max(w, h) / 150)))
    return out

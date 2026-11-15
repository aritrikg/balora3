"""Data contracts and image operations for the BALoRA results app."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
METHODS = {
    "frozen": "Frozen SAM2",
    "deterministic": "Deterministic LoRA",
    "balora": "Gaussian Decoder LoRA",
}
COLORS = ["#7789aa", "#2e58c5", "#007c74"]
METRICS = {
    "Dice": ("dice", "Higher indicates more overlap."),
    "IoU": ("iou", "Higher indicates more overlap."),
    "HD95 (px)": ("hd95", "Lower is better. Distances are measured on resized 2D slices."),
    "ASSD (px)": ("assd", "Lower is better. Distances are measured on resized 2D slices."),
    "ECE": ("ece", "Lower is better. This calculation includes background pixels."),
    "Brier score": ("brier", "Lower is better. This calculation includes background pixels."),
    "Negative log-likelihood": ("nll", "Lower is better. This calculation includes background pixels."),
}
PAIR = "Equal volume–organ pairs"
PROMPT = "Equal prompts"
REFERENCE_COLOR = (80, 231, 170)
PREDICTION_COLOR = (255, 181, 79)
PROMPT_COLOR = (255, 255, 170)
MI_COLORS = np.asarray([
    [14, 9, 39], [74, 19, 105], [168, 43, 104], [242, 108, 64], [254, 241, 145]
])


def organ_label(slug):
    return slug.replace("_", " ").capitalize()


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def read_table(name):
    return pd.read_csv(DATA / name)


def aggregate(records, groups, metric, weighting):
    """Average within each volume–organ pair before pooling when requested."""
    if weighting == PAIR:
        pairs = list(dict.fromkeys([*groups, "volume", "organ"]))
        records = records.groupby(pairs, as_index=False, observed=True)[metric].mean()
    return records.groupby(groups, as_index=False, observed=True)[metric].mean()


def read_case(stem):
    case_dir = DATA / "examples"
    with np.load(case_dir / f"{stem}.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def edge(mask):
    """Render one-pixel inner contours, including boundaries at the image edge."""
    padded = np.pad(mask, 1)
    core = (
        padded[:-2, 1:-1] & padded[2:, 1:-1]
        & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return mask & ~core


def crop_bounds(arrays, x, y):
    region = arrays["reference"].astype(bool)
    for method in METHODS:
        region |= arrays[method].astype(bool)
    rows, cols = np.where(region)
    height, width = region.shape
    min_y, max_y = min(y, int(rows.min())), max(y, int(rows.max()))
    min_x, max_x = min(x, int(cols.min())), max(x, int(cols.max()))
    radius = max(92, (max(max_y - min_y, max_x - min_x) + 65) // 2)
    cy, cx = (min_y + max_y) // 2, (min_x + max_x) // 2
    return max(0, cx-radius), max(0, cy-radius), min(width, cx+radius), min(height, cy+radius)


def show_prompt(rgb, x, y, radius):
    rgb[y, max(0, x-radius):x+radius+1] = PROMPT_COLOR
    rgb[max(0, y-radius):y+radius+1, x] = PROMPT_COLOR


def case_view(arrays, x, y, method, view, show_reference, prompt, zoom):
    ref = arrays["reference"].astype(bool)
    rgb = np.repeat(arrays["ct"][:, :, None], 3, axis=2)
    if method == "mi":
        mi = arrays["mi"].astype(np.float32)
        scale = np.clip(mi / 0.15, 0, 1)
        heat = np.stack([np.interp(scale, np.linspace(0, 1, len(MI_COLORS)), MI_COLORS[:, i])
                         for i in range(3)], axis=-1)
        visible = mi >= 0.005
        rgb[visible] = (rgb[visible] * .18 + heat[visible] * .82).astype(np.uint8)
    if method in METHODS:
        pred = arrays[method].astype(bool)
        if view == "Mask fills":
            rgb[pred] = (rgb[pred] * .48 + np.asarray(PREDICTION_COLOR) * .52).astype(np.uint8)
        elif view == "Mask errors":
            for mask, color in ((pred & ref, (38, 170, 151)), (pred & ~ref, (234, 100, 84)),
                                (ref & ~pred, (95, 145, 253))):
                rgb[mask] = (rgb[mask] * .24 + np.asarray(color) * .76).astype(np.uint8)
        else:
            rgb[edge(pred)] = PREDICTION_COLOR
    if method == "reference" or (show_reference and view != "Mask errors"):
        rgb[edge(ref)] = REFERENCE_COLOR
    if prompt:
        show_prompt(rgb, x, y, radius=8 if zoom else 12)
    image = Image.fromarray(rgb)
    return image.crop(crop_bounds(arrays, x, y)) if zoom else image

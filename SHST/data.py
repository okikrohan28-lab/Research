"""
All data processing in one place:
    files (image / npz / csv)  ->  features per row  ->  normalized Xs (source), Xt (target), yt (target labels or None)

Expected folder layout:
    dataset/image/name.png    image  -> features from features.py
    dataset/npz/name.npz      npz    -> needs key "X" (N x D); optional "shape" (H, W)
    dataset/csv/name.csv      csv    -> all numeric columns are the features (remove any label column first)
Ground truth (required): one entry per target file in "ground_truth".
    image target -> mask image (e.g. dataset/image/010_mask.png)
    csv target   -> a csv file (usually the target csv itself) containing the label_column
"""
import os

import numpy as np
import pandas as pd
from PIL import Image

from features import extract_features

IMAGE_TYPES = ("jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff")


def resolve_path(path):
    """Allow names written without extension (e.g. "csv/winequality-white"): try the usual extensions."""
    if os.path.isfile(path):
        return path
    for ext in ("csv", "npz") + IMAGE_TYPES:
        for e in (ext, ext.upper()):
            if os.path.isfile(f"{path}.{e}"):
                return f"{path}.{e}"
    raise FileNotFoundError(f"File not found: {path} (also tried .csv, .npz and image extensions)")


def load_file(path, method="intensity_xy", label_column=None):
    """Read ONE file -> {"X": (N, D) features, "image": 2-D array or None, "shape": (H, W) or None, "y": labels or None}"""
    path = resolve_path(path)
    ext = path.lower().split(".")[-1]

    if ext in IMAGE_TYPES:                                    # image -> one row per pixel
        image = np.array(Image.open(path).convert("L"), dtype=float)
        X = extract_features(image, method)
        step = find_step(image.shape, len(X))                 # a method may keep only every step-th pixel (e.g. Gabor)
        small = image[::step, ::step]                         # shrink the image the same way
        return {"X": X, "image": small, "shape": small.shape, "full_shape": image.shape, "step": step, "y": None}

    if ext == "npz":                                          # npz -> features are already inside
        f = np.load(path)
        shape = tuple(int(v) for v in f["shape"]) if "shape" in f.files else None
        return {"X": np.asarray(f["X"], dtype=float), "image": None, "shape": shape, "full_shape": None, "step": 1, "y": None}

    if ext == "csv":                                          # csv -> numeric columns are the features
        df = pd.read_csv(path, sep=None, engine="python")     # sep=None: auto-detects "," or ";" (wine-quality uses ";")
        df.columns = [str(c).strip().strip('"') for c in df.columns]
        y = None
        if label_column:                                      # label column = ground truth, NOT a feature
            if label_column not in df.columns:
                raise ValueError(f"label_column '{label_column}' not in {path}. Columns: {list(df.columns)}")
            y = df[label_column].to_numpy()
            df = df.drop(columns=[label_column])
        num = df.apply(pd.to_numeric, errors="coerce")        # text columns -> NaN
        num = num.dropna(axis=1, how="all")                   # keep only numeric columns
        if num.shape[1] == 0:
            raise ValueError(f"No numeric columns found in {path}")
        X = np.nan_to_num(num.to_numpy(dtype=float))
        return {"X": X, "image": None, "shape": None, "full_shape": None, "step": 1, "y": y}

    raise ValueError(f"Unsupported file type: {path}")


def find_step(shape, n_rows):
    """Pixel spacing of the feature rows: smallest step whose kept grid has exactly n_rows pixels (1 = all pixels)."""
    h, w = shape
    for step in range(1, max(h, w) + 1):
        if len(range(0, h, step)) * len(range(0, w, step)) == n_rows:
            return step
    raise ValueError(f"{n_rows} feature rows do not fit a {h}x{w} image (expected one row per kept pixel)")


def mask_to_labels(mask, n_clusters, shape):
    """Ground-truth mask image -> one integer label per pixel (grey levels are binned if more than n_clusters)."""
    if mask.shape != tuple(shape):
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((shape[1], shape[0]), Image.NEAREST), dtype=float)
    values, labels = np.unique(mask, return_inverse=True)
    if len(values) > n_clusters:
        edges = np.linspace(mask.min(), mask.max(), n_clusters + 1)[1:-1]
        labels = np.digitize(mask.ravel(), edges)
    return np.ravel(labels).astype(int)


def load_csv_labels(path, label_column):
    """Read only the label column of a csv file."""
    if not label_column:
        raise ValueError("label_column must be set when ground_truth is a csv file")
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not in {path}. Columns: {list(df.columns)}")
    return df[label_column].to_numpy()


def normalize(Xs, Xt):
    """Paper Section 4.1: all data normalized to [0, 1] (shared min/max of source and target)."""
    both = np.vstack([Xs, Xt])
    lo, hi = both.min(axis=0), both.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (Xs - lo) / span, (Xt - lo) / span


def prepare_data(folder, source, target, ground_truth, method, n_clusters, label_column=None):
    """
    Returns Xs, Xt, yt, targets  (images may be subsampled, e.g. Gabor STEP=8: target["shape"] is the kept grid)
      Xs, Xt  : normalized feature tables (all source rows / all target rows)
      yt      : target labels read from ground_truth (one entry per target, same order):
                  image file -> mask image, csv file -> its label_column
      targets : one dict per target file (name, image, shape, start, stop) to cut Xt back into images
    """
    if not ground_truth:
        raise ValueError("ground_truth is required: give one file per target (e.g. the same name as the target)")
    if len(ground_truth) != len(target):
        raise ValueError("ground_truth needs one entry per target file")

    src = [load_file(os.path.join(folder, f), method, label_column) for f in source]
    tgt = [load_file(os.path.join(folder, f), method, label_column) for f in target]

    parts, kinds = [], set()
    for t, gt_file in zip(tgt, ground_truth):
        gt_path = resolve_path(os.path.join(folder, gt_file))
        if gt_path.lower().endswith(".csv"):                    # csv: labels come from the label column
            y = load_csv_labels(gt_path, label_column)
            if len(y) != len(t["X"]):
                raise ValueError(f"{gt_path} has {len(y)} rows but the target has {len(t['X'])}")
            parts.append(y)
            kinds.add("csv")
        else:                                                   # image: mask -> one label per pixel
            mask = np.array(Image.open(gt_path).convert("L"), dtype=float)
            if t["step"] > 1 and mask.shape == t["full_shape"]:   # keep the same pixels as the features
                mask = mask[::t["step"], ::t["step"]]
            parts.append(mask_to_labels(mask, n_clusters, t["shape"]))
            kinds.add("mask")
    if len(kinds) > 1:
        raise ValueError("ground_truth must be all csv files or all mask images")
    yt = np.concatenate(parts)
    if kinds == {"csv"}:
        yt = np.unique(yt, return_inverse=True)[1].astype(int)

    Xs = np.vstack([d["X"] for d in src])
    Xt = np.vstack([d["X"] for d in tgt])
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError(f"Source has {Xs.shape[1]} features but target has {Xt.shape[1]}")
    Xs, Xt = normalize(Xs, Xt)

    targets, start = [], 0
    for name, d in zip(target, tgt):
        targets.append({"name": os.path.splitext(os.path.basename(name))[0], "image": d["image"],
                        "shape": d["shape"], "start": start, "stop": start + len(d["X"])})
        start += len(d["X"])
    return Xs, Xt, yt, targets

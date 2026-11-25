"""Scores (NMI, ARI, ...) and plots."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score,
                             v_measure_score, silhouette_score)
from sklearn.metrics.cluster import contingency_matrix


def make_reference(X, n_clusters, seed=0):
    """
    No ground truth available -> build a pseudo reference labelling with K-Means on the target features.
    NMI / ARI then measure agreement with this reference (not true accuracy).
    """
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)


def evaluate(y_ref, y_pred, X=None):
    """Same keys for every algorithm: nmi, ari, accuracy, v_measure (vs. y_ref) and silhouette (needs no labels)."""
    cm = contingency_matrix(y_ref, y_pred)
    rows, cols = linear_sum_assignment(-cm)                  # best one-to-one cluster <-> class matching
    out = {
        "nmi": float(normalized_mutual_info_score(y_ref, y_pred)),
        "ari": float(adjusted_rand_score(y_ref, y_pred)),
        "accuracy": float(cm[rows, cols].sum() / cm.sum()),
        "v_measure": float(v_measure_score(y_ref, y_pred)),
        "silhouette": float("nan"),
    }
    if X is not None and 2 <= len(np.unique(y_pred)) < len(y_pred):
        out["silhouette"] = float(silhouette_score(X, y_pred, sample_size=min(5000, len(y_pred)), random_state=0))
    return out


def align_labels(y_ref, y_pred):
    """Recolor clusters to match the reference label ids (only for drawing, scores are not affected)."""
    cm = contingency_matrix(y_ref, y_pred)
    rows, cols = linear_sum_assignment(-cm)
    ref_ids, pred_ids = np.unique(y_ref), np.unique(y_pred)
    mapping = {pred_ids[c]: ref_ids[r] for r, c in zip(rows, cols)}
    next_id = int(ref_ids.max()) + 1
    out = np.empty_like(y_pred)
    for p in pred_ids:
        if p not in mapping:
            mapping[p] = next_id
            next_id += 1
        out[y_pred == p] = mapping[p]
    return out


def save_image(path, label_image):
    """Save a label image as grey-scale PNG (labels stretched to 0..255)."""
    img = np.asarray(label_image, dtype=float)
    span = img.max() - img.min()
    Image.fromarray(((img - img.min()) / span * 255 if span > 0 else np.zeros_like(img)).astype(np.uint8)).save(path)


def metric_title(name, m):
    """Panel title with ALL metrics."""
    return (f"{name}\n"
            f"NMI={m['nmi']:.3f}  ARI={m['ari']:.3f}\n"
            f"Sil={m['silhouette']:.3f}  Time={m['time_s']:.2f}s")


def plot_comparison(original, segmentations, scores):
    """One row: Original | every method, each titled with NMI, ARI, Silhouette, Time.
    scores: {method: full result dict with nmi, ari, silhouette, time_s}"""
    panels = []
    if original is not None:
        panels.append(("Original", original, "gray"))
    for name, seg in segmentations.items():                  # every method gets its own panel
        panels.append((metric_title(name, scores[name]), seg, "tab10"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5.5))
    for ax, (title, img, cmap) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_tabular(X, labels_by_method, max_points=5000):
    """csv / npz without an image grid: 2-D PCA scatter, one panel per method."""
    from sklearn.decomposition import PCA
    idx = np.random.default_rng(0).choice(len(X), min(max_points, len(X)), replace=False)
    Z = PCA(n_components=min(2, X.shape[1])).fit_transform(X[idx])
    if Z.shape[1] == 1:
        Z = np.hstack([Z, np.zeros_like(Z)])
    fig, axes = plt.subplots(1, len(labels_by_method), figsize=(4 * len(labels_by_method), 4))
    for ax, (name, labels) in zip(np.atleast_1d(axes), labels_by_method.items()):
        ax.scatter(Z[:, 0], Z[:, 1], c=np.asarray(labels)[idx], s=4, cmap="tab10")
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    return fig

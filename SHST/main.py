"""
Run:  python main.py      (open the terminal inside this folder)
Edit the SETTINGS below, nothing else is needed.

Flow:  for every dataset (Image1, Image2, ...):
         files -> features -> normalize (data.py)
         every algorithm is run N_RUNS times (different random seeds) on the same (Xs, Xt)
         NMI / ARI are scored against the ground-truth mask (evaluation.py)
       -> paper-style tables "mean ± std" printed + one comparison figure per dataset
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import prepare_data
from evaluation import evaluate, align_labels, plot_comparison, plot_tabular
from algorithms.fcm import fcm
from algorithms.t1_kt_fcm import t1_kt_fcm
from algorithms.tsc import tsc
from algorithms.shst import shst_fcm

# ============================== SETTINGS ==============================
DATASET_FOLDER = "dataset"

# One entry per row of the paper table. ground_truth (required) = one file per target file, same order:
#   image target -> mask image;  csv target -> csv with the label column (just reuse the target name).
DATASETS = {
    "Image": {"source": ["image/000.png"],
               "target": ["image/001.png"],
            
               "ground_truth": ["image/001.png"]}                         # e.g. ["image/010_mask.png"]

    # --- CSV / tabular example (file extension is optional). Uncomment to use: ---
    #"Tabular": {"source": ["csv/winequality-red.csv"],
              #"target": ["csv/winequality-white.csv"],
              #"ground_truth": ["csv/winequality-white.csv"],
              #"label_column": "quality",                 # optional: column used as true labels (removed from features)
              #"n_clusters": 7},                          # optional: per-dataset number of clusters (default N_CLUSTERS)
}
N_RUNS = 10                                   # repetitions per algorithm -> mean ± std

FEATURES = "intensity_xy"                     # images only: "intensity_xy" | "gabor" (see features.py)
N_CLUSTERS = 2                                # bone, water, soft tissue
M = 2.0                                       # fuzzy index (paper grid 1.1, 1.2, 1.5, 1.7, 2, 2.5)
LAMBDA = 1000                                 # lambda of SHST (paper grid 1e-4 ... 1e4)
MAX_ITER = 100
OUTPUT_FOLDER = "outputs"
SHOW_PLOTS = True

ALGORITHMS = {                                # k = number of clusters of the current dataset
    #"FCM":       lambda Xs, Xt, k: fcm(Xs, Xt, k, M, MAX_ITER),
    #"TSC":       lambda Xs, Xt, k: tsc(Xs, Xt, k),
    #"T1-KT-FCM": lambda Xs, Xt, k: t1_kt_fcm(Xs, Xt, k, M, LAMBDA, MAX_ITER),
    "SHST-FCM":  lambda Xs, Xt, k: shst_fcm(Xs, Xt, k, M, LAMBDA, MAX_ITER),
}
# ======================================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
tables = {k: {} for k in ("nmi", "ari", "silhouette", "time_s")}     # metric -> {dataset: {algorithm: "mean±std"}}
figures = []

for ds_name, cfg in DATASETS.items():
    # 1) data
    k = cfg.get("n_clusters", N_CLUSTERS)
    label_col = cfg.get("label_column")
    Xs, Xt, yt, targets = prepare_data(DATASET_FOLDER, cfg["source"], cfg["target"],
                                       cfg["ground_truth"], FEATURES, k, label_column=label_col)
    reference = f"label column '{label_col}'" if label_col else "ground-truth masks"
    if label_col and "n_clusters" not in cfg:
        k = len(np.unique(yt))
    print(f"\n=== {ds_name}: source {Xs.shape}, target {Xt.shape}, reference: {reference}")

    # 2) run every algorithm N_RUNS times
    runs, first_labels = {}, {}
    for name, run in ALGORITHMS.items():
        runs[name] = []
        for r in range(N_RUNS):
            np.random.seed(r)                                         # different random start each run
            start = time.perf_counter()
            labels = run(Xs, Xt, k)
            elapsed = time.perf_counter() - start
            runs[name].append({"time_s": elapsed, **evaluate(yt, labels, Xt)})
            if r == 0:
                first_labels[name] = labels
        print(f"  {name:10s} NMI {np.mean([x['nmi'] for x in runs[name]]):.4f}  "
              f"ARI {np.mean([x['ari'] for x in runs[name]]):.4f}")

    for metric in tables:
        tables[metric][ds_name] = {n: f"{np.nanmean([x[metric] for x in rs]):.4f}±{np.nanstd([x[metric] for x in rs]):.4f}"
                                   for n, rs in runs.items()}

    # 3) comparison figure (titles show the mean over runs, picture = run 0)
    mean_scores = {n: {m: float(np.nanmean([x[m] for x in rs])) for m in ("nmi", "ari", "silhouette", "time_s")}
                   for n, rs in runs.items()}
    if all(t["shape"] is not None for t in targets):
        first = targets[0]
        segs = {n: align_labels(yt[first["start"]:first["stop"]], lab[first["start"]:first["stop"]]).reshape(first["shape"])
                for n, lab in first_labels.items()}
        fig = plot_comparison(first["image"], segs, mean_scores)
    else:
        fig = plot_tabular(Xt, first_labels)
    fig.savefig(os.path.join(OUTPUT_FOLDER, f"comparison_{ds_name}.png"), dpi=150, bbox_inches="tight")
    figures.append(fig)

# 4) paper-style tables: rows = datasets, columns = algorithms, cells = mean±std over N_RUNS
titles = {"nmi": "NMI", "ari": "ARI", "silhouette": "Silhouette", "time_s": "Time (s)"}
for metric, title in titles.items():
    print(f"\nAverage (±Standard Deviation) of {title} in the target domain over {N_RUNS} runs")
    print(pd.DataFrame(tables[metric]).T.rename_axis("Dataset").to_string())

if SHOW_PLOTS:
    plt.show()

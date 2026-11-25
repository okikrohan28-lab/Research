"""
TECM on the head-CT pair, with noise added to the TARGET features.
Needs tecm.py in the same folder (uses its tecm_core / hard_labels / cluster_accuracy).
Run:  python headct_noise.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tecm import tecm_core, hard_labels, cluster_accuracy

# ---------------- settings ----------------
SRC, TGT = "headct_src.npz", "headct_trg.npz"
C = 5                           # tissue classes (labels 0..4)
LAMBDAS = [1, 10, 100, 1000]    # grid of transfer strengths to search
BETA = 1.5
NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0]   # noise std as a multiple of each feature's own std
SEED = 0

# ---------------- load ----------------
src, tgt = np.load(SRC), np.load(TGT)
Xs = src["features"].astype(float)
Xt_clean = tgt["features"].astype(float)
yt = tgt["labels"]

# image shape from the sampling grid (91 x 83 for the target)
H, W = len(np.unique(tgt["row"])), len(np.unique(tgt["col"]))

def add_noise(X, level, seed=SEED):
    """Gaussian noise, scaled per feature: sigma_k = level * std(feature k)."""
    rng = np.random.default_rng(seed)
    return X + rng.normal(size=X.shape) * level * X.std(0)

def run(Xs, Xt, lam):
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-8          # standardize both with SOURCE stats
    Xs_, Xt_ = (Xs - mu) / sd, (Xt - mu) / sd
    base = tecm_core(Xs_, C, lam=0, beta=BETA)      # ECM on source -> barycenters
    res = tecm_core(Xt_, C, Vsrc_bary=base["bary"], lam=lam, beta=BETA)
    res["labels"] = hard_labels(res)
    res["acc"] = cluster_accuracy(yt, res["labels"])
    return res

# ---------------- run (grid search over lambda) ----------------
fig, axes = plt.subplots(len(NOISE_LEVELS), 4, figsize=(14, 3.6 * len(NOISE_LEVELS)))
print(f"{'noise':>6} | {'ECM':>7} | " + " | ".join(f"lam={l:<5}" for l in LAMBDAS) + " | best lam")
for i, lvl in enumerate(NOISE_LEVELS):
    Xt = add_noise(Xt_clean, lvl)
    np.savez(f"headct_trg_noise{lvl}.npz", **{**dict(tgt), "features": Xt.astype(np.float32)})

    ecm = run(Xs, Xt, 0)
    results = {lam: run(Xs, Xt, lam) for lam in LAMBDAS}       # grid search
    best_lam = max(results, key=lambda l: results[l]["acc"])   # pick lam with highest accuracy
    tecm = results[best_lam]
    print(f"{lvl:>6} | {ecm['acc']:>7.4f} | " +
          " | ".join(f"{results[l]['acc']:<9.4f}" for l in LAMBDAS) + f" | {best_lam}")

    amb = (tecm["M"][:, tecm["card"] > 1].sum(1) > 0.5).reshape(H, W)
    panels = [(yt.reshape(H, W), f"ground truth (noise={lvl})"),
              (ecm["labels"].reshape(H, W), f"ECM  acc={ecm['acc']:.3f}"),
              (tecm["labels"].reshape(H, W), f"TECM best lam={best_lam}  acc={tecm['acc']:.3f}"),
              (amb, "TECM ambiguous pixels")]
    for ax, (img, title) in zip(axes[i], panels):
        ax.imshow(img, cmap="gray" if title.endswith("pixels") else "Blues")
        ax.set_title(title); ax.axis("off")

plt.tight_layout()
plt.savefig("tecm_headct_noise.png", dpi=150)
print("saved tecm_headct_noise.png")

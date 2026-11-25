"""
TECM: Transfer learning-based Evidential C-Means (Jiao et al.)
Simple, commented implementation. Only needs numpy (scipy used for accuracy only).

Flow:
  1. Load source .npz and target .npz (Gabor features, optional labels)
  2. Run ECM on the SOURCE  -> learn "barycenters" of all focal sets (the transferred knowledge)
  3. Run TECM on the TARGET -> ECM objective + lam * (target barycenters close to source barycenters)
  4. Return credal partition M (mass on every subset of clusters) + hard labels

Paper equation numbers are given in comments, e.g. (Eq. 12).
"""
import numpy as np
import pandas as pd
from itertools import combinations


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def make_focal_sets(c, max_card=2):
    """
    Build all non-empty focal sets A_j (subsets of the c clusters).
    Returns S  (c x F): S[k, j] = 1 if cluster k is in focal set j   (Eq. 2)
            card (F,) : cardinality c_j of each focal set
    max_card=2 -> only singletons and pairs (much cheaper: O(c^2) instead of O(2^c)),
    exactly the speed-up suggested in the paper. Use max_card=c for the full 2^c - 1 sets.
    """
    cols = []
    for size in range(1, min(max_card, c) + 1):
        for comb in combinations(range(c), size):
            col = np.zeros(c)
            col[list(comb)] = 1
            cols.append(col)
    S = np.array(cols).T
    return S, S.sum(0)


def sqdist(A, B):
    """Squared Euclidean distance between every row of A and every row of B."""
    return np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T, 0)


def barycenters(V, S, card):
    """Barycenter of every focal set = mean of the centers of its clusters (Eq. 2)."""
    return (S.T @ V) / card[:, None]


# ----------------------------------------------------------------------------
# Core algorithm (ECM when Vsrc_bary is None, TECM otherwise)
# ----------------------------------------------------------------------------
def tecm_core(X, c, Vsrc_bary=None, lam=0.0, alpha=1.0, beta=1.5, gamma=2.0,
              delta=None, max_card=2, max_iter=100, eps=1e-3, seed=0):
    """
    X          : (n, p) data to cluster
    c          : number of clusters
    Vsrc_bary  : (Fs, p) source barycenters (None -> plain ECM)
    lam        : transfer strength. 0 = plain ECM, too large = "negative transfer"
    alpha      : penalizes large focal sets (higher -> fewer ambiguous objects)
    beta       : fuzziness of the masses (>1). The paper uses 2, but with high-dimensional
                 features (Gabor) beta=2 makes all centers collapse together; 1.5 (or
                 1.2) is much safer. If clusters look merged, lower beta first.
    gamma      : fuzziness of the source<->target association (>1)
    delta      : distance to the "empty set" (outlier threshold). None = auto (see below)
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    S, card = make_focal_sets(c, max_card)           # focal sets of the TARGET
    transfer = Vsrc_bary is not None and lam > 0

    # delta must match the scale of your distances. The paper's delta=10 only works for
    # their data; with high-dim Gabor features distances are large, so we scale it:
    # delta^2 = 4 * average squared distance of the points to the data mean.
    if delta is None:
        delta = 2 * np.sqrt(np.mean(((X - X.mean(0))**2).sum(1)))

    V = kmeans_init(X, c, rng)                       # init: k-means centers
    J_old = np.inf

    for it in range(max_iter):
        Vbar = barycenters(V, S, card)               # (F, p)
        d = sqdist(X, Vbar) + 1e-10                  # d_ij, (n, F)

        # ---- Step 1: credal partition M (Eq. 12, 13) ----
        w = (card**alpha * d) ** (-1.0 / (beta - 1))
        M = w / (w.sum(1, keepdims=True) + delta ** (-2.0 / (beta - 1)))
        M_empty = 1 - M.sum(1)                       # mass on the empty set = outlier-ness
        Mb = M**beta

        # ---- Step 2: association matrix R between source and target barycenters (Eq. 18)
        if transfer:
            dsv = sqdist(Vsrc_bary, Vbar) + 1e-10    # (Fs, F)
            rw = (card**alpha * dsv) ** (-1.0 / (gamma - 1))
            R = rw / rw.sum(1, keepdims=True)        # each source barycenter distributes mass 1
            Rg = R**gamma

        # ---- Step 3: update cluster centers V by solving a linear system (Eq. 21-25) ----
        ca1, ca2 = card**(alpha - 1), card**(alpha - 2)
        B = S @ (ca1[:, None] * Mb.T) @ X                       # B1
        H = S @ np.diag(ca2 * Mb.sum(0)) @ S.T                  # H1
        if transfer:
            B += lam * (S @ (ca1[:, None] * Rg.T) @ Vsrc_bary)  # + lam * B2
            H += lam * (S @ np.diag(ca2 * Rg.sum(0)) @ S.T)     # + lam * H2
        V = np.linalg.lstsq(H, B, rcond=None)[0]                # (H1+lam*H2) V = B1+lam*B2

        # ---- Step 4: objective (Eq. 4) and stopping test ----
        Vbar = barycenters(V, S, card)
        d = sqdist(X, Vbar)
        J = (card**alpha * Mb * d).sum() + (delta**2 * M_empty**beta).sum()
        if transfer:
            J += lam * (card**alpha * Rg * sqdist(Vsrc_bary, Vbar)).sum()
        if abs(J_old - J) < eps:
            break
        J_old = J

    return dict(M=M, M_empty=M_empty, V=V, S=S, card=card, J=J, n_iter=it + 1,
                bary=barycenters(V, S, card))


def kmeans_init(X, c, rng, n_iter=20):
    """Tiny k-means (k-means++ seeding + Lloyd) used to initialise the centers V.
    A good start matters: ECM/FCM started from random points can collapse all centers
    onto the data mean when the features are high-dimensional (e.g. Gabor)."""
    V = X[rng.integers(len(X))][None]
    for _ in range(c - 1):                                   # k-means++ seeding
        dmin = sqdist(X, V).min(1)
        V = np.vstack([V, X[rng.choice(len(X), p=dmin / dmin.sum())]])
    for _ in range(n_iter):                                  # Lloyd iterations
        lab = sqdist(X, V).argmin(1)
        V = np.array([X[lab == k].mean(0) if (lab == k).any() else V[k] for k in range(c)])
    return V


def hard_labels(res):
    """Credal partition -> hard labels via pignistic probability (split each mass equally)."""
    betp = (res["M"] / res["card"]) @ res["S"].T                 # (n, c)
    betp /= np.maximum(1 - res["M_empty"], 1e-12)[:, None]
    return betp.argmax(1)


def cluster_accuracy(y_true, y_pred):
    """Clustering accuracy: best one-to-one match between clusters and classes (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    classes, y_true = np.unique(y_true, return_inverse=True)
    conf = np.zeros((y_pred.max() + 1, len(classes)))
    for p_, t_ in zip(y_pred, y_true):
        conf[p_, t_] += 1
    r, c_ = linear_sum_assignment(-conf)
    return conf[r, c_].sum() / len(y_true)



def run_tecm(src_npz, tgt_npz, feat_key="features", label_key="labels",
             c_src=4, c_tgt=3, lam=10.0, alpha=1.0, beta=1.5, gamma=2.0,
             delta=None, max_card=2, standardize=True, seed=0, verbose=True):
    """
    src_npz / tgt_npz : paths to .npz files holding Gabor features (and optionally labels)
    feat_key          : array name for features inside the npz (any shape -> flattened to (n, p))
    label_key         : array name for labels. Optional: only used to guess the number of
                        clusters and to print accuracy.
    c_src / c_tgt     : number of clusters (None -> number of unique labels in that file).
                        They may be different - that is the point of TECM.
    Returns dict with: M (n x F credal partition), focal_sets (S), labels (hard labels),
                       M_empty (outlier mass), accuracy (if target labels exist).
    """
    # ---- load ----
    src, tgt = np.load(src_npz), np.load(tgt_npz)
    Xs = src[feat_key].reshape(len(src[feat_key]), -1).astype(float)
    Xt = tgt[feat_key].reshape(len(tgt[feat_key]), -1).astype(float)
    ys = src[label_key].ravel() if label_key in src.files else None
    yt = tgt[label_key].ravel() if label_key in tgt.files else None
    c_src = c_src or len(np.unique(ys))
    c_tgt = c_tgt or len(np.unique(yt))

    # ---- standardize (use SOURCE statistics for both so they share one feature space) ----
    if standardize:
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-8
        Xs, Xt = (Xs - mu) / sd, (Xt - mu) / sd

    # ---- 1) ECM on source -> knowledge = barycenters of all source focal sets ----
    src_res = tecm_core(Xs, c_src, lam=0, alpha=alpha, beta=beta, delta=delta,
                        max_card=max_card, seed=seed)
    # ---- 2) TECM on target, guided by those barycenters ----
    res = tecm_core(Xt, c_tgt, Vsrc_bary=src_res["bary"], lam=lam, alpha=alpha, beta=beta,
                    gamma=gamma, delta=delta, max_card=max_card, seed=seed)

    res["labels"] = hard_labels(res)
    res["X"] = Xt
    res["focal_sets"] = res.pop("S")
    if yt is not None:
        res["accuracy"] = cluster_accuracy(yt, res["labels"])
    if verbose:
        print(f"source: {Xs.shape}, c={c_src} | target: {Xt.shape}, c={c_tgt} | "
              f"iterations: {res['n_iter']}")
        if yt is not None:
            print(f"TECM accuracy (lam={lam}): {res['accuracy']:.4f}")
    return res


"""
if __name__ == "__main__":
    SRC, TGT = "headct_src.npz", "headct_trg.npz"   # <-- your files
    KW = dict(feat_key="features", label_key="labels")  # <-- your array names

    base = run_tecm(SRC, TGT, lam=0, verbose=False, **KW)
    print("ECM  accuracy:", base.get("accuracy", "no labels"))

    for lam in [0.1, 0.5, 1, 5, 10, 50, 100, 300, 500, 1000]:
        r = run_tecm(SRC, TGT, lam=lam, verbose=False, **KW)
        print(f"TECM lam={lam:<6} accuracy:", r.get("accuracy", "no labels"))

"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    SRC, TGT = "src.npz", "trg3.npz"
    KW = dict(feat_key="features", label_key="labels")

    lams = [0, 1, 10, 100, 1000]          # 0 = plain ECM (no transfer)
    fig, axes = plt.subplots(2, len(lams), figsize=(3.5 * len(lams), 7))

    for j, lam in enumerate(lams):
        r = run_tecm(SRC, TGT, lam=lam, verbose=False, **KW)   # use the loop's lam
        n = len(r["labels"]); H = W = int(np.sqrt(n))

        seg = r["labels"].reshape(H, W)
        amb = (r["M"][:, r["card"] > 1].sum(1) > 0.5).reshape(H, W)

        title = "ECM" if lam == 0 else f"TECM lam={lam}"
        if "accuracy" in r:
            title += f"\nacc={r['accuracy']:.3f}"

        axes[0, j].imshow(seg, cmap="Blues")
        axes[0, j].set_title(title)
        axes[1, j].imshow(amb, cmap="vanimo")
        axes[1, j].set_title("ambiguous pixels")

    for a in axes.ravel():
        a.axis("off")

    plt.tight_layout()
    plt.savefig("tecm_seg.png", dpi=150)
    plt.show()


"""
SHST-FCM (Xia et al., 2020): shared hidden space transfer fuzzy c-means.

Input : Xs (source rows), Xt (target rows)   -> never loads, saves or plots anything
Output: cluster label of every target row

Conventions: X (N, d) samples as rows | H (r, d) projection with orthonormal rows (H H^T = I)
             Z = X H^T (N, r) hidden-space samples | V (C, r) centers | U (N, C) memberships
             Omega (d, d) MMD matrix.  The paper writes H x_i with column vectors; here that is X @ H.T.
"""
import numpy as np

EPS = 1e-10


# ------------------------------------------------------------------ transfer part (MMD)
def compute_omega(Xs, Xt):
    """
    Paper Eq. 3-4 (symmetric form under Eq. 17).  MMD in the hidden space is
        || mean(H xt) - mean(H xs) ||^2 = tr(H Omega H^T),  Omega = (mean_t - mean_s)(mean_t - mean_s)^T
    (the double sums of Eq. 3 collapse to the two sample means).
    """
    diff = Xt.mean(axis=0) - Xs.mean(axis=0)
    return np.outer(diff, diff)


def init_H(X, r, method, rng):
    """
    Start value of H (r x d) with orthonormal rows so H H^T = I (constraint of Eq. 5 / 6).
    "pca": top-r principal directions of the data.  "qr": random orthonormal rows via QR decomposition.
    """
    d = X.shape[1]
    if method == "pca":
        Xc = X - X.mean(axis=0)
        _, vecs = np.linalg.eigh(Xc.T @ Xc / len(X))          # eigenvalues in ascending order
        return vecs[:, ::-1][:, :r].T
    Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    return Q.T


def project(X, H):
    """Paper Section 3.1: a sample in the shared hidden space is H x_i  (here Z = X @ H.T)."""
    return X @ H.T


# ------------------------------------------------------------------ fuzzy clustering part
def sq_dist(Z, V):
    """||z_i - v_j||^2 for every sample i and center j -> (N, C)."""
    return np.maximum((Z ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None, :] - 2.0 * Z @ V.T, 0.0)


def update_centers(X, U, m, H):
    """Paper Eq. 10: v_j = sum_i mu_ij^m (H x_i) / sum_i mu_ij^m."""
    W = U ** m
    return (W.T @ project(X, H)) / (W.sum(axis=0)[:, None] + EPS)


def update_membership(X, V, m, H):
    """
    Paper Eq. 14: mu_ij = (1/||H x_i - v_j||^2)^(1/(m-1)) / sum_k (1/||H x_i - v_k||^2)^(1/(m-1)).
    Written in log form (same value, no overflow).
    """
    d2 = np.maximum(sq_dist(project(X, H), V), EPS)
    log_w = -np.log(d2) / (m - 1.0)
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def objective(X, U, V, H, Omega, lam, m):
    """Paper Eq. 6: J = sum_ij mu^m ||H x_i - v_j||^2 + lambda * tr(H Omega H^T).  Returns (J, fcm part, mmd part)."""
    fcm_loss = float(((U ** m) * sq_dist(project(X, H), V)).sum())
    mmd_loss = float(np.sum((H @ Omega) * H))                 # tr(H Omega H^T)
    return fcm_loss + lam * mmd_loss, fcm_loss, mmd_loss


# ------------------------------------------------------------------ update of H (gradient on the Grassmann manifold)
def gradient_H(X, U, V, H, Omega, lam, m):
    """
    Paper Eq. 18: dG/dH, written without the paper's huge diagonal matrices (same result):
        dG/dH = 2 (H A - B) + 2 lambda H Omega
        A = sum_i s_i x_i x_i^T  with s_i = sum_j mu_ij^m  (d x d)     B = sum_j v_j (sum_i mu_ij^m x_i)^T  (r x d)
    """
    W = U ** m
    s = W.sum(axis=1)
    A = (X * s[:, None]).T @ X
    B = V.T @ (W.T @ X)
    return 2.0 * (H @ A - B) + 2.0 * lam * (H @ Omega)


def grassmann_gradient(H, grad):
    """
    Paper Eq. 19 / 26: Riemannian gradient (I - H H^T) dG/dH.  With H stored as rows (r x d, H H^T = I)
    the same projection is  grad (I - H^T H)  (the paper's form would be exactly zero for row-stored H).
    """
    return grad - (grad @ H.T) @ H


def step_size(X, U, V, H, Omega, lam, m, D, eta_max):
    """
    Paper Eq. 21-25: exact line search of G(H - eta D) (a quadratic in eta), then eta = min(eta, 1e-7).
        eta = [sum_ij mu^m (H x_i - v_j).(D x_i) + lam tr(H Omega D^T)] / [sum_i s_i ||D x_i||^2 + lam tr(D Omega D^T)]
    """
    W = U ** m
    s = W.sum(axis=1)
    Z = project(X, H)
    DX = X @ D.T
    num = float(((s[:, None] * Z - W @ V) * DX).sum() + lam * np.sum((H @ Omega) * D))
    den = float((s[:, None] * DX ** 2).sum() + lam * np.sum((D @ Omega) * D))
    if not np.isfinite(num) or not np.isfinite(den) or den <= EPS:
        return eta_max
    return min(max(num / den, 0.0), eta_max)                  # Eq. 25


def update_H(H, D, eta):
    """Paper Eq. 26: H <- H - eta * D, then QR keeps the rows orthonormal (constraint H H^T = I of Eq. 5 / 6)."""
    Q, R = np.linalg.qr((H - eta * D).T)
    signs = np.where(np.diag(R) < 0, -1.0, 1.0)               # keep the original orientation
    return (Q * signs).T


# ------------------------------------------------------------------ Algorithm 1 (the whole method)
def shst_fcm(Xs, Xt, n_clusters=5, m=2.0, lam=100.0, max_iter=100, latent_dim=None,
             eta_max=1e-7, h_init="pca", tol=1e-6, seed=0, verbose=False):
    """
    Paper Algorithm 1.  lam = lambda of Eq. 6 (paper grid 1e-4 ... 1e4), m = fuzzy index (1.1 ... 2.5),
    latent_dim = r of the hidden space (None = about 0.75 d, must be smaller than d).
    """
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError("Source and target must have the same number of features (Section 3.1).")
    rng = np.random.default_rng(seed)
    d = Xs.shape[1]
    r = latent_dim or min(max(1, round(0.75 * d)), max(1, d - 1))

    X = np.vstack([Xs, Xt])                                   # X_{s+t}: both domains are clustered together (Eq. 6)
    Omega = compute_omega(Xs, Xt)                             # Eq. 3-4
    H = init_H(X, r, h_init, rng)
    U = rng.random((len(X), n_clusters))                      # random fuzzy partition, rows sum to 1
    U /= U.sum(axis=1, keepdims=True)

    prev_loss = np.inf
    for it in range(max_iter):
        V = update_centers(X, U, m, H)                                            # Eq. 10
        U = update_membership(X, V, m, H)                                         # Eq. 14
        D = grassmann_gradient(H, gradient_H(X, U, V, H, Omega, lam, m))          # Eq. 18, 26
        eta = step_size(X, U, V, H, Omega, lam, m, D, eta_max)                    # Eq. 24-25
        H = update_H(H, D, eta)                                                   # Eq. 26
        loss, fcm_loss, mmd_loss = objective(X, U, V, H, Omega, lam, m)           # Eq. 6
        if verbose:
            print(f"  SHST iter {it + 1:3d}: J={loss:.4f} (fcm {fcm_loss:.4f}, mmd {mmd_loss:.6f}), eta={eta:.2e}")
        if abs(prev_loss - loss) <= tol * max(1.0, abs(loss)):                    # stop when G(t) - G(t-1) <= delta
            break
        prev_loss = loss

    V = update_centers(X, U, m, H)                            # final consistent V, U for the last H
    U = update_membership(X, V, m, H)
    return np.argmax(U, axis=1)[len(Xs):]                     # hard labels of the target rows

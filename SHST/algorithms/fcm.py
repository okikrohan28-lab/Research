"""
Baseline: plain FCM (paper Eq. 1).
Paper Section 4.1: FCM "simply combines the samples in the source and target domains together".
Input: Xs, Xt.  Output: cluster label of every target row.
"""
import numpy as np

EPS = 1e-10


def sq_dist(X, V):
    """||x_i - v_j||^2 for every sample i and center j -> (N, C)."""
    return np.maximum((X ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None, :] - 2.0 * X @ V.T, 0.0)


def memberships(X, V, m):
    """FCM membership update: mu_ij proportional to (1/d_ij^2)^(1/(m-1)); every row sums to 1."""
    log_w = -np.log(np.maximum(sq_dist(X, V), EPS)) / (m - 1.0)     # log form avoids overflow
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def fcm(Xs, Xt, n_clusters=3, m=2.0, max_iter=100, tol=1e-6, seed=0):
    X = np.vstack([Xs, Xt])                                   # no transfer: just stack both domains
    rng = np.random.default_rng(seed)
    U = rng.random((len(X), n_clusters))
    U /= U.sum(axis=1, keepdims=True)
    prev = np.inf
    for _ in range(max_iter):
        W = U ** m
        V = (W.T @ X) / (W.sum(axis=0)[:, None] + EPS)        # centers
        U = memberships(X, V, m)                              # memberships
        J = ((U ** m) * sq_dist(X, V)).sum()                  # Eq. 1
        if abs(prev - J) <= tol * max(1.0, abs(J)):
            break
        prev = J
    return np.argmax(U, axis=1)[len(Xs):]                     # keep the target rows

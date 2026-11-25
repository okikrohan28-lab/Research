"""
Baseline: T1-KT-FCM (type-I knowledge-transfer-oriented c-means), SIMPLIFIED version.
Paper Section 4.2: it "uses the cluster centers from the source domain as the transfer knowledge".
  1) FCM on the source -> source centers Vs
  2) cluster the target with  J = sum mu^m ||x - v||^2 + lam * sum_j ||v_j - Vs_j||^2
     closed form center:  v_j = (sum_i mu_ij^m x_i + lam * Vs_j) / (sum_i mu_ij^m + lam)
Replace with the exact T1-KT-FCM formulas if you need the original method.
Input: Xs, Xt.  Output: cluster label of every target row.
"""
import numpy as np

EPS = 1e-10


def sq_dist(X, V):
    return np.maximum((X ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None, :] - 2.0 * X @ V.T, 0.0)


def memberships(X, V, m):
    log_w = -np.log(np.maximum(sq_dist(X, V), EPS)) / (m - 1.0)
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def t1_kt_fcm(Xs, Xt, n_clusters=3, m=2.0, lam=100.0, max_iter=100, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)

    # step 1: knowledge from the source = its cluster centers
    U = rng.random((len(Xs), n_clusters))
    U /= U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        W = U ** m
        Vs = (W.T @ Xs) / (W.sum(axis=0)[:, None] + EPS)
        U = memberships(Xs, Vs, m)

    # step 2: cluster the target, centers are pulled towards the source centers
    V, prev = Vs.copy(), np.inf
    for _ in range(max_iter):
        U = memberships(Xt, V, m)
        W = U ** m
        V = (W.T @ Xt + lam * Vs) / (W.sum(axis=0)[:, None] + lam)
        J = (W * sq_dist(Xt, V)).sum() + lam * ((V - Vs) ** 2).sum()
        if abs(prev - J) <= tol * max(1.0, abs(J)):
            break
        prev = J
    return np.argmax(memberships(Xt, V, m), axis=1)

"""
Baseline: TSC (transfer spectral clustering), SIMPLIFIED stand-in.
One k-nearest-neighbour graph is built over a random subset of source + target samples, so the
two domains are linked through the same graph; spectral clustering is run on it, then every
target row gets the label of its nearest subset points.  (Subsampling keeps it fast.)
Input: Xs, Xt.  Output: cluster label of every target row.
"""
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier


def tsc(Xs, Xt, n_clusters=3, n_neighbors=10, max_points=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice(len(Xs), min(max_points, len(Xs)), replace=False)
    t = rng.choice(len(Xt), min(max_points, len(Xt)), replace=False)
    Xa = np.vstack([Xs[s], Xt[t]])                            # shared graph over both domains
    sub_labels = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors",
                                    n_neighbors=n_neighbors, assign_labels="kmeans",
                                    random_state=seed).fit_predict(Xa)
    return KNeighborsClassifier(n_neighbors=5).fit(Xa, sub_labels).predict(Xt)

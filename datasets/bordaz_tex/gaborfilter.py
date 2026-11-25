"""
gabor_npz.py: Gabor feature extraction in the style of the TECM paper (Kyrki et al. [64]).
Image -> per-pixel Gabor magnitude features -> .npz readable by tecm.py
Needs: numpy, opencv-python
"""
import numpy as np
import cv2

# ---- Gabor bank (MUST be identical for source and target) ----
N_FREQ   = 4                      # number of frequency channels
N_ORIENT = 6                      # number of orientations (0, 30, ..., 150 deg)
F_MAX    = 0.25                   # highest frequency (cycles/pixel at WORK resolution)
FREQS    = [F_MAX / (np.sqrt(2) ** k) for k in range(N_FREQ)]   # geometric spacing
THETAS   = [np.pi * k / N_ORIENT for k in range(N_ORIENT)]

OUT  = 90                         # final grid, as in the paper (n = 8100 pixels)
WORK = 360                        # filtering resolution (multiple of OUT)
SMOOTH = True                     # Gaussian smoothing of magnitudes, scaled to each wavelength


def gabor_features(path, noise_sigma=0.0, seed=0):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"cannot read {path}"
    img = cv2.resize(img, (WORK, WORK), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if noise_sigma > 0:           # the paper adds Gaussian noise (sigma 0.05 / 0.1) to target images
        img += np.random.default_rng(seed).normal(0, noise_sigma, img.shape).astype(np.float32)
    img -= img.mean()

    feats = []
    for f in FREQS:
        lam = 1.0 / f                              # wavelength in pixels
        sigma = 0.56 * lam                         # ~1 octave bandwidth
        ks = int(6 * sigma) | 1                    # odd kernel size
        for th in THETAS:
            ke = cv2.getGaborKernel((ks, ks), sigma, th, lam, 1.0, 0)
            ko = cv2.getGaborKernel((ks, ks), sigma, th, lam, 1.0, np.pi / 2)
            ke -= ke.mean(); ko -= ko.mean()       # zero-mean -> ignores brightness
            re = cv2.filter2D(img, cv2.CV_32F, ke)
            im = cv2.filter2D(img, cv2.CV_32F, ko)
            mag = np.sqrt(re**2 + im**2)           # phase-invariant energy
            if SMOOTH:
                mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=lam)
            feats.append(mag)

    F = np.stack(feats, -1)                        # (WORK, WORK, N_FREQ*N_ORIENT)
    F = cv2.resize(F, (OUT, OUT), interpolation=cv2.INTER_AREA)
    return F.reshape(-1, F.shape[-1])              # (OUT*OUT, 24), row-major


def save_npz(img_path, out_name, labels=None, **kw):
    X = gabor_features(img_path, **kw)
    d = dict(features=X, H=OUT, W=OUT)
    if labels is not None:                         # labels: (OUT, OUT) array
        d["labels"] = np.asarray(labels).reshape(-1)
    np.savez(out_name, **d)
    print(out_name, X.shape)


if __name__ == "__main__":
    save_npz("src.png", "src.npz")
    save_npz("trg1.png", "trg1.npz", noise_sigma=0.05)                       # clean source, no noise
    save_npz("trg1.png", "trg2.npz", noise_sigma=0.1)     # noisy target, like T3-1
    save_npz("trg3.png", "trg3.npz", noise_sigma=0.05)
    save_npz("trg4.png", "trg4.npz", noise_sigma=0.05)
    save_npz("trg5.png", "trg5.npz", noise_sigma=0.05)
    save_npz("trg6.png", "trg6.npz", noise_sigma=0.05)
   
"""
Feature extraction for IMAGES (csv / npz files already contain features, so they skip this file).

To add a new method:  1) write a function  my_method(image) -> array of shape (N pixels, D features)
                      2) add it to FEATURE_METHODS at the bottom.
Then set FEATURES = "my_method" in main.py.  Nothing else changes.
"""
import numpy as np


def intensity_xy(image):
    """Paper Section 4.1: 3 features per pixel = CT value, X coordinate, Y coordinate."""
    h, w = image.shape
    ys, xs = np.mgrid[0:h, 0:w]
    x = xs / max(w - 1, 1)
    y = ys / max(h - 1, 1)
    return np.stack([image.ravel(), x.ravel(), y.ravel()], axis=1)


def gabor(image, step=8):
    """
    Same as Gabor-downsampled.ipynb: 4 orientations x 4 frequencies = 16 OpenCV Gabor filters.
    Every filter is applied to the FULL image (raw response), then only every step-th pixel is kept
    (640x640, step 8 -> 80x80 = 6400 rows).  Output: (kept pixels, 16).
    """
    import cv2                                               # only needed when Gabor is used
    img = np.clip(image, 0, 255).astype(np.uint8)            # the notebook filters a uint8 image
    responses = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for freq in [1 / 14, 1 / 28, 1 / 56, 1 / 112]:
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 1 / freq, 0.5, 0, ktype=cv2.CV_32F)
            responses.append(cv2.filter2D(img, cv2.CV_32F, kernel)[::step, ::step])
    return np.stack(responses, axis=-1).reshape(-1, len(responses)).astype(np.float32)


FEATURE_METHODS = {
    "intensity_xy": intensity_xy,
    "gabor": gabor,
}


def extract_features(image, method="intensity_xy"):
    return FEATURE_METHODS[method](image)

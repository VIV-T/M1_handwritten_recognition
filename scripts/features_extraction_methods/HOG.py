# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:13:23 2026

@author: veroe
"""

# -*- coding: utf-8 -*-
"""
Handcrafted feature extraction using only OpenCV + NumPy
Features:
- HOG descriptors
"""

import cv2

# --------------------------------------------------
# HOG descriptor initialization
# --------------------------------------------------
hog = cv2.HOGDescriptor(
    _winSize=(64, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

# --------------------------------------------------
# Feature extraction function (HOG)
# --------------------------------------------------
def extract_hog_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None

    # Resize to HOG expected size
    img = cv2.resize(img, (64, 128))

    # Compute HOG descriptor
    hog_vector = hog.compute(img)

    if hog_vector is None:
        return None

    return hog_vector.flatten()
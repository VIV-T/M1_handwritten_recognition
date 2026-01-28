# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:13:23 2026

@author: veroe
"""

# -*- coding: utf-8 -*-
"""
Handcrafted feature extraction using only OpenCV + NumPy
Features:
- Hu moments (7)
- Shape descriptors: area, perimeter, circularity, roundness, aspect ratio, extent, solidity
- Gradient statistics
- Curvature statistics
"""

import cv2
import numpy as np
import csv
import os

# --------------------------------------------------
# Feature extraction function
# --------------------------------------------------
def extract_hu_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None

    # Resize for normalization
    img = cv2.resize(img, (32, 32))

    # --------------------------------------------------
    # Binarization & contour extraction
    # --------------------------------------------------
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Keep the largest contour
    c = max(contours, key=cv2.contourArea)

    # --------------------------------------------------
    # 1 Hu moments (global shape)
    # --------------------------------------------------
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    

    # --------------------------------------------------
    # 2 Feature vector concatenation
    # --------------------------------------------------
    feature_vector = np.concatenate([
        hu  # 7
    ])

    return feature_vector
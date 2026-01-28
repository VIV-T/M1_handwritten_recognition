# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:13:23 2026

@author: veroe
"""

# -*- coding: utf-8 -*-
"""
Handcrafted feature extraction using only OpenCV + NumPy
Features (11):
- Shape descriptors: area, perimeter, circularity, roundness, aspect ratio, extent, solidity
- Gradient statistics
- Curvature statistics
"""

import cv2
import numpy as np

# --------------------------------------------------
# Feature extraction function
# --------------------------------------------------
def extract_geometric_features(img_path):
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
    # 2️⃣ Basic shape descriptors
    # --------------------------------------------------
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0.0
    extent = area / (w * h) if w * h > 0 else 0.0

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # --------------------------------------------------
    # 3️⃣ Circularity & Roundness
    # --------------------------------------------------
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    (xc, yc), radius = cv2.minEnclosingCircle(c)
    diameter = 2 * radius
    roundness = 4 * area / (np.pi * diameter ** 2) if diameter > 0 else 0.0

    # --------------------------------------------------
    # 4️⃣ Gradient-based statistics
    # --------------------------------------------------
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mean = np.mean(magnitude)
    grad_std = np.std(magnitude)

    # --------------------------------------------------
    # 5️⃣ Curvature descriptors (contour-based)
    # --------------------------------------------------
    c_pts = c[:, 0, :].astype(float)
    x_smooth = np.convolve(c_pts[:, 0], np.ones(3)/3, mode='same')  # simple smoothing
    y_smooth = np.convolve(c_pts[:, 1], np.ones(3)/3, mode='same')

    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-10)
    curvature_mean = np.mean(curvature)
    curvature_std = np.std(curvature)

    # --------------------------------------------------
    # 6️⃣ Feature vector concatenation
    # --------------------------------------------------
    feature_vector = np.concatenate([
        
        np.array([
            area, perimeter, circularity, roundness,
            aspect_ratio, extent, solidity,
            grad_mean, grad_std, curvature_mean, curvature_std
        ])  # 11
    ])

    return feature_vector

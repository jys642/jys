# -*- coding: utf-8 -*-
"""
缺陷特征提取模块 —— 为随机森林二次校验提供特征向量
================================================================
从检测框裁剪区域内提取几何（轮廓/形状）、灰度、纹理三类特征。
训练与推理必须调用同一个 extract_features，保证特征空间一致。
================================================================
"""
import cv2
import numpy as np

# 特征顺序（与 extract_features 返回值一一对应，用于特征重要度说明）
FEATURE_NAMES = [
    "area", "width", "height", "aspect_ratio", "fill_ratio",
    "perimeter", "circularity", "hu1", "hu2",
    "mean_gray", "std_gray", "min_gray", "max_gray", "gray_range",
    "laplacian_var", "grad_mean", "grad_std", "edge_density",
]


def extract_features(img_bgr, bbox):
    """从 bbox=(x1, y1, x2, y2) 区域提取特征，返回一维 float 列表。

    特征共 18 维（见 FEATURE_NAMES）：
        - 几何：面积、宽高、宽高比、填充率、周长、圆度、Hu 矩(2)
        - 灰度：均值、标准差、最小值、最大值、极差
        - 纹理：拉普拉斯方差（锐度）、梯度均值/标准差、边缘密度
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    crop = img_bgr[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    area = float(w * h)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 缺陷前景分割：Otsu 阈值 + 形态学闭运算，取最大连通域近似缺陷区域
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(c))
        perimeter = float(cv2.arcLength(c, True))
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu = [-np.sign(x) * np.log10(abs(x) + 1e-10) for x in hu]
    else:
        contour_area = perimeter = 0.0
        hu = [0.0] * 7

    fill_ratio = contour_area / area if area > 0 else 0.0
    circularity = (4 * np.pi * contour_area / (perimeter ** 2)) if perimeter > 0 else 0.0

    mean_g, std_g = float(gray.mean()), float(gray.std())
    min_g, max_g = float(gray.min()), float(gray.max())

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_mean, grad_std = float(grad.mean()), float(grad.std())

    edge = cv2.Canny(gray, 50, 150)
    edge_density = float((edge > 0).sum()) / area if area > 0 else 0.0

    feat = [
        area, float(w), float(h), (w / h if h > 0 else 0.0), fill_ratio,
        perimeter, circularity, hu[0], hu[1],
        mean_g, std_g, min_g, max_g, max_g - min_g,
        lap_var, grad_mean, grad_std, edge_density,
    ]
    return [float(v) for v in feat]

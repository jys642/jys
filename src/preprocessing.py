# -*- coding: utf-8 -*-
"""
图像预处理模块 —— 对应《方案设计》三大算法模块之一（机器视觉图像预处理）
================================================================
提供灰度化、高斯滤波去噪、CLAHE 对比度增强、ROI 裁剪、尺寸归一化等操作。
推理时对上传的药板图像执行完整预处理流水线，抑制铝箔反光/噪声、增强对比度，
优化模型输入质量。
================================================================
"""
import cv2

from src.config import IMG_SIZE


def to_grayscale(img):
    """灰度化：转灰度后复制回三通道（供灰度模式训练/展示使用）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def gaussian_denoise(img, ksize=3):
    """高斯滤波去噪：抑制铝箔表面噪声与轻微反光。"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def clahe_enhance(img, clip=2.0, grid=8):
    """CLAHE 对比度增强：LAB 空间对亮度通道做限制对比度自适应直方图均衡。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def roi_crop(img, bbox):
    """ROI 区域裁剪：bbox=(x1, y1, x2, y2)。

    合成数据整图即完整 ROI；真实产线数据可据此锁定药板有效区域、剔除背景。
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    return img[y1:y2, x1:x2]


def resize_norm(img, size=IMG_SIZE):
    """尺寸归一化：统一缩放到 size×size。"""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess(img, grayscale=False, bbox=None):
    """完整预处理流水线：ROI → 去噪 → 对比度增强 → 尺寸归一化。

    参数：
        img        BGR 图像（numpy 数组）
        grayscale  是否灰度化（默认关闭，保留色彩供 YOLO 检测）
        bbox       可选 ROI 裁剪区域 (x1, y1, x2, y2)
    """
    if bbox is not None:
        img = roi_crop(img, bbox)
    if grayscale:
        img = to_grayscale(img)
    img = gaussian_denoise(img)
    img = clahe_enhance(img)
    img = resize_norm(img)
    return img

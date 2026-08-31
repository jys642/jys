# -*- coding: utf-8 -*-
"""
训练随机森林二次校验模型
================================================================
1. 从 data/processed 的 train+val 划分中，依据真实标注框提取缺陷区域特征；
2. 训练随机森林分类器（6 类缺陷），输出特征重要度；
3. 保存模型到 models/rf_classifier.pkl。

用法：
    python scripts/train_rf.py
================================================================
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import INDEX_JSON, PROC_IMG, PROC_LBL, RF_MODEL
from src.classifier import DefectClassifier
from src.features import FEATURE_NAMES, extract_features


def parse_yolo_label(path):
    """读取 YOLO 标注，返回 [(cls, cx, cy, w, h), ...]（归一化坐标）。"""
    out = []
    for line in open(path, encoding="utf-8"):
        p = line.strip().split()
        if len(p) == 5:
            out.append([int(p[0]), *map(float, p[1:])])
    return out


def load_features():
    """从真实标注框提取特征矩阵 X 与类别标签 y。"""
    index = json.load(open(INDEX_JSON, encoding="utf-8"))
    X, y = [], []
    for name in index["splits"]["train"] + index["splits"]["val"]:
        stem = os.path.splitext(name)[0]
        img = cv2.imread(os.path.join(PROC_IMG, name))
        if img is None:
            continue
        H, W = img.shape[:2]
        for cls, cx, cy, w, h in parse_yolo_label(os.path.join(PROC_LBL, stem + ".txt")):
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            X.append(extract_features(img, (x1, y1, x2, y2)))
            y.append(cls)
    return np.asarray(X), np.asarray(y)


def main():
    X, y = load_features()
    print(f"特征矩阵：{X.shape}，标签：{len(y)}")

    clf = DefectClassifier.train(X, y, RF_MODEL)
    print("随机森林已训练并保存：", RF_MODEL)

    order = np.argsort(clf.feature_importances_)[::-1]
    print("\n特征重要度（降序）：")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:<16} {clf.feature_importances_[i]:.4f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
训练 YOLO 缺陷检测模型
================================================================
1. 依据 data/processed/index.json 的划分，构建 ultralytics YOLO 数据集目录；
2. 基于 yolo8n 预训练权重微调，训练药板缺陷检测模型；
3. 将最佳权重复制到 models/best.pt。

用法：
    python scripts/train_yolo.py
================================================================
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (CLASSES, INDEX_JSON, MODELS_DIR, PROC_IMG, PROC_LBL,
                        YOLO_BATCH, YOLO_DATASET, YOLO_EPOCHS, YOLO_IMGSZ, YOLO_MODEL)
from src.detection import YoloDetector


def build_yolo_dataset():
    """按 index.json 的 train/val 划分，把 processed 数据整理成 YOLO 目录结构。"""
    index = json.load(open(INDEX_JSON, encoding="utf-8"))
    for split in ("train", "val"):
        img_dst = os.path.join(YOLO_DATASET, "images", split)
        lbl_dst = os.path.join(YOLO_DATASET, "labels", split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)
        for name in index["splits"][split]:
            stem = os.path.splitext(name)[0]
            shutil.copy(os.path.join(PROC_IMG, name), os.path.join(img_dst, name))
            shutil.copy(os.path.join(PROC_LBL, stem + ".txt"), os.path.join(lbl_dst, stem + ".txt"))

    data_yaml = os.path.join(YOLO_DATASET, "data.yaml")
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {YOLO_DATASET.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    return data_yaml


def main():
    data_yaml = build_yolo_dataset()
    print("YOLO 数据集已构建：", data_yaml)

    best = YoloDetector.train(data_yaml, epochs=YOLO_EPOCHS,
                              imgsz=YOLO_IMGSZ, batch=YOLO_BATCH)

    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy(best, YOLO_MODEL)
    print("已保存最佳权重：", YOLO_MODEL)


if __name__ == "__main__":
    main()

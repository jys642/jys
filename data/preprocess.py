# -*- coding: utf-8 -*-
"""
药板药片外观缺陷智能检测系统 —— 数据预处理脚本
================================================================
对 /data/raw 下的自建合成药板数据集执行工业图像预处理，并输出到 /data/processed。

预处理流水线（对应《方案设计》图像预处理模块）：
    1. 灰度化   to_grayscale      —— 可选（默认关闭，保留色彩供 YOLO 检测）
    2. 高斯滤波 去噪               —— GaussianBlur 3×3，抑制铝箔噪声
    3. 对比度增强                 —— CLAHE（限制对比度自适应直方图均衡化）
    4. ROI 区域裁剪              —— 对整幅图即完整 ROI（合成数据无需额外裁剪）
    5. 尺寸归一化                 —— 统一缩放到 640×640

数据增广（扩充样本、提升泛化）：
    hflip    水平翻转（标注 x 取反）
    rot90    顺时针旋转 90°（标注坐标变换）
    bright   亮度增强（标注不变）

划分：按 80/10/10 划分为 train / val / test，固定随机种子保证可复现。

输出目录：
    data/processed/images/*.jpg        预处理 + 增广后的图像
    data/processed/labels/*.txt       对应 YOLO 标注
    data/processed/index.json         划分索引（含类别、预处理与增广说明）

用法：
    python data/preprocess.py
================================================================
"""
import os
import glob
import json
import random

import numpy as np
import cv2

SEED = 42
IMG_SIZE = 640
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_IMG = os.path.join(HERE, "raw", "images")
RAW_LBL = os.path.join(HERE, "raw", "labels")
OUT_IMG = os.path.join(HERE, "processed", "images")
OUT_LBL = os.path.join(HERE, "processed", "labels")


# ---------------------------------------------------------------- 预处理函数
def to_grayscale(img):
    """灰度化：转灰度后复制为三通道（供灰度模式训练/展示使用）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def gaussian_denoise(img, ksize=3):
    """高斯滤波去噪：抑制铝箔表面噪声与轻微反光。"""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def clahe_enhance(img, clip=2.0, grid=8):
    """CLAHE 对比度增强：在 LAB 空间对亮度通道做自适应直方图均衡。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_norm(img, size=IMG_SIZE):
    """尺寸归一化：统一缩放到 size×size。"""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def preprocess(img, grayscale=False):
    """完整预处理流水线：去噪 → 对比度增强 → 尺寸归一化。"""
    if grayscale:
        img = to_grayscale(img)
    img = gaussian_denoise(img)
    img = clahe_enhance(img)
    img = resize_norm(img)
    return img


# ---------------------------------------------------------------- 标注变换
def parse_labels(path):
    """读取 YOLO 标注，返回 [(cls, cx, cy, w, h), ...]。"""
    labels = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            p = line.strip().split()
            if len(p) == 5:
                labels.append([int(p[0]), *map(float, p[1:])])
    return labels


def format_label(cls, cx, cy, w, h):
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def transform_hflip(labels):
    """水平翻转：cx' = 1 - cx。"""
    return [[c, 1 - x, y, w, h] for c, x, y, w, h in labels]


def transform_rot90(labels):
    """顺时针旋转 90°：cx' = 1 - cy, cy' = cx, w' = h, h' = w。"""
    return [[c, 1 - y, x, h, w] for c, x, y, w, h in labels]


# ---------------------------------------------------------------- 增广
def augment(img, labels, name):
    """返回 [(文件名, 图像, 标注), ...]，含原图与增广样本。"""
    out = [("", img, labels)]  # 原图

    # 水平翻转
    out.append(("_hflip", cv2.flip(img, 1), transform_hflip(labels)))
    # 顺时针旋转 90°
    out.append(("_rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                transform_rot90(labels)))
    # 亮度增强
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=8)
    out.append(("_bright", bright, labels))

    return out


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LBL, exist_ok=True)

    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(RAW_IMG, "*.jpg")))
    random.shuffle(files)

    # 划分
    n = len(files)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    index = {"train": [], "val": [], "test": []}
    count = 0

    for split_name, names in splits.items():
        for name in names:
            stem = os.path.splitext(name)[0]
            img = cv2.imread(os.path.join(RAW_IMG, name))
            if img is None:
                continue
            img = preprocess(img)
            labels = parse_labels(os.path.join(RAW_LBL, stem + ".txt"))

            for suffix, aug_img, aug_labels in augment(img, labels, stem):
                out_name = stem + suffix + ".jpg"
                cv2.imwrite(os.path.join(OUT_IMG, out_name), aug_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                with open(os.path.join(OUT_LBL, stem + suffix + ".txt"),
                          "w", encoding="utf-8") as f:
                    f.write("\n".join(format_label(*x) for x in aug_labels))
                index[split_name].append(out_name)
                count += 1

    # 类别信息
    class_path = os.path.join(HERE, "raw", "classes.txt")
    classes = [line.strip() for line in open(class_path, encoding="utf-8") if line.strip()]
    class_zh = {
        "missing": "缺粒", "crack": "裂片", "broken": "破损",
        "flash": "飞边", "stain": "污渍", "empty": "漏装",
    }

    index_out = {
        "dataset": "药板药片外观缺陷自建合成数据集（预处理后）",
        "classes": classes,
        "class_names_zh": {c: class_zh.get(c, "") for c in classes},
        "image_size": [IMG_SIZE, IMG_SIZE],
        "total_images": count,
        "preprocessing": {
            "denoise": "GaussianBlur 3x3",
            "contrast": "CLAHE clip=2.0 grid=8x8 (LAB-L)",
            "resize": "640x640",
            "grayscale": "available (off by default, keeps color for YOLO)",
        },
        "augmentation": ["origin", "hflip", "rot90", "bright"],
        "split_ratio": SPLIT,
        "splits": index,
    }
    with open(os.path.join(HERE, "processed", "index.json"), "w", encoding="utf-8") as f:
        json.dump(index_out, f, ensure_ascii=False, indent=2)

    print(f"预处理完成：共输出 {count} 张图像")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} 张原始图 -> {len(index[k])} 张（含增广）")
    print("索引文件：data/processed/index.json")


if __name__ == "__main__":
    main()

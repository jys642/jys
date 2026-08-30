# -*- coding: utf-8 -*-
"""
药板药片外观缺陷智能检测系统 —— 自建合成数据集生成脚本
================================================================
功能：生成泡罩药板（2×5 药窝）合成图像与 YOLO 格式标注，模拟 6 类药片缺陷。

缺陷类别（与 classes.txt 顺序一致，也与《选题说明》一致）：
    0 missing 缺粒   1 crack 裂片   2 broken 破损
    3 flash  飞边    4 stain 污渍   5 empty 漏装

输出目录：
    data/raw/images/*.jpg   合成泡罩药板图像（640×640）
    data/raw/labels/*.txt   YOLO 标注（class cx cy w h，均已归一化）
    data/raw/classes.txt    类别名称

用法：
    python data/generate_dataset.py [图片数量]   （默认 120）
================================================================
"""
import os
import sys
import random

import numpy as np
import cv2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CLASSES = ["missing", "crack", "broken", "flash", "stain", "empty"]
CLASS_CN = {
    "missing": "缺粒", "crack": "裂片", "broken": "破损",
    "flash": "飞边", "stain": "污渍", "empty": "漏装",
}

IMG = 640
ROWS, COLS = 2, 5
POCKET_R = 52          # 药窝（气泡）半径
GX = 90                # 水平边距
GY = 165               # 垂直边距
X_STEP = (IMG - 2 * GX) // (COLS - 1)
Y_STEP = (IMG - 2 * GY) // (ROWS - 1)
DEFECT_PROB = 0.30     # 每个药窝出现缺陷的概率

TABLET_COLORS = [
    (232, 232, 238),   # 白
    (215, 85, 90),     # 红
    (85, 140, 230),    # 蓝
    (90, 200, 140),    # 绿
    (238, 182, 82),    # 橙
    (180, 120, 205),   # 紫
]
FOIL_BASE = (196, 198, 205)


def make_foil():
    """铝箔背景：底色 + 高斯噪声 + 轻微水平渐变。"""
    img = np.full((IMG, IMG, 3), FOIL_BASE, dtype=np.int16)
    noise = np.random.normal(0, 4.0, (IMG, IMG, 3)).astype(np.int16)
    img = img + noise
    grad = np.linspace(-6, 6, IMG, dtype=np.int16).reshape(1, IMG, 1)
    img = img + grad
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_pocket(img, cx, cy):
    """泡罩凹坑：带立体感阴影与高光的透明气泡。"""
    cv2.circle(img, (cx, cy), POCKET_R + 6, (150, 152, 160), 3)
    cv2.circle(img, (cx, cy), POCKET_R, (225, 226, 232), -1)
    cv2.circle(img, (cx - 12, cy - 12), POCKET_R - 10, (238, 239, 244), -1)
    cv2.circle(img, (cx + 10, cy + 10), POCKET_R - 14, (205, 207, 214), -1)


def draw_round_tablet(img, cx, cy, color, r=34):
    """圆形药片：主体 + 轮廓 + 中间刻痕。"""
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
    cv2.line(img, (cx - r + 6, cy), (cx + r - 6, cy), (60, 60, 60), 2)


def draw_capsule(img, cx, cy, color):
    """胶囊药片：两段式椭圆 + 中缝。"""
    length, radius = 80, 26
    cv2.ellipse(img, (cx, cy), (length // 2, radius), 0, 0, 360, color, -1)
    lighter = tuple(min(255, c + 35) for c in color)
    cv2.ellipse(img, (cx - length // 4, cy), (length // 4, radius), 0, 0, 360, lighter, -1)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), (55, 55, 55), 2)
    cv2.ellipse(img, (cx, cy), (length // 2, radius), 0, 0, 360, (0, 0, 0), 2)


def draw_crack(img, cx, cy, r=38):
    """裂片：药片上的深色折线裂纹。"""
    pts = []
    x0, y0 = cx - r // 2, cy - r // 2
    for i in range(5):
        pts.append([x0 + random.randint(-6, 6), y0 + i * (r // 2)])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, (40, 40, 40), 3)


def draw_broken(img, cx, cy, color, r=36):
    """破损：药片碎裂成多个碎片 + 裂纹。"""
    for _ in range(4):
        fx = cx + random.randint(-r + 8, r - 8)
        fy = cy + random.randint(-r + 8, r - 8)
        fr = random.randint(7, 14)
        pts = np.array([
            [fx - fr, fy - fr], [fx + fr, fy - fr],
            [fx + fr, fy + fr], [fx - fr, fy + fr],
        ], np.int32) + np.random.randint(-4, 5, (4, 2))
        cv2.fillPoly(img, [pts], color)
    cv2.line(img, (cx - r, cy - r), (cx + r, cy + r), (50, 50, 50), 2)
    cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), (50, 50, 50), 2)


def draw_flash(img, cx, cy, color, r=34):
    """飞边：药片边缘不规则的毛刺溢出。"""
    for _ in range(12):
        ang = random.uniform(0, 2 * np.pi)
        spike = random.randint(10, 22)
        x2 = int(cx + (r + spike) * np.cos(ang))
        y2 = int(cy + (r + spike) * np.sin(ang))
        x1 = int(cx + (r - 2) * np.cos(ang))
        y1 = int(cy + (r - 2) * np.sin(ang))
        cv2.line(img, (x1, y1), (x2, y2), color, 4)


def draw_stain(img, cx, cy, r=40):
    """污渍：药片/铝箔上的深色污点。"""
    for _ in range(3):
        sx = cx + random.randint(-r, r)
        sy = cy + random.randint(-r, r)
        sr = random.randint(8, 18)
        cv2.circle(img, (sx, sy), sr, (60, 55, 50), -1)
        cv2.circle(img, (sx, sy), sr, (45, 40, 36), 2)


def draw_empty_foil(img, cx, cy):
    """漏装：药窝被刺破，铝箔破损/压痕。"""
    cv2.circle(img, (cx, cy), POCKET_R - 6, (120, 118, 110), -1)
    for _ in range(4):
        ang = random.uniform(0, 2 * np.pi)
        x2 = int(cx + (POCKET_R - 8) * np.cos(ang))
        y2 = int(cy + (POCKET_R - 8) * np.sin(ang))
        cv2.line(img, (cx, cy), (x2, y2), (60, 58, 52), 2)


def gen_image(idx):
    """生成一张图像及其标注列表。返回 (img, labels)。"""
    img = make_foil()
    labels = []
    defect_cnt = 0

    for r in range(ROWS):
        for c in range(COLS):
            cx = GX + c * X_STEP
            cy = GY + r * Y_STEP
            draw_pocket(img, cx, cy)

            defect = None
            if random.random() < DEFECT_PROB:
                defect = random.choice(CLASSES)

            color = random.choice(TABLET_COLORS)
            is_capsule = random.random() < 0.4

            if defect is None:
                if is_capsule:
                    draw_capsule(img, cx, cy, color)
                else:
                    draw_round_tablet(img, cx, cy, color)
            elif defect == "missing":
                pass  # 缺粒：药窝为空
            elif defect == "empty":
                draw_empty_foil(img, cx, cy)  # 漏装：铝箔破损
            elif defect == "broken":
                draw_broken(img, cx, cy, color)  # 破损：仅碎片
            else:
                if is_capsule:
                    draw_capsule(img, cx, cy, color)
                else:
                    draw_round_tablet(img, cx, cy, color)
                if defect == "crack":
                    draw_crack(img, cx, cy)
                elif defect == "flash":
                    draw_flash(img, cx, cy, color)
                elif defect == "stain":
                    draw_stain(img, cx, cy)

            if defect is not None:
                defect_cnt += 1
                cls = CLASSES.index(defect)
                half = POCKET_R + 10
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(IMG, cx + half)
                y2 = min(IMG, cy + half)
                bx = (x1 + x2) / 2 / IMG
                by = (y1 + y2) / 2 / IMG
                bw = (x2 - x1) / IMG
                bh = (y2 - y1) / IMG
                labels.append(f"{cls} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

    return img, labels, defect_cnt


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    here = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(here, "raw", "images")
    lbl_dir = os.path.join(here, "raw", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    total_defects = 0
    for i in range(1, n + 1):
        img, labels, dc = gen_image(i)
        if dc == 0:  # 兜底：确保每张图至少 1 个缺陷
            cx = GX + random.randint(0, COLS - 1) * X_STEP
            cy = GY + random.randint(0, ROWS - 1) * Y_STEP
            draw_empty_foil(img, cx, cy)
            cls = CLASSES.index("empty")
            half = POCKET_R + 10
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(IMG, cx + half), min(IMG, cy + half)
            labels.append(f"{cls} {(x1+x2)/2/IMG:.6f} {(y1+y2)/2/IMG:.6f} "
                          f"{(x2-x1)/IMG:.6f} {(y2-y1)/IMG:.6f}")
            dc = 1

        name = f"blister_{i:04d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(lbl_dir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
        total_defects += dc

    with open(os.path.join(here, "raw", "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASSES) + "\n")

    print(f"已生成 {n} 张图像，共 {total_defects} 个缺陷标注")
    print(f"图像目录：{img_dir}")
    print(f"标注目录：{lbl_dir}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
可视化绘制模块
================================================================
在检测图上绘制缺陷框与中文类别标签。

OpenCV 的 cv2.putText 只支持 ASCII（中文会渲染成 ????），
故改用 PIL ImageDraw + 系统中文字体渲染，保证类别中文标签正确显示。
================================================================
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import CLASS_CN

# 常见中文字体候选路径（按优先级），取第一个存在的
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑粗体
    "C:/Windows/Fonts/simhei.ttf",     # 黑体
    "C:/Windows/Fonts/simsun.ttc",     # 宋体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

_FONT_CACHE = {}


def _load_font(size):
    """按需加载系统中文字体；找不到则回退到 PIL 默认字体（中文可能仍无法显示）。"""
    if size not in _FONT_CACHE:
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    _FONT_CACHE[size] = ImageFont.truetype(path, size)
                    return _FONT_CACHE[size]
                except Exception:
                    continue
        _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_boxes(img_bgr, defects):
    """在 BGR 图像上绘制缺陷框与中文类别标签，返回标注图（BGR ndarray）。

    defects 中每项需含键：
        bbox: [x1, y1, x2, y2]
        final_class: 英文类名（如 "missing"，经 CLASS_CN 映射为中文）
        conf: 置信度
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(16)

    for d in defects:
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        cn = CLASS_CN.get(d["final_class"], d["final_class"])
        label = f"{cn} {d['conf']:.2f}"

        # 缺陷框
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        # 标签文本（置于框上方）+ 红色背景色块，避免与图像底色重叠
        text_xy = (x1, max(0, y1 - 20))
        try:
            tb = draw.textbbox(text_xy, label, font=font)
        except Exception:
            tb = (x1, max(0, y1 - 20), x1 + len(label) * 10, max(0, y1 - 20) + 20)
        draw.rectangle(tb, fill=(255, 0, 0))
        draw.text(text_xy, label, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

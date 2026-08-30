---
license: mit
task_categories:
  - object-detection
tags:
  - pharmaceutical
  - blister-pack
  - defect-detection
  - yolo
  - computer-vision
pretty_name: Pharmaceutical Blister Pack Defect Dataset (Synthetic)
size_categories:
  - n<1K
---

# 药板药片外观缺陷数据集（自建合成）

面向「药板药片外观缺陷智能检测系统」课程设计自建的合成泡罩药板缺陷数据集，用于工业质量控制场景下的缺陷检测。

## 数据集简介

- **图像**：640×640 JPG，泡罩药板（2×5 药窝），铝箔背景 + 圆形/胶囊药片。
- **规模**：原始 120 张；预处理 + 增广后 480 张。
- **标注**：YOLO 格式（`class cx cy w h`，归一化坐标）。
- **性质**：合成数据（脚本生成），非真实产线采集，仅用于课程设计流程验证。

## 缺陷类别（6 类）

| ID | 类别 | 中文 |
|---|---|---|
| 0 | missing | 缺粒 |
| 1 | crack | 裂片 |
| 2 | broken | 破损 |
| 3 | flash | 飞边 |
| 4 | stain | 污渍 |
| 5 | empty | 漏装 |

## 目录结构

```
raw/
  images/            # 120 张原始图像
  labels/            # YOLO 标注
  classes.txt        # 类别名
processed/
  images/            # 480 张预处理 + 增广图像
  labels/            # 对应标注
  index.json         # train/val/test 划分索引
```

## 预处理

灰度化（可选）、高斯滤波去噪、CLAHE 对比度增强、ROI 裁剪、尺寸归一化；增广：水平翻转、旋转 90°、亮度增强。

## 引用

如使用本数据集，请注明为自建合成数据集。

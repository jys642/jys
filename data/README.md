# 数据说明（/data）

本目录存放「药板药片外观缺陷智能检测系统」课程设计所需的数据，采用**「公开数据集引用 + 自建小样本提交」**的双轨策略：自建合成数据直接提交到本仓库保证项目可运行、可复现；公开数据集作为大规模训练来源在下方给出有效链接。

## 一、数据来源

### 1. 公开数据集（引用链接）
以下公开数据集作为真实工业数据来源，链接有效、可免费下载：

| 数据集 | 链接 | 规模 | 说明 |
|---|---|---|---|
| Paracetamol Defect Dataset (Kaggle) | https://www.kaggle.com/datasets/arthurkray/paracetamol-defect-dataset | 3500 张 / 66.87MB / 200×200 | 5 类：normal、broken、colored、stained、unknown，工业质检场景 |
| Pill Defect Dataset (Kaggle) | https://www.kaggle.com/datasets/pudpawat/pill-defect-dataset | 363 张 / 30.49MB | defect / normal 二分类，含真实产线旋转、噪声、光照变化 |
| Pill-Defects (GitHub 镜像) | https://github.com/PerceptiLabs/Pill-Defects | defect / normal + data.csv | 上者的 GitHub 镜像，可直接 git clone |

> Kaggle 数据集需登录后下载；GitHub 镜像可直接克隆。公开数据集类别与本项目 6 类缺陷不完全一一对应，实际训练时可据此做类别映射或再标注。

### 2. 自建数据集（开源到 HuggingFace / ModelScope，引用链接）
因目标缺陷类别（缺粒、裂片、破损、飞边、污渍、漏装）需与《选题说明》严格一致，且公开泡罩数据集的类别/标注格式不统一，本项目用脚本 `generate_dataset.py` 合成自建泡罩药板缺陷数据集，并开源到 HuggingFace 平台，通过链接引用。

- **HuggingFace 链接**：https://huggingface.co/datasets/<你的用户名>/pharma-blister-defect-dataset （待上传后填入实际链接）
- **上传脚本**：`data/upload_to_hf.py`（登录 HF 后一键上传；ModelScope 魔搭同理，可参照脚本改用 `modelscope` 库）
- **数据集卡片**：`data/hf_dataset_card.md`（上传时作为仓库 README）
- **规模**：120 张图像（640×640，2×5 药窝布局），6 类缺陷，共 372 个标注，类别均衡。
- **标注**：YOLO 格式（`class cx cy w h`，均归一化），类别见 `raw/classes.txt`。
- **本地副本**：仓库内 `data/raw`、`data/processed` 保留数据副本，用于本地复现与流程验证（合成数据，非真实产线图像）。

## 二、目录结构

```
data/
├── README.md               # 本文件：数据来源与预处理说明
├── generate_dataset.py     # 自建数据生成脚本
├── preprocess.py           # 数据预处理脚本
├── raw/                    # 原始数据（自建合成）
│   ├── images/             # 120 张原始泡罩药板图像
│   ├── labels/             # 120 个 YOLO 标注
│   └── classes.txt         # 类别名（missing/crack/broken/flash/stain/empty）
└── processed/              # 预处理 + 增广后的数据
    ├── images/             # 480 张预处理后图像
    ├── labels/             # 对应标注（含增广坐标变换）
    └── index.json          # 划分索引（train/val/test + 类别 + 预处理说明）
```

## 三、缺陷类别定义

| ID | 英文 | 中文 | 说明 |
|---|---|---|---|
| 0 | missing | 缺粒 | 药窝内药片缺失，铝箔完好 |
| 1 | crack | 裂片 | 药片表面存在裂纹 |
| 2 | broken | 破损 | 药片碎裂成多块 |
| 3 | flash | 飞边 | 药片边缘毛刺/溢出 |
| 4 | stain | 污渍 | 药片或铝箔表面污点 |
| 5 | empty | 漏装 | 药窝被刺破/铝箔破损伴随空窝 |

> 注：`missing`（缺粒）与 `empty`（漏装）视觉上都表现为药窝为空，判定口径为：缺粒 = 铝箔完好、仅药片缺失；漏装 = 铝箔破损/压痕伴随空窝。答辩时按此口径解释。

## 四、数据预处理

预处理脚本 `preprocess.py` 实现《方案设计》图像预处理模块的完整流水线，并输出到 `processed/`：

1. **灰度化**（可选，默认关闭）——保留色彩供 YOLO 检测；灰度化函数 `to_grayscale()` 已实现，供推理管线按需调用。
2. **高斯滤波去噪** —— `GaussianBlur 3×3`，抑制铝箔噪声。
3. **CLAHE 对比度增强** —— LAB 空间对亮度通道做限制对比度自适应直方图均衡（clip=2.0, grid=8×8）。
4. **ROI 区域裁剪** —— 合成数据整图即完整 ROI，真实数据可据此裁剪药板有效区域。
5. **尺寸归一化** —— 统一缩放到 640×640。

**数据增广**（提升泛化）：水平翻转（标注 x 取反）、顺时针旋转 90°（标注坐标变换）、亮度增强。每张原始图增广为 4 张（原图 + 3 增广），合计 480 张。

**划分**：按 80/10/10 划分为 train（384）/ val（48）/ test（48），固定随机种子（seed=42）保证可复现，划分结果写入 `processed/index.json`。

## 五、复现方法

```bash
# 1. 重新生成自建数据（覆盖 data/raw）
python data/generate_dataset.py 120

# 2. 重新执行预处理 + 增广 + 划分（覆盖 data/processed）
python data/preprocess.py
```

依赖：Python 3.9+、numpy、opencv-python（`cv2`）。

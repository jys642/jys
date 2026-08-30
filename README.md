# 药板药片外观缺陷智能检测系统

面向制药厂泡罩包装药板生产线的 B/S 架构智能质检系统。用户通过浏览器上传药板图像，系统自动完成药片缺陷检测（缺粒、裂片、破损、飞边、污渍、漏装 6 类），输出缺陷位置、类别与置信度；检测结果持久化入库，支持历史记录查询与缺陷统计。

**技术栈**：机器视觉预处理（OpenCV）＋ YOLO 目标检测（PyTorch）＋ 随机森林二次校验（Scikit-learn）＋ FastAPI/SQLite 后端 ＋ HTML/Bootstrap 前端。

> 本仓库为「制造智能技术课程设计」项目，按阶段推进开发，全程使用 AI 编程工具（Claude Code）辅助，并如实留存提示词追溯记录。

## 项目结构

```
.
├── README.md                 # 本文件：项目总说明
├── 选题说明.md               # 项目题目与涉及技术方向
├── 方案设计.md               # 功能需求、技术路线、实施计划
├── 学习笔记.md               # 课程知识点学习笔记
├── task_plan.md / findings.md / progress.md   # 规划工作流文件
├── data/                     # 数据资源（见下方「数据」）
│   ├── README.md             # 数据来源与预处理详细说明
│   ├── generate_dataset.py   # 自建数据生成脚本
│   ├── preprocess.py         # 数据预处理脚本
│   ├── raw/                  # 原始数据（自建合成，120 张 + YOLO 标注）
│   └── processed/            # 预处理 + 增广后数据（480 张 + index.json）
└── prompt/                   # AI 工具提示词追溯记录
    └── prompt_log.json       # 与 AI 交流的提示词/会话日志
```

## 数据

### 数据来源（阶段一）
- **公开数据集**（引用链接）：
  - [Paracetamol Defect Dataset (Kaggle)](https://www.kaggle.com/datasets/arthurkray/paracetamol-defect-dataset) — 3500 张 / 5 类
  - [Pill Defect Dataset (Kaggle)](https://www.kaggle.com/datasets/pudpawat/pill-defect-dataset) — 363 张 / defect+normal
  - [Pill-Defects (GitHub)](https://github.com/PerceptiLabs/Pill-Defects) — defect/normal + data.csv
- **自建小数据集**（数据量小，直接提交到仓库 `/data` 目录）：
  - 用 `data/generate_dataset.py` 合成 120 张泡罩药板图像（6 类缺陷、YOLO 标注），原始数据在 `data/raw`，预处理+增广结果在 `data/processed`
  - 如需开源到 HuggingFace / ModelScope，可运行 `data/upload_to_hf.py` 上传后再引用平台链接

> 完整的数据来源、类别定义与目录说明见 [data/README.md](data/README.md)。

### 数据预处理（阶段一）
`data/preprocess.py` 实现工业图像预处理流水线，输出预处理结果与划分索引到 `data/processed/`：

1. 灰度化（可选，默认保留色彩供 YOLO）
2. 高斯滤波去噪（GaussianBlur 3×3）
3. CLAHE 对比度增强（LAB-L，clip=2.0, grid=8×8）
4. ROI 区域裁剪（合成数据整图即 ROI）
5. 尺寸归一化（640×640）

数据增广：水平翻转、旋转 90°、亮度增强。划分：80/10/10 → train/val/test（384/48/48），固定种子保证可复现，索引见 [data/processed/index.json](data/processed/index.json)。

## AI 工具提示词追溯

本课程设计全程使用 **Claude Code**（模型 deepseek-v4-pro）辅助开发。与 AI 的交流记录（提示词、AI 操作、结果摘要）以 JSON 形式留存于 [prompt/prompt_log.json](prompt/prompt_log.json)，并随每个阶段同步更新。

> 上下文压缩前备份规则：复制 `prompt/prompt_log.json` 为 `prompt/prompt_log_backup_<日期>.json` 后再追加新记录。

## 快速开始

```bash
# 重新生成自建数据
python data/generate_dataset.py 120

# 重新执行预处理 + 增广 + 划分
python data/preprocess.py
```

依赖：Python 3.9+、numpy、opencv-python。

## 开发阶段规划

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段一 | 方案确认与数据准备 | ✅ 已完成（本阶段） |
| 阶段二 | 算法模块开发（预处理 / YOLO / 随机森林） | 待开发 |
| 阶段三 | 后端服务接口开发（FastAPI + SQLite） | 待开发 |
| 阶段四 | 前端 UI 页面开发 | 待开发 |
| 阶段五 | 系统集成与功能测试 | 待开发 |
| 阶段六 | 系统输出与文档整理 | 待开发 |

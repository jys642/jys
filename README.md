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
├── src/                      # 算法模块（阶段二）
│   ├── config.py             # 全局配置（类别 / 路径 / 超参数）
│   ├── preprocessing.py      # 图像预处理模块
│   ├── detection.py          # YOLO 缺陷检测模块
│   ├── features.py           # 缺陷特征提取
│   ├── classifier.py         # 随机森林二次校验模块
│   └── pipeline.py           # 端到端检测管线
├── scripts/                  # 训练、推理与测试脚本
│   ├── train_yolo.py         # 训练 YOLO 检测模型
│   ├── train_rf.py           # 训练随机森林校验模型
│   ├── demo.py               # 端到端推理 Demo
│   └── test_system.py        # 系统集成与功能测试（阶段五）
├── models/                   # 训练好的模型权重
│   ├── best.pt               # YOLO11n 检测模型（微调）
│   └── rf_classifier.pkl     # 随机森林二次校验模型
├── app/                      # 后端服务（阶段三，FastAPI + SQLite）
│   ├── main.py               # FastAPI 应用与接口路由
│   ├── database.py           # SQLite 数据库（记录 + 缺陷明细）
│   └── schemas.py            # Pydantic 响应模型
├── frontend/                 # 前端页面（阶段四，HTML + Bootstrap）
│   ├── index.html            # 单页应用（上传检测 / 历史 / 统计）
│   ├── style.css             # 自定义样式
│   └── app.js                # 交互逻辑（对接后端 API）
├── requirements.txt          # Python 依赖清单
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

## 算法模块（阶段二）

三大算法模块位于 `src/`，训练脚本在 `scripts/`，模型权重在 `models/`，与《方案设计》技术路线一一对应：

### 1. 图像预处理模块（[src/preprocessing.py](src/preprocessing.py)）
实现灰度化（可选）、高斯滤波去噪、CLAHE 对比度增强、ROI 区域裁剪、尺寸归一化，统一为 640×640 输入，抑制铝箔反光/噪声，优化模型输入。

### 2. YOLO 缺陷检测模块（[src/detection.py](src/detection.py)）
基于 ultralytics **YOLO11n** 预训练权重在自建合成数据集上微调，输出缺陷定位框、初步类别与置信度。训练：`python scripts/train_yolo.py`（产物 `models/best.pt`）。

### 3. 随机森林二次校验模块（[src/classifier.py](src/classifier.py) + [src/features.py](src/features.py)）
从检测框区域提取 18 维特征（几何/灰度/纹理），用随机森林（200 棵树）对 YOLO 结果二次分类：RF 置信度足够时修正误检类别，否则保留 YOLO 结果。训练：`python scripts/train_rf.py`（产物 `models/rf_classifier.pkl`）。

### 端到端管线（[src/pipeline.py](src/pipeline.py)）
`预处理 → YOLO 检测 → 随机森林校验` 串联为 `DetectionPipeline.run()`，`scripts/demo.py` 提供命令行推理演示并输出可视化标注图。

## 后端服务（阶段三）

基于 FastAPI + SQLite，调用阶段二算法管线完成在线检测与结果持久化（入口 [app/main.py](app/main.py)）。

### 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/detect` | 上传药板图片，执行预处理 + YOLO + 随机森林校验，返回缺陷结果并入库 |
| GET | `/api/records` | 历史检测记录列表（分页 `limit`/`offset`） |
| GET | `/api/records/{id}` | 单条记录详情（含缺陷明细） |
| GET | `/api/statistics` | 缺陷类型统计（总量 + 各类别计数） |
| GET | `/api/image/{name}` | 返回已上传图像（原图 / 标注图） |

### 数据库表

- `records`：检测记录（图片路径、时间、缺陷数、质检结论）
- `defects`：缺陷明细（类别、置信度、框坐标、YOLO/RF 校验结果）

### 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 交互式 API 文档：http://127.0.0.1:8000/docs
```

## 前端页面（阶段四）

单页应用（[frontend/index.html](frontend/index.html)），Bootstrap + 原生 JS + Chart.js，由后端托管。启动后端后浏览器访问 `http://127.0.0.1:8000/` 即可使用。

### 页面功能

- **在线检测**：拖拽/选择上传药板图片并预览，检测后展示缺陷标注图、缺陷类别/置信度/YOLO-RF 校验详情、质检结论（合格/不合格）
- **历史记录**：表格展示历史检测记录，点击查看单条详情（标注图 + 缺陷明细）
- **统计分析**：累计检测次数 / 缺陷数 / 平均每板缺陷三张卡片 + 缺陷类型分布柱状图

### 技术要点

- 三个视图通过顶部导航切换，前端 `fetch` 调用 `/api/*` 接口
- 缺陷框标注、类别、置信度由后端返回的标注图 + 结构化 JSON 双重呈现
- 后端以 `StaticFiles` 托管前端，同源部署，避免跨域问题

## 系统集成与功能测试（阶段五）

系统级测试脚本 [scripts/test_system.py](scripts/test_system.py)，覆盖三部分：

1. **测试样本验证**：对 test 集全部 48 张样本执行端到端推理，与真实标注框做 IoU 匹配，输出整体与类别级 Precision / Recall / F1，并对比「YOLO + 随机森林」与「仅 YOLO」。
2. **后端接口全流程**：用 TestClient 走通「上传检测 → 入库 → 历史 → 详情 → 统计 → 图片返回 → 404 兜底」。
3. **前端静态资源**：校验首页与 css/js 正常返回。

```bash
python scripts/test_system.py            # 跑全部 test 集（48 张）
python scripts/test_system.py --limit 10 # 只跑前 10 张（快速验证）
```

**测试结论**（见 [progress.md](progress.md)）：test 集最终 Precision=0.969、Recall=1.000、F1=0.984。测试中发现并修复了随机森林二次校验的融合策略缺陷——原「RF 置信度 ≥0.5 即覆盖 YOLO」过于激进，会在 YOLO 已高置信时被 RF 中等置信的误判覆盖；现已引入 `YOLO_CONF_GATE` 门限，改为「YOLO 自信优先、YOLO 不确定时才允许 RF 覆盖」，使 RF 真正作为「不确定时的安全网」发挥作用。

## AI 工具提示词追溯

本课程设计全程使用 **Claude Code**（模型 deepseek-v4-pro）辅助开发。与 AI 的交流记录（提示词、AI 操作、结果摘要）以 JSON 形式留存于 [prompt/prompt_log.json](prompt/prompt_log.json)，并随每个阶段同步更新。

> 上下文压缩前备份规则：复制 `prompt/prompt_log.json` 为 `prompt/prompt_log_backup_<日期>.json` 后再追加新记录。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 1. 重新生成自建数据
python data/generate_dataset.py 120

# 2. 重新执行预处理 + 增广 + 划分
python data/preprocess.py

# 3. 训练算法模型（YOLO 检测 + 随机森林校验）
python scripts/train_yolo.py
python scripts/train_rf.py

# 4. 端到端推理 Demo（输出检测结果与可视化标注图）
python scripts/demo.py

# 5. 系统集成与功能测试（阶段五：test 集评估 + 接口全流程 + 前端）
python scripts/test_system.py

# 6. 启动后端服务（同时托管前端页面）
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 浏览器访问 http://127.0.0.1:8000/ 使用系统
```

依赖：Python 3.9+，见 [requirements.txt](requirements.txt)。

## 开发阶段规划

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段一 | 方案确认与数据准备 | ✅ 已完成（本阶段） |
| 阶段二 | 算法模块开发（预处理 / YOLO / 随机森林） | ✅ 已完成（本阶段） |
| 阶段三 | 后端服务接口开发（FastAPI + SQLite） | ✅ 已完成（本阶段） |
| 阶段四 | 前端 UI 页面开发 | ✅ 已完成（本阶段） |
| 阶段五 | 系统集成与功能测试 | ✅ 已完成（本阶段） |
| 阶段六 | 系统输出与文档整理 | 待开发 |

# 进度（progress）

## 2026-08-30（阶段一：数据准备，已完成）
- 已克隆 GitHub 仓库 jys642/jys 到 f:\JYS（main 分支同步）。
- 已读取 README.md、方案设计.md、选题说明.md、学习笔记.md，并提取任务书 PDF 文本核对课设要求。
- 已确认 Python 3.9 + numpy 1.26.4 + OpenCV 4.10.0 + PIL 10.4.0 环境可用。
- 已调研公开数据集，确认 Kaggle 数据集链接有效。
- 已建立规划文件 task_plan.md / findings.md / progress.md。
- 【数据来源】编写 data/generate_dataset.py，生成 120 张自建合成泡罩药板图像 + YOLO 标注（6 类缺陷，372 个标注，类别均衡），提交到 data/raw。
- 【数据来源】编写 data/README.md，说明公开数据集引用（Kaggle×2、GitHub×1）+ 自建数据说明 + 类别定义。
- 【数据预处理】编写 data/preprocess.py，实现灰度化/高斯滤波/CLAHE/ROI/尺寸归一化 + 增广（翻转/旋转/亮度）+ 80/10/10 划分，输出 480 张到 data/processed，生成 index.json（含类别、中文映射、预处理说明）。
- 【提示词追溯】创建 prompt/prompt_log.json，记录 3 条 AI 交流记录（克隆、数据准备、选题合规确认），并注明压缩前备份规则。
- 更新 README.md，同步数据来源、预处理、提示词追溯说明。

## 待办（下一阶段）
- 阶段二：算法模块开发（图像预处理模块 / YOLO 检测 / 随机森林校验）。
- 将 AI 提示词记录在每个阶段持续更新到 prompt/prompt_log.json。

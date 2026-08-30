# 任务计划（task_plan）

## 目标
完成「药板药片外观缺陷智能检测系统」课程设计**阶段一（数据准备）** 的三项考核内容：
1. 数据来源准备（/data 目录）
2. 数据预处理（预处理程序 + /data 预处理结果/索引）
3. AI 工具提示词追溯（/prompt 目录）

## 阶段与状态
- [x] 阶段 0：环境确认（Python/numpy/OpenCV/PIL 可用）+ 公开数据集调研
- [x] 阶段 1：建立规划文件
- [x] 阶段 2：生成自建合成数据集 → /data/raw（120 张 + YOLO 标注）
- [x] 阶段 3：编写数据来源说明 → /data/README.md
- [x] 阶段 4：预处理程序 + 预处理结果/索引 → /data/processed（480 张 + index.json）
- [x] 阶段 5：AI 提示词追溯 → /prompt/prompt_log.json
- [x] 阶段 6：更新 README.md

## 数据策略（已落地）
- 主数据源：引用公开数据集（Kaggle paracetamol-defect-dataset 等，链接有效），在 /data/README.md 与 README.md 说明。
- 自建小样本：脚本合成泡罩药板图像 + YOLO 标注，直接提交到 /data/raw，保证项目可运行、可演示。
- 缺陷类别（与选题说明一致）：missing 缺粒 / crack 裂片 / broken 破损 / flash 飞边 / stain 污渍 / empty 漏装。

## 后续阶段（待规划）
- 阶段二：算法模块开发（预处理 / YOLO 检测 / 随机森林校验）。
- 阶段三～六：后端、前端、集成测试、文档。

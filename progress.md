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

## 2026-08-31（阶段二：算法模块开发，已完成）
- 搭建 src/ 算法模块包：config（类别/路径/超参数）、preprocessing（预处理）、detection（YOLO 检测）、features（18 维特征提取）、classifier（随机森林校验）、pipeline（端到端管线）。
- 编写 scripts/train_yolo.py（按 index.json 构建 YOLO 数据集 + 微调）、scripts/train_rf.py（真实标注框提取特征 + 训练）、scripts/demo.py（端到端推理可视化）。
- 训练随机森林二次校验模型：1364 个缺陷样本、18 维特征、200 棵树，保存 models/rf_classifier.pkl；特征重要度 Top 为 hu1/circularity/min_gray/gray_range/std_gray。
- 训练 YOLO 检测模型：基于 yolo11n 预训练权重微调（yolo8n 已不被 ultralytics 8.4 支持且 GitHub API 限流，故改用 yolo11n），保存 models/best.pt。
- 补充 requirements.txt、models/README.md，.gitignore 忽略 runs/ 与 data/yolo_dataset/。
- 更新 README.md：新增「算法模块（阶段二）」章节，同步项目结构与快速开始命令。

## 2026-08-31（阶段三：后端服务接口开发，已完成）
- 搭建 app/ 后端包：main（FastAPI 应用 + 路由）、database（SQLite 建表 + 高层查询函数）、schemas（Pydantic 响应模型）。
- 实现 5 个接口：POST /api/detect（上传+检测+入库）、GET /api/records（分页列表）、GET /api/records/{id}（详情）、GET /api/statistics（统计）、GET /api/image/{name}（返回图片）。
- SQLite 两张表 records（检测记录）+ defects（缺陷明细），外键关联，质检结论（合格/不合格）自动判定。
- 管线增强：pipeline.run 返回 (结果, 预处理图)，便于在模型输入图上绘制标注框。
- 用 TestClient 自测全部接口通过（检测 3 处缺陷、入库、列表、详情、统计、图片返回、404 兜底）。
- .gitignore 忽略 app/detection.db 与 app/uploads/，requirements.txt 增加 python-multipart。

## 2026-08-31（阶段四：前端 UI 页面开发，已完成）
- 编写 frontend/ 单页应用：index.html（上传检测/历史记录/统计分析三视图）、style.css（医药工业质检风格）、app.js（fetch 对接后端 API）。
- 对接 4 个功能页面：图片上传 + 缺陷可视化、历史记录查询（含详情弹窗）、质检缺陷统计分析（卡片 + Chart.js 柱状图）。
- 修改 app/main.py：用 StaticFiles 托管前端静态资源，首页路由返回 index.html，同源部署免跨域。
- 用 TestClient 与真实 uvicorn 冒烟测试：首页/静态资源/检测/统计/标注图接口全部正常。

## 待办（下一阶段）
- 阶段五：系统集成与功能测试（全流程联调 + 测试样本验证）。
- 阶段六：系统输出与文档整理。
- 将 AI 提示词记录在每个阶段持续更新到 prompt/prompt_log.json。

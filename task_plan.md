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

## 阶段二（算法模块开发，已完成）
- [x] 搭建 src/ 算法模块包（config / preprocessing / detection / features / classifier / pipeline）
- [x] 训练随机森林二次校验模型 → models/rf_classifier.pkl（1364 样本 / 18 特征 / 200 棵树）
- [x] 训练 YOLO 检测模型（yolo11n 微调）→ models/best.pt
- [x] 端到端推理 demo（scripts/demo.py）
- [x] requirements.txt / models 说明 / .gitignore / README 同步

## 阶段三（后端服务接口开发，已完成）
- [x] 搭建 app/ 后端包（main / database / schemas）
- [x] 实现图片上传+检测、历史查询、详情、统计、图片返回 5 个接口
- [x] SQLite 建表（records + defects）与持久化
- [x] TestClient 全接口自测通过
- [x] .gitignore / requirements.txt / README 同步

## 阶段四（前端 UI 页面开发，已完成）
- [x] 编写 frontend/ 单页应用（index.html / style.css / app.js）
- [x] 实现上传检测、历史记录、统计分析三大页面
- [x] app/main.py 托管前端静态资源（StaticFiles + 首页路由）
- [x] TestClient + uvicorn 端到端联调通过

## 阶段五（系统集成与功能测试，已完成）
- [x] 编写 scripts/test_system.py（test 集端到端评估 + 后端接口全流程 + 前端静态资源）
- [x] 48 张 test 样本 IoU 匹配评估（Precision/Recall/F1 + 类别级 + YOLO/RF 对比）
- [x] 修复随机森林融合策略缺陷（引入 YOLO_CONF_GATE，RF 改为 YOLO 不确定时的安全网）
- [x] 后端 5 接口 + 前端静态资源全流程联调通过
- [x] 测试产物清理（detection.db / uploads 不入库）

## 后续阶段（待规划）
- 阶段六：系统输出与文档整理。

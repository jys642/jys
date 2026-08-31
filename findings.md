# 调研发现（findings）

## 项目定位
- 题目：药板药片外观缺陷智能检测系统（B/S 架构）
- 技术栈：机器视觉预处理（OpenCV）＋ YOLO 目标检测（PyTorch）＋ 随机森林二次校验（Scikit-learn）＋ FastAPI/SQLite 后端 ＋ HTML/Bootstrap 前端
- 缺陷类别（6 类）：缺粒、裂片、破损、飞边、污渍、漏装

## 公开数据集调研结论
- Kaggle `arthurkray/paracetamol-defect-dataset`：66.87MB / 3500 张 / 200×200 / 5 类（normal、broken、colored、stained、unknown），真实有效。
- Kaggle `pudpawat/pill-defect-dataset`：30.49MB / 363 张 / defect+normal，真实有效。
- GitHub `PerceptiLabs/Pill-Defects`：defect/normal + data.csv，真实有效（上述数据集的镜像）。
- CSDN/GitCode 泡罩专用数据集（622/3685/6987 张，YOLO/COCO/VOC）：泡罩语义更贴合，但走 CSDN 下载通道，需积分/付费，稳定性差，不作为主引用。
- 未发现 HuggingFace / ModelScope 上对应泡罩缺陷公开数据集。

## 决策
- 主引用 Kaggle 公开数据集（链接有效、可下载），作为大规模训练数据来源。
- 同时自建小规模合成数据集（脚本生成 + YOLO 标注）直接提交 /data，保证课程设计可运行与可复现。

## 阶段二算法调研结论
- ultralytics 8.4.91 已把「命名资产」迁移到 YOLO11/SAM/RT-DETR 系列，`yolo8n.pt` 不在直连清单内，下载需走 GitHub API（本环境 IP 触发 403 限流）；`yolo11n.pt` 在清单内，走直连下载 `.../releases/download/v8.4.0/yolo11n.pt`（HTTP 200）。
- 决策：YOLO 检测模块采用 **yolo11n** 预训练权重微调（比 v8 更新、更快、精度相当），文档同步说明。
- 随机森林 18 维特征中，几何（hu1、circularity、perimeter、fill_ratio）与灰度（min_gray、gray_range、std_gray）贡献最大，符合「缺陷形态/灰度差异」直觉。

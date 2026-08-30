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

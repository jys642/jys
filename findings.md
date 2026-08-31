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

## 阶段三后端调研结论
- FastAPI 文件上传（UploadFile）需 python-multipart 依赖；用 starlette TestClient（依赖 httpx）可脱离真实服务器对接口做全链路自测。
- SQLite 用标准库 sqlite3 + Row 工厂即可满足课程设计轻量持久化，无需引入 ORM。

## 阶段四前端调研结论
- 前端采用 Bootstrap 5 + 原生 fetch + Chart.js；后端用 StaticFiles 托管前端实现同源部署，避免 CORS 配置与跨域问题。
- 单页应用（SPA）三视图切换比多页面更适合质检工作流：上传→看结果→查历史→看统计在一个入口内完成。

## 阶段五系统集成测试结论
- **发现并修复了随机森林融合策略缺陷**：原融合逻辑「RF 置信度 ≥ 0.5 就用 RF 覆盖 YOLO」过于激进。在 test 集上诊断发现——RF 改对 3 处（YOLO 置信度低 0.26~0.31 时）、改错 8 处（YOLO 已高置信 0.996~0.999，RF 中等置信 0.48~0.76 却覆盖），导致 F1 从 0.984 掉到 0.944。
- **修复方案**：引入 `YOLO_CONF_GATE=0.5` 门限，改为「YOLO 自信 → 信 YOLO；YOLO 不确定且 RF 自信 → 信 RF」。修复后 RF 不再拖累精度，test 集最终 F1=0.984、Recall=1.0、Precision=0.969。
- **经验**：多模型融合不能只看单模型置信度，要按「主模型是否自信」决定是否允许辅助模型覆盖；辅助模型应定位为「主模型不确定时的安全网」而非无条件纠错。
- test 集 48 张全部走通「预处理→YOLO→RF→入库→历史→详情→统计→图片」全流程，后端 5 接口 + 前端静态资源均正常。

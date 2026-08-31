# 模型说明（/models）

存放「药板药片外观缺陷智能检测系统」训练出的算法模型权重。

| 文件 | 模型 | 用途 | 生成方式 |
|---|---|---|---|
| `best.pt` | YOLO11n（微调） | 缺陷目标检测：定位缺陷 + 初步类别 | `python scripts/train_yolo.py` |
| `rf_classifier.pkl` | 随机森林（200 棵树） | 二次校验：对检测结果再分类、修正误检 | `python scripts/train_rf.py` |

> 训练基于自建合成数据集 `data/processed`（train/val 划分，YOLO 格式标注，6 类缺陷）。模型在 CPU 上训练，规模小，便于课程设计复现与演示。

## 重新训练

```bash
# 训练 YOLO 检测模型（基于 yolo11n 预训练权重微调）
python scripts/train_yolo.py

# 训练随机森林二次校验模型
python scripts/train_rf.py
```

训练产物：YOLO 的中间训练日志在 `runs/`（已 gitignore），最佳权重复制到本目录 `best.pt`。

## 端到端推理

```bash
python scripts/demo.py               # 对 test 集第一张图推理
python scripts/demo.py blister_0001  # 指定图像
```

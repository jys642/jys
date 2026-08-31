# -*- coding: utf-8 -*-
"""
全局配置 —— 药板药片外观缺陷智能检测系统
================================================================
统一管理缺陷类别、图像尺寸、数据/模型路径与训练超参数，
供各算法模块引用，避免散落硬编码。
================================================================
"""
import os

# ---------------------------------------------------------------- 缺陷类别
# 顺序与 data/raw/classes.txt 一致，也与《选题说明》严格对齐
CLASSES = ["missing", "crack", "broken", "flash", "stain", "empty"]
CLASS_CN = {
    "missing": "缺粒", "crack": "裂片", "broken": "破损",
    "flash": "飞边", "stain": "污渍", "empty": "漏装",
}
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------- 图像
IMG_SIZE = 640          # 输入图像尺寸（预处理统一缩放）

# ---------------------------------------------------------------- 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
DATA_DIR = os.path.join(ROOT, "data")
RAW_IMG = os.path.join(DATA_DIR, "raw", "images")
RAW_LBL = os.path.join(DATA_DIR, "raw", "labels")
PROC_IMG = os.path.join(DATA_DIR, "processed", "images")
PROC_LBL = os.path.join(DATA_DIR, "processed", "labels")
INDEX_JSON = os.path.join(DATA_DIR, "processed", "index.json")
YOLO_DATASET = os.path.join(DATA_DIR, "yolo_dataset")  # 训练用 YOLO 目录（脚本生成）

MODELS_DIR = os.path.join(ROOT, "models")
YOLO_MODEL = os.path.join(MODELS_DIR, "best.pt")            # YOLO 训练出的最佳权重
RF_MODEL = os.path.join(MODELS_DIR, "rf_classifier.pkl")    # 随机森林二次校验模型

# ---------------------------------------------------------------- 训练超参数
SEED = 42
YOLO_EPOCHS = 50
YOLO_IMGSZ = 640
YOLO_BATCH = 8
YOLO_CONF = 0.25      # YOLO 推理置信度阈值
YOLO_IOU = 0.45       # NMS IoU 阈值
RF_CONF = 0.5         # 随机森林二次校验置信度阈值（低于则保留 YOLO 结果）
YOLO_CONF_GATE = 0.5  # YOLO 置信度门限：低于此值才允许 RF 覆盖（YOLO 自信时优先信 YOLO）

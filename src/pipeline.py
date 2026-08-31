# -*- coding: utf-8 -*-
"""
端到端检测管线 —— 组合三大算法模块
================================================================
流水线：图像预处理 → YOLO 缺陷检测 → 随机森林二次校验 → 最终质检结果。

二次校验策略：
    对每个 YOLO 检测框提取特征，用随机森林再分类；
    - 若 RF 置信度 ≥ 阈值，以 RF 类别修正 YOLO 初步类别（修正误检）；
    - 否则保留 YOLO 结果（RF 不确定时不强行覆盖）。
================================================================
"""
from src.config import RF_CONF
from src.classifier import DefectClassifier
from src.detection import YoloDetector
from src.features import extract_features
from src.preprocessing import preprocess


class DetectionPipeline:
    """整合预处理 + YOLO 检测 + 随机森林校验的完整推理管线。"""

    def __init__(self, yolo_model_path, rf_model_path):
        self.detector = YoloDetector(yolo_model_path)
        self.classifier = DefectClassifier(rf_model_path)

    def run(self, img_bgr, yolo_conf=0.25, rf_conf=RF_CONF):
        """输入原始 BGR 图像，返回 (检测结果列表, 预处理后图像)。

        检测结果列表项：[{
            "bbox": [x1,y1,x2,y2],
            "yolo_class": str, "yolo_conf": float,
            "rf_class": str, "rf_conf": float,
            "final_class_id": int, "final_class": str, "conf": float,
        }, ...]
        预处理后图像为 640×640（模型实际输入），供可视化标注框对齐。
        """
        proc = preprocess(img_bgr)                       # 1. 预处理
        dets = self.detector.detect(proc, conf=yolo_conf)  # 2. YOLO 检测

        results = []
        for d in dets:
            feat = extract_features(proc, d["bbox"])     # 3. 特征提取
            rf = self.classifier.predict(feat)           # 4. RF 二次校验

            if rf["proba"] >= rf_conf:                   # 5. 校验结果融合
                final_cls, final_name = rf["class_id"], rf["class_name"]
            else:
                final_cls, final_name = d["class_id"], d["class_name"]

            results.append({
                "bbox": d["bbox"],
                "yolo_class": d["class_name"],
                "yolo_conf": round(d["conf"], 4),
                "rf_class": rf["class_name"],
                "rf_conf": round(rf["proba"], 4),
                "final_class_id": final_cls,
                "final_class": final_name,
                "conf": round(d["conf"], 4),
            })
        return results, proc

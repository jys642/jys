# -*- coding: utf-8 -*-
"""
YOLO 缺陷检测模块 —— 对应《方案设计》三大算法模块之二（深度学习目标检测）
================================================================
封装 ultralytics YOLO 模型，提供训练与推理接口。
推理输出药片缺陷的定位框、初步类别与置信度，供后续随机森林二次校验。
================================================================
"""
import os

from src.config import CLASSES, IMG_SIZE, YOLO_CONF, YOLO_IOU


class YoloDetector:
    """YOLO 检测器：加载权重、执行推理、输出结构化检测结果。"""

    def __init__(self, model_path):
        from ultralytics import YOLO  # 延迟导入，未安装 ultralytics 时不阻塞其他模块
        self.model = YOLO(model_path)
        self.names = self.model.names  # {0: 'missing', 1: 'crack', ...}

    def detect(self, img_bgr, conf=YOLO_CONF, iou=YOLO_IOU):
        """对预处理后的 BGR 图像推理，返回检测结果列表。

        返回：[{"bbox": [x1,y1,x2,y2], "class_id": int, "class_name": str, "conf": float}, ...]
        """
        results = self.model.predict(img_bgr, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls,
                    "class_name": self.names.get(cls, CLASSES[cls] if cls < len(CLASSES) else str(cls)),
                    "conf": conf,
                })
        return dets

    @staticmethod
    def train(data_yaml, epochs=50, imgsz=640, batch=8, project="runs", name="detect"):
        """训练 YOLO 模型（基于 yolo11n 预训练权重微调），返回最佳权重路径。

        训练产物路径以 ultralytics 实际写入的 trainer.save_dir 为准，
        避免手工拼接 project/name（detect 任务默认 project 为 runs/detect 会嵌套）。
        """
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                    project=project, name=name, seed=42, verbose=True)
        save_dir = model.trainer.save_dir
        best = getattr(model.trainer, "best", None) or os.path.join(save_dir, "weights", "best.pt")
        return best

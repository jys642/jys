# -*- coding: utf-8 -*-
"""
随机森林二次校验模块 —— 对应《方案设计》三大算法模块之三（机器学习集成分类）
================================================================
提取缺陷区域特征，用随机森林对 YOLO 初步检测结果做二次分类校验：
修正误检类别、过滤低置信度样本，输出更可靠的最终质检结论。
================================================================
"""
import os

import joblib

from src.config import CLASSES


class DefectClassifier:
    """随机森林分类器：加载模型、对单个特征向量二次校验。"""

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        """输入单个特征向量（长度与 src.features.FEATURE_NAMES 一致）。

        返回：{"class_id": int, "class_name": str, "proba": float, "proba_all": [..]}
        """
        proba = self.model.predict_proba([features])[0]
        cid = int(self.model.predict([features])[0])
        return {
            "class_id": cid,
            "class_name": CLASSES[cid],
            "proba": float(proba[cid]),
            "proba_all": [float(p) for p in proba],
        }

    @staticmethod
    def train(X, y, model_path):
        """训练随机森林模型并保存。X: 特征矩阵, y: 类别标签。"""
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X, y)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(clf, model_path)
        return clf

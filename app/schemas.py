# -*- coding: utf-8 -*-
"""Pydantic 响应模型 —— 接口返回结构定义（自动生成 OpenAPI 文档）。"""
from typing import List, Optional

from pydantic import BaseModel


class DefectItem(BaseModel):
    """单个缺陷的完整校验结果。"""
    class_id: int
    class_name: str
    class_cn: str
    confidence: float
    bbox: List[float]
    yolo_class: str
    yolo_conf: float
    rf_class: str
    rf_conf: float


class DetectionResult(BaseModel):
    """一次检测的返回结果。"""
    record_id: int
    image_name: str
    annotated_path: str
    defect_count: int
    conclusion: str
    defects: List[DefectItem]


class RecordSummary(BaseModel):
    """历史记录摘要。"""
    id: int
    image_name: str
    created_at: str
    defect_count: int
    conclusion: str


class RecordDetail(BaseModel):
    """单条记录详情（含缺陷明细）。"""
    id: int
    image_name: str
    image_path: str
    created_at: str
    defect_count: int
    conclusion: str
    defects: List[DefectItem]


class ClassCount(BaseModel):
    """单个缺陷类别的统计计数。"""
    class_id: int
    class_name: str
    class_cn: str
    count: int


class Statistics(BaseModel):
    """缺陷统计结果。"""
    total_records: int
    total_defects: int
    by_class: List[ClassCount]

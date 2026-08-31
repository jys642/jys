# -*- coding: utf-8 -*-
"""
FastAPI 后端服务 —— 药板药片外观缺陷智能检测系统
================================================================
对外接口（对应《方案设计》后端接口清单）：
    POST /api/detect        图片上传 + 缺陷检测（预处理 + YOLO + 随机森林校验）+ 入库
    GET  /api/records       历史检测记录列表（分页）
    GET  /api/records/{id}  单条记录详情（含缺陷明细）
    GET  /api/statistics    缺陷类型统计
    GET  /api/image/{name}  返回已上传图像（供前端展示）

启动：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
================================================================
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.config import CLASS_CN, RF_MODEL, ROOT, YOLO_MODEL
from src.pipeline import DetectionPipeline
from app import database as db
from app.schemas import (DefectItem, DetectionResult, RecordDetail,
                         RecordSummary, Statistics)

UPLOAD_DIR = os.path.join(ROOT, "app", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="药板药片外观缺陷智能检测系统 API", version="1.0.0")

# 跨域：阶段四前端页面跨端口调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测管线（加载 YOLO + 随机森林模型）
pipeline = DetectionPipeline(YOLO_MODEL, RF_MODEL)


@app.on_event("startup")
def on_startup():
    db.init_db()


def draw_boxes(img, defects):
    """在图像上绘制缺陷框与中文类别标签，返回标注图。"""
    annotated = img.copy()
    for d in defects:
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cn = CLASS_CN.get(d["final_class"], d["final_class"])
        label = f"{cn} {d['conf']:.2f}"
        cv2.putText(annotated, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return annotated


@app.get("/")
def root():
    return {
        "service": "药板药片外观缺陷智能检测系统 API",
        "endpoints": {
            "POST /api/detect": "上传图片并检测缺陷",
            "GET /api/records": "历史检测记录（分页）",
            "GET /api/records/{id}": "单条记录详情",
            "GET /api/statistics": "缺陷类型统计",
            "GET /api/image/{name}": "获取已上传图像",
        },
    }


@app.post("/api/detect", response_model=DetectionResult)
def detect(file: UploadFile = File(...)):
    """上传药板图片 → 预处理 + YOLO 检测 + 随机森林校验 → 入库并返回结果。"""
    # 1. 读取并解码图片
    data = file.file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片，请上传 JPG/PNG 等格式")

    # 2. 保存原图
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    image_name = ts + ext
    image_path = os.path.join(UPLOAD_DIR, image_name)
    cv2.imwrite(image_path, img)

    # 3. 执行检测（预处理 + YOLO + 随机森林）
    results, proc = pipeline.run(img)

    # 4. 生成标注图（在模型输入图上绘制）
    annotated_name = ts + "_annotated.jpg"
    annotated_path = os.path.join(UPLOAD_DIR, annotated_name)
    cv2.imwrite(annotated_path, draw_boxes(proc, results))

    # 5. 持久化
    record_id = db.save_detection(image_name, image_path,
                                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), results)

    # 6. 返回结构化结果
    return DetectionResult(
        record_id=record_id,
        image_name=image_name,
        annotated_path=f"/api/image/{annotated_name}",
        defect_count=len(results),
        conclusion="不合格" if results else "合格",
        defects=[DefectItem(
            class_id=r["final_class_id"],
            class_name=r["final_class"],
            class_cn=CLASS_CN.get(r["final_class"], r["final_class"]),
            confidence=r["conf"],
            bbox=r["bbox"],
            yolo_class=r["yolo_class"],
            yolo_conf=r["yolo_conf"],
            rf_class=r["rf_class"],
            rf_conf=r["rf_conf"],
        ) for r in results],
    )


@app.get("/api/records", response_model=list[RecordSummary])
def list_records(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    """历史检测记录列表（按时间倒序）。"""
    return db.list_records(limit=limit, offset=offset)


@app.get("/api/records/{record_id}", response_model=RecordDetail)
def get_record(record_id: int):
    """单条记录详情（含缺陷明细）。"""
    detail = db.get_record_detail(record_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    return detail


@app.get("/api/statistics", response_model=Statistics)
def statistics():
    """缺陷类型统计。"""
    return db.get_statistics()


@app.get("/api/image/{name}")
def get_image(name: str):
    """返回已上传图像（原图或标注图）。"""
    safe = os.path.basename(name)  # 防止路径穿越
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(path)

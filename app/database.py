# -*- coding: utf-8 -*-
"""
SQLite 数据库模块 —— 检测记录与缺陷明细持久化
================================================================
两张表：
    records  检测记录（图片路径、时间、缺陷数、质检结论）
    defects  缺陷明细（类别、置信度、框坐标、YOLO/RF 校验结果）

对外提供 save_detection / list_records / get_record_detail / get_statistics，
main.py 只调用这些高层函数，不直接接触 SQL。
================================================================
"""
import os
import sqlite3

from src.config import CLASSES, CLASS_CN, ROOT

DB_PATH = os.path.join(ROOT, "app", "detection.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name   TEXT NOT NULL,
    image_path   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    defect_count INTEGER NOT NULL DEFAULT 0,
    conclusion   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS defects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   INTEGER NOT NULL,
    class_id    INTEGER NOT NULL,
    class_name  TEXT NOT NULL,
    class_cn    TEXT NOT NULL,
    confidence  REAL NOT NULL,
    bbox_x1     REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    yolo_class  TEXT, yolo_conf REAL,
    rf_class    TEXT, rf_conf REAL,
    FOREIGN KEY (record_id) REFERENCES records (id)
);
"""


def get_conn():
    """打开数据库连接（Row 工厂便于按列名访问）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """建表（幂等）。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def save_detection(image_name, image_path, created_at, defects):
    """保存一条检测记录及其缺陷明细，返回记录 id。"""
    conclusion = "不合格" if defects else "合格"
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO records (image_name, image_path, created_at, defect_count, conclusion) "
            "VALUES (?, ?, ?, ?, ?)",
            (image_name, image_path, created_at, len(defects), conclusion),
        )
        record_id = cur.lastrowid
        for d in defects:
            b = d["bbox"]
            conn.execute(
                "INSERT INTO defects (record_id, class_id, class_name, class_cn, confidence, "
                "bbox_x1, bbox_y1, bbox_x2, bbox_y2, yolo_class, yolo_conf, rf_class, rf_conf) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record_id, d["final_class_id"], d["final_class"],
                 CLASS_CN.get(d["final_class"], d["final_class"]), d["conf"],
                 b[0], b[1], b[2], b[3],
                 d["yolo_class"], d["yolo_conf"], d["rf_class"], d["rf_conf"]),
            )
        conn.commit()
        return record_id
    finally:
        conn.close()


def list_records(limit=20, offset=0):
    """分页返回检测记录摘要（不含缺陷明细）。"""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, image_name, created_at, defect_count, conclusion "
            "FROM records ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_record_detail(record_id):
    """返回单条记录及其缺陷明细；不存在则返回 None。"""
    conn = get_conn()
    try:
        rec = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        if rec is None:
            return None
        defects = conn.execute(
            "SELECT class_id, class_name, class_cn, confidence, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            "yolo_class, yolo_conf, rf_class, rf_conf "
            "FROM defects WHERE record_id = ?", (record_id,)
        ).fetchall()
        detail = dict(rec)
        detail["defects"] = [
            {
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "class_cn": d["class_cn"],
                "confidence": d["confidence"],
                "bbox": [d["bbox_x1"], d["bbox_y1"], d["bbox_x2"], d["bbox_y2"]],
                "yolo_class": d["yolo_class"],
                "yolo_conf": d["yolo_conf"],
                "rf_class": d["rf_class"],
                "rf_conf": d["rf_conf"],
            } for d in defects
        ]
        return detail
    finally:
        conn.close()


def get_statistics():
    """统计缺陷类型分布（按 6 类，含零计数）与总量。"""
    conn = get_conn()
    try:
        total_records = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        total_defects = conn.execute("SELECT COUNT(*) FROM defects").fetchone()[0]
        counts = {r["class_name"]: r["c"] for r in conn.execute(
            "SELECT class_name, COUNT(*) AS c FROM defects GROUP BY class_name"
        ).fetchall()}
        by_class = [
            {"class_id": i, "class_name": c, "class_cn": CLASS_CN.get(c, c), "count": counts.get(c, 0)}
            for i, c in enumerate(CLASSES)
        ]
        return {"total_records": total_records, "total_defects": total_defects, "by_class": by_class}
    finally:
        conn.close()

# -*- coding: utf-8 -*-
"""
系统集成与功能测试 —— 对应《方案设计》阶段五
================================================================
覆盖三部分：
    1. 测试样本验证：对 test 集全部样本执行「预处理 + YOLO + 随机森林」端到端推理，
       与真实标注框做 IoU 匹配，计算精确率 / 召回率 / F1（含类别级与 YOLO/RF 对比）。
    2. 后端接口全流程：用 TestClient 走通「上传检测 → 入库 → 历史 → 详情 → 统计 → 图片返回」。
    3. 前端静态资源：校验首页与 css/js 可正常返回。

用法：
    python scripts/test_system.py            # 跑全部 test 集（48 张）
    python scripts/test_system.py --limit 10 # 只跑前 10 张（快速验证）
================================================================
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from fastapi.testclient import TestClient

from src.config import (CLASS_CN, CLASSES, INDEX_JSON, NUM_CLASSES,
                        PROC_IMG, PROC_LBL, RF_MODEL, YOLO_MODEL)
from src.pipeline import DetectionPipeline

IOU_THRESHOLD = 0.5  # 检测框匹配 IoU 阈值


# ---------------------------------------------------------------- 工具函数
def iou(box1, box2):
    """两个 [x1,y1,x2,y2] 框的 IoU。"""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_gt(name):
    """读取一张 test 图像的 YOLO 格式真值标注，转成 640×640 像素坐标。"""
    label_file = os.path.join(PROC_LBL, name.replace(".jpg", ".txt"))
    boxes = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            x1 = (cx - w / 2) * 640; y1 = (cy - h / 2) * 640
            x2 = (cx + w / 2) * 640; y2 = (cy + h / 2) * 640
            boxes.append({"class_id": cls, "bbox": [x1, y1, x2, y2]})
    return boxes


def match(preds, gts):
    """贪心 IoU 匹配，返回 (tp, fp, fn, per_class)。class_id 取各结果里给出的类别。"""
    matched = [False] * len(preds)
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(NUM_CLASSES)}
    tp = fp = fn = 0

    for gt in gts:
        best_iou, best_idx = 0.0, -1
        for i, p in enumerate(preds):
            if matched[i]:
                continue
            v = iou(p["bbox"], gt["bbox"])
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_idx >= 0 and best_iou >= IOU_THRESHOLD:
            matched[best_idx] = True
            if preds[best_idx]["class_id"] == gt["class_id"]:
                tp += 1
                per_class[gt["class_id"]]["tp"] += 1
            else:  # 位置对但类别错：真值类别记为漏检，预测类别记为误检
                fn += 1; fp += 1
                per_class[gt["class_id"]]["fn"] += 1
                per_class[preds[best_idx]["class_id"]]["fp"] += 1
        else:
            fn += 1
            per_class[gt["class_id"]]["fn"] += 1

    for i, p in enumerate(preds):
        if not matched[i]:
            fp += 1
            per_class[p["class_id"]]["fp"] += 1
    return tp, fp, fn, per_class


def summarize(tp, fp, fn, per_class, title):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\n  {title}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  |  Precision={precision:.3f}  "
          f"Recall={recall:.3f}  F1={f1:.3f}")
    print("    类别            P      R     F1   (TP/FP/FN)")
    for c in range(NUM_CLASSES):
        s = per_class[c]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"    {CLASS_CN[CLASSES[c]]:<6}({CLASSES[c]:<7}) {p:.3f} {r:.3f} {f:.3f}  "
              f"({s['tp']}/{s['fp']}/{s['fn']})")
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------- 测试样本验证
def test_evaluation(limit):
    print("=" * 72)
    print("【1】测试样本验证：test 集端到端推理 + 真值 IoU 匹配")
    print("=" * 72)
    index = json.load(open(INDEX_JSON, encoding="utf-8"))
    names = index["splits"]["test"][:limit]

    pipe = DetectionPipeline(YOLO_MODEL, RF_MODEL)

    final_tp = final_fp = final_fn = 0
    yolo_tp = yolo_fp = yolo_fn = 0
    final_pc = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(NUM_CLASSES)}
    yolo_pc = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(NUM_CLASSES)}

    for i, name in enumerate(names, 1):
        img = cv2.imread(os.path.join(PROC_IMG, name))
        gts = load_gt(name)
        results, _ = pipe.run(img)

        # 最终结果（YOLO + RF 校验）
        final_preds = [{"class_id": r["final_class_id"], "bbox": r["bbox"]} for r in results]
        t, f, fn, pc = match(final_preds, gts)
        final_tp += t; final_fp += f; final_fn += fn
        for c in range(NUM_CLASSES):
            final_pc[c]["tp"] += pc[c]["tp"]
            final_pc[c]["fp"] += pc[c]["fp"]
            final_pc[c]["fn"] += pc[c]["fn"]

        # 仅 YOLO（不含 RF 二次校验）：pipeline 结果里已有 yolo_class，映射回 class_id
        yolo_preds = [{"class_id": CLASSES.index(r["yolo_class"]), "bbox": r["bbox"]}
                      for r in results]
        t, f, fn, pc = match(yolo_preds, gts)
        yolo_tp += t; yolo_fp += f; yolo_fn += fn
        for c in range(NUM_CLASSES):
            yolo_pc[c]["tp"] += pc[c]["tp"]
            yolo_pc[c]["fp"] += pc[c]["fp"]
            yolo_pc[c]["fn"] += pc[c]["fn"]

        if i % 10 == 0 or i == len(names):
            print(f"  已评估 {i}/{len(names)} 张…")

    print(f"\n  共评估 {len(names)} 张 test 样本。")
    final_metric = summarize(final_tp, final_fp, final_fn, final_pc, "YOLO + 随机森林校验（最终结果）")
    yolo_metric = summarize(yolo_tp, yolo_fp, yolo_fn, yolo_pc, "仅 YOLO（未二次校验）")
    return {"final": final_metric, "yolo": yolo_metric}


# ---------------------------------------------------------------- 后端 + 前端
def test_backend_and_frontend():
    print("\n" + "=" * 72)
    print("【2】后端接口全流程 + 前端静态资源")
    print("=" * 72)
    from app import database as db
    from app.main import app

    db.init_db()  # 确保表存在（TestClient 上下文管理器也会触发 startup，双保险）

    # 准备一张测试图
    sample = os.path.join(PROC_IMG, "blister_0112.jpg")
    with open(sample, "rb") as f:
        img_bytes = f.read()

    with TestClient(app) as client:
        # 2.1 上传检测
        r = client.post("/api/detect",
                        files={"file": ("blister_0112.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 200, f"detect 失败：{r.status_code}"
        data = r.json()
        assert "record_id" in data and "defects" in data and "annotated_path" in data
        rid = data["record_id"]
        print(f"  POST /api/detect        -> 200，record_id={rid}，"
              f"缺陷 {data['defect_count']} 处，结论「{data['conclusion']}」")

        # 2.2 历史记录
        r = client.get("/api/records")
        assert r.status_code == 200 and len(r.json()) >= 1
        print(f"  GET  /api/records       -> 200，{len(r.json())} 条记录")

        # 2.3 详情
        r = client.get(f"/api/records/{rid}")
        assert r.status_code == 200 and r.json()["id"] == rid
        assert len(r.json()["defects"]) == data["defect_count"]
        print(f"  GET  /api/records/{rid}  -> 200，含 {len(r.json()['defects'])} 条缺陷明细")

        # 2.4 统计
        r = client.get("/api/statistics")
        assert r.status_code == 200
        stat = r.json()
        assert stat["total_records"] >= 1 and len(stat["by_class"]) == NUM_CLASSES
        print(f"  GET  /api/statistics    -> 200，累计 {stat['total_records']} 条 / "
              f"{stat['total_defects']} 处缺陷")

        # 2.5 图片返回（原图 + 标注图）
        r = client.get(f"/api/image/{data['image_name']}")
        assert r.status_code == 200 and len(r.content) > 0
        annotated = data["annotated_path"].rsplit("/", 1)[-1]
        r2 = client.get(f"/api/image/{annotated}")
        assert r2.status_code == 200 and len(r2.content) > 0
        print("  GET  /api/image/{name}  -> 200（原图 + 标注图）")

        # 2.6 404 兜底
        r = client.get("/api/records/999999")
        assert r.status_code == 404
        r = client.get("/api/image/not_exist.jpg")
        assert r.status_code == 404
        print("  GET  不存在资源         -> 404 兜底正常")

        # 2.7 前端静态资源
        r = client.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        r = client.get("/static/style.css")
        assert r.status_code == 200
        r = client.get("/static/app.js")
        assert r.status_code == 200
        print("  GET  /  + /static/*     -> 首页 + css/js 均 200")


# ---------------------------------------------------------------- 入口
def main():
    parser = argparse.ArgumentParser(description="系统集成与功能测试")
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 张 test 样本（0=全部）")
    args = parser.parse_args()

    limit = args.limit or 48
    test_evaluation(limit)
    test_backend_and_frontend()

    print("\n" + "=" * 72)
    print("全部测试通过 ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()

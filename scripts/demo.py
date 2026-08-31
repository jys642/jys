# -*- coding: utf-8 -*-
"""
端到端推理 Demo
================================================================
对 test 划分中的一张药板图像执行「预处理 + YOLO 检测 + 随机森林二次校验」，
在控制台打印结构化检测结果，并输出可视化标注图到 runs/demo/。

用法：
    python scripts/demo.py                 # 使用 test 集第一张图
    python scripts/demo.py blister_0001    # 指定图像名（可省略 .jpg）
================================================================
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLASS_CN, INDEX_JSON, PROC_IMG, RF_MODEL, YOLO_MODEL
from src.pipeline import DetectionPipeline

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "demo")


def draw_results(img, results):
    """在图像上绘制检测框与类别标签。"""
    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cn = CLASS_CN.get(r["final_class"], r["final_class"])
        label = f"{cn} {r['conf']:.2f}"
        cv2.putText(img, label, (x1, max(10, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def main():
    index = json.load(open(INDEX_JSON, encoding="utf-8"))
    test_names = index["splits"]["test"]

    name = sys.argv[1] if len(sys.argv) > 1 else test_names[0]
    if not name.endswith(".jpg"):
        name += ".jpg"

    img_path = os.path.join(PROC_IMG, name)
    img = cv2.imread(img_path)
    if img is None:  # 指定名不存在则回退到 test 集第一张
        name, img = test_names[0], cv2.imread(os.path.join(PROC_IMG, test_names[0]))

    pipe = DetectionPipeline(YOLO_MODEL, RF_MODEL)
    results, proc = pipe.run(img)

    print(f"图像：{name}")
    print(f"检出缺陷 {len(results)} 处：")
    for r in results:
        cn = CLASS_CN.get(r["final_class"], r["final_class"])
        print(f"  {cn:>4}  置信度 {r['conf']:.3f}  "
              f"(YOLO={r['yolo_class']}/{r['yolo_conf']:.3f}, "
              f"RF={r['rf_class']}/{r['rf_conf']:.3f})  "
              f"框={[round(v) for v in r['bbox']]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, name)
    cv2.imwrite(out_path, draw_results(proc, results))
    print(f"可视化结果：{out_path}")


if __name__ == "__main__":
    main()

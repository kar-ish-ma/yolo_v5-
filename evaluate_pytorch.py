import torch
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def percentile_stats(times):
    return {
        "mean": float(np.mean(times)),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }


def evaluate():
    model = YOLO("yolov5s.pt")

    print("Running validation on coco128...")
    results = model.val(data="src/data/coco128.yaml", imgsz=640)

    metrics = results.results_dict

    # Extract metrics safely as Python floats
    precision = float(metrics.get("metrics/precision(B)", 0))
    recall = float(metrics.get("metrics/recall(B)", 0))
    mAP50 = float(metrics.get("metrics/mAP50(B)", 0))
    mAP5095 = float(metrics.get("metrics/mAP50-95(B)", 0))

    f1 = float((2 * precision * recall) / (precision + recall + 1e-6))

    print("Profiling latency...")
    latencies = []

    for _ in range(50):
        start = time.time()
        model("src/data/images/bus.jpg")
        latencies.append((time.time() - start) * 1000)

    latency = percentile_stats(latencies)

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "coco128",
        "mAP_50": mAP50,
        "mAP_50_95": mAP5095,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": {
            "mean": float(latency["mean"]),
            "p50": float(latency["p50"]),
            "p95": float(latency["p95"]),
            "p99": float(latency["p99"]),
        }
    }

    Path("results").mkdir(exist_ok=True)

    with open("results/pytorch_metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("Saved: results/pytorch_metrics.json")


if __name__ == "__main__":
    evaluate()
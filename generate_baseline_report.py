import json
from pathlib import Path
from datetime import datetime


def load_metrics(file):
    if Path(file).exists():
        with open(file, encoding="utf-8") as f:
            return json.load(f)
    return None


def format_latency(latency):
    if not latency:
        return "Latency data not available."

    return f"""
Latency (ms):
Mean: {latency.get("mean")}
P50:  {latency.get("p50")}
P95:  {latency.get("p95")}
P99:  {latency.get("p99")}
"""


def pytorch_section(data):
    if not data:
        return "PYTORCH MODEL\nStatus: NOT AVAILABLE\n"

    return f"""
PYTORCH MODEL
--------------------------
mAP@0.5:      {data.get("mAP_50")}
mAP@0.5:0.95: {data.get("mAP_50_95")}
Precision:    {data.get("precision")}
Recall:       {data.get("recall")}
F1:           {data.get("f1")}

{format_latency(data.get("latency_ms"))}
"""


def onnx_section(data, pytorch_data=None):
    if not data:
        return "ONNX MODEL\nStatus: NOT AVAILABLE\n"

    latency_text = format_latency(data.get("latency_ms"))

    speedup_text = ""
    if pytorch_data and data.get("latency_ms"):
        pt_mean = pytorch_data.get("latency_ms", {}).get("mean")
        onnx_mean = data.get("latency_ms", {}).get("mean")

        if pt_mean and onnx_mean:
            speedup = pt_mean / onnx_mean
            speedup_text = f"\nSpeedup vs PyTorch: {speedup:.2f}x\n"

    return f"""
ONNX MODEL
--------------------------
Accuracy: Same as PyTorch (same weights, evaluated in PyTorch)

{latency_text}
{speedup_text}
"""


def generate():
    pt = load_metrics("results/pytorch_metrics.json")
    onnx = load_metrics("results/onnx_metrics.json")

    report = f"""
================================================================================
YOLOv5 BENCHMARK REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{pytorch_section(pt)}

{onnx_section(onnx, pt)}

================================================================================
"""

    Path("docs").mkdir(exist_ok=True)

    with open("docs/baseline_metrics.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    generate()
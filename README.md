# YOLOv5 Benchmarking and Optimization Pipeline

## Overview

This repository provides a complete benchmarking and optimization pipeline for YOLOv5. It includes:

- PyTorch evaluation on coco128  
- ONNX export and runtime benchmarking  
- ONNX optimization comparison  
- Automated benchmark report generation  
- Cross-platform automation (Windows and Linux)  

The full workflow can be executed using a single command.

---

## Model Details

- Model: YOLOv5s  
- Input Size: 640x640  
- Dataset: coco128 (subset of COCO)

### Evaluation Metrics

- mAP@0.5  
- mAP@0.5:0.95  
- Precision  
- Recall  
- F1 Score  
- Latency (Mean, P50, P95, P99 in ms)  
- FPS  
- ONNX Speedup  

---

## Repository Structure

```
project/
│
├── src/
│   ├── data/
│   ├── inference/
│   ├── models/
│   └── training/
│
├── exports.py
├── evaluate_pytorch.py
├── test_onnx.py
├── optimize_onnx.py
├── generate_baseline_report.py
├── setup_and_run.ps1
├── run_pipeline.sh
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- Git
- Windows PowerShell (for Windows users)
- Bash (for Linux/Mac users)

All Python dependencies are listed in:

```
requirements.txt
```

---

# Quick Start

## Windows

### 1. Clone Repository

```powershell
git clone https://github.com/kar-ish-ma/yolo_v5-.git
cd yolo_v5-
```

### 2. Allow Script Execution (One-time per session)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Run Full Pipeline

```powershell
.\setup_and_run.ps1
```

This will automatically:

1. Create virtual environment  
2. Install dependencies  
3. Export ONNX model  
4. Evaluate PyTorch model  
5. Run ONNX optimization benchmark  
6. Generate benchmark report  
7. Run visualization inference  

---

## Linux / Mac

### 1. Clone Repository

```bash
git clone https://github.com/kar-ish-ma/yolo_v5-.git
cd yolo_v5-
```

### 2. Make Script Executable

```bash
chmod +x run_pipeline.sh
```

### 3. Run Full Pipeline

```bash
./run_pipeline.sh
```

---

# Manual Execution (Advanced)

## Create Virtual Environment

Linux / Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Export ONNX

```bash
python exports.py
```

## Evaluate PyTorch

```bash
python evaluate_pytorch.py
```

## Run ONNX Benchmark

```bash
python optimize_onnx.py
```

## Generate Report

```bash
python generate_baseline_report.py
```

## Run Visualization Inference

```bash
python src/inference/infer.py --weights yolov5s.pt --source src/data/images/bus.jpg
```

---

# Output Files

After execution:

```
results/
 ├── pytorch_metrics.json
 ├── onnx_metrics.json
 ├── onnx_output.jpg
 └── optimized ONNX models

docs/
 └── baseline_metrics.txt
```

---

# Example Benchmark Output

```
============================================================
BENCHMARK RESULTS
============================================================
Optimization              Time (ms)    FPS        Speedup
------------------------------------------------------------
Baseline                  98.63        10.14      1.00x
Basic Optimizations       117.12       8.54       0.84x
All Optimizations         99.18        10.08      0.99x
Extended Optimizations    105.97       9.44       0.93x
```

---

# Benchmarking Methodology

## PyTorch

- Validation performed on coco128  
- Accuracy metrics computed using Ultralytics API  
- Latency measured over multiple forward passes  

## ONNX

- Model exported from PyTorch  
- Graph optimization enabled (ORT_ENABLE_ALL)  
- CPU execution provider  
- Latency averaged across multiple runs  
- Speedup computed relative to baseline ONNX  

Accuracy remains identical between PyTorch and ONNX since weights are unchanged. ONNX comparison focuses on runtime performance.

---

# Common Issues

## ModuleNotFoundError

Ensure virtual environment is activated and dependencies are installed:

```
pip install -r requirements.txt
```

## Report Shows None Values

Ensure `evaluate_pytorch.py` ran successfully before generating the report.

## PowerShell Script Blocked

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

# Updating the Repository

After making changes:

```bash
git add .
git commit -m "Describe your changes"
git push
```

---

# Notes

- Designed for reproducible benchmarking  
- Cross-platform compatible  
- Automation prevents metric overwriting  
- Suitable for academic evaluation and optimization experiments  

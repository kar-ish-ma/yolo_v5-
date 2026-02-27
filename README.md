# YOLOv5 Benchmarking and Optimization Pipeline

## Overview

This repository provides a complete benchmarking and optimization pipeline for YOLOv5. It includes:

- PyTorch evaluation on coco128
- ONNX export and inference benchmarking
- ONNX optimization comparison
- Automated benchmark report generation
- Cross-platform automation (Windows and Linux)

The entire workflow can be executed using a single command.

---

## Model Details

Model: YOLOv5s  
Input Size: 640x640  
Dataset: coco128 (subset of COCO)

### Evaluation Metrics

- mAP@0.5 — Mean Average Precision at IoU 0.5  
- mAP@0.5:0.95 — COCO standard metric  
- Precision — TP / (TP + FP)  
- Recall — TP / (TP + FN)  
- F1 Score — Harmonic mean of precision and recall  
- Latency — Mean, P50, P95, P99 (ms)  
- FPS — Frames per second  
- ONNX Speedup — Relative runtime improvement  

---

## Repository Structure

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

---

## Requirements

- Python 3.8+
- Git
- Windows PowerShell (for Windows users)
- Bash (for Linux/Mac users)

All dependencies are listed in:

requirements.txt

---

# Quick Start

## Windows

### Step 1: Clone Repository

git clone https://github.com/kar-ish-ma/yolo_v5-.git  
cd yolo_v5-

### Step 2: Allow Script Execution (One-time per session)

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

### Step 3: Run Full Pipeline

.\setup_and_run.ps1

This automatically:

1. Creates virtual environment  
2. Installs dependencies  
3. Exports ONNX model  
4. Evaluates PyTorch model  
5. Runs ONNX optimization benchmark  
6. Generates benchmark report  
7. Runs visualization inference  

---

## Linux / Mac

### Step 1: Clone Repository

git clone https://github.com/kar-ish-ma/yolo_v5-.git  
cd yolo_v5-

### Step 2: Make Script Executable

chmod +x run_pipeline.sh

### Step 3: Run Full Pipeline

./run_pipeline.sh

---

# Manual Execution (Advanced)

## Create Virtual Environment

Linux / Mac:

python3 -m venv .venv  
source .venv/bin/activate  

Windows:

python -m venv .venv  
.\.venv\Scripts\Activate.ps1  

## Install Dependencies

pip install -r requirements.txt

## Export ONNX

python exports.py

## Evaluate PyTorch

python evaluate_pytorch.py

## Run ONNX Benchmark

python optimize_onnx.py

## Generate Report

python generate_baseline_report.py

## Run Visualization Inference

python src/inference/infer.py --weights yolov5s.pt --source src/data/images/bus.jpg

---

# Output Files

After execution:

results/
 ├── pytorch_metrics.json
 ├── onnx_metrics.json
 ├── onnx_output.jpg
 └── optimized ONNX models

docs/
 └── baseline_metrics.txt

---

# Example Benchmark Output

============================================================
BENCHMARK RESULTS
============================================================
Optimization              Time (ms)    FPS        Speedup
------------------------------------------------------------
Baseline                  98.63        10.14      1.00x
Basic Optimizations       117.12       8.54       0.84x
All Optimizations         99.18        10.08      0.99x
Extended Optimizations    105.97       9.44       0.93x

---

# Benchmarking Methodology

PyTorch:
- Validation performed on coco128
- Accuracy metrics computed using Ultralytics API
- Latency measured over multiple forward passes

ONNX:
- Model exported from PyTorch
- Graph optimization enabled (ORT_ENABLE_ALL)
- CPU execution provider
- Latency averaged across multiple runs
- Speedup computed relative to baseline ONNX

Accuracy remains identical between PyTorch and ONNX since weights are unchanged.
ONNX comparison focuses on runtime performance.

---

# Common Issues

ModuleNotFoundError:
- Ensure virtual environment is activated
- Re-run pip install -r requirements.txt

Report Shows None Values:
- Ensure evaluate_pytorch.py ran successfully before generating the report

PowerShell Script Blocked:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

---

# Updating the Repository

git add .  
git commit -m "Describe your changes"  
git push  

---

# Notes

- Designed for reproducible benchmarking
- Cross-platform compatible
- Automation prevents metric overwriting
- Suitable for academic evaluation and optimization experiments

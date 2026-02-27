#!/bin/bash

echo "======================================="
echo "YOLOv5 FULL PIPELINE AUTOMATION"
echo "======================================="

set -e  # Stop if any command fails

# 1️⃣ Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo ""
    echo "[1/8] Creating virtual environment..."
    python3 -m venv .venv
fi

# 2️⃣ Activate venv
echo ""
echo "[2/8] Activating virtual environment..."
source .venv/bin/activate

# 3️⃣ Install dependencies
echo ""
echo "[3/8] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ Export ONNX
echo ""
echo "[4/8] Exporting ONNX model..."
python exports.py

# 5️⃣ Evaluate PyTorch
echo ""
echo "[5/8] Running PyTorch evaluation..."
python evaluate_pytorch.py

# 6️⃣ Run ONNX optimization benchmark
echo ""
echo "[6/8] Running ONNX optimization benchmark..."
python optimize_onnx.py

# 7️⃣ Generate Benchmark Report
echo ""
echo "[7/8] Generating benchmark report..."
python generate_baseline_report.py

# 8️⃣ Run visualization inference (does NOT overwrite metrics)
echo ""
echo "[8/8] Running PyTorch visualization inference..."
python src/inference/infer.py --weights yolov5s.pt --source src/data/images/bus.jpg

echo ""
echo "======================================="
echo "FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "======================================="

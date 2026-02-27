Write-Host "======================================="
Write-Host "YOLOv5 FULL PIPELINE AUTOMATION"
Write-Host "======================================="

$ErrorActionPreference = "Stop"

try {

    if (!(Test-Path ".venv")) {
        Write-Host "`n[1/8] Creating virtual environment..."
        python -m venv .venv
    }

    Write-Host "`n[2/8] Activating virtual environment..."
    .\.venv\Scripts\Activate.ps1

    Write-Host "`n[3/8] Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt

    Write-Host "`n[4/8] Exporting ONNX..."
    python exports.py

    Write-Host "`n[5/8] Running PyTorch evaluation..."
    python evaluate_pytorch.py

    Write-Host "`n[6/8] Running ONNX optimization benchmark..."
    python optimize_onnx.py

    Write-Host "`n[7/8] Generating benchmark report..."
    python generate_baseline_report.py

    Write-Host "`n[8/8] Running visualization inference..."
    python src/inference/infer.py --weights yolov5s.pt --source src/data/images/bus.jpg

    Write-Host "`n======================================="
    Write-Host "FULL PIPELINE COMPLETED SUCCESSFULLY"
    Write-Host "======================================="

}
catch {
    Write-Host "`nERROR OCCURRED:"
    Write-Host $_
}
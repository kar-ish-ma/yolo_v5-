
import torch
import onnx
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent.parent))

def export_onnx(weights='yolov5s.pt', imgsz=640, opset=17):
    """Export YOLOv5 to ONNX with opset 17"""

    print(f"\n{'='*50}")
    print(f"EXPORTING TO ONNX (Opset {opset})")
    print(f"{'='*50}")
    
    # Load model
    print(f"Loading model: {weights}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                          path=weights, trust_repo=True)
    model.eval()
    
    # Export
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    export_path = Path(weights).stem + '.onnx'
    
    torch.onnx.export(
    model.model,
    dummy_input,
    export_path,
    opset_version=opset,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}},
    export_params=True,
    do_constant_folding=True,
)
    
    # Verify
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"\n✓ ONNX Export Successful!")
    print(f"  Model: {export_path}")
    print(f"  Opset: {opset}")
    print(f"  Input: 1x3x{imgsz}x{imgsz}")
    print(f"{'='*50}\n")
    
    return export_path

if __name__ == '__main__':
    export_onnx(opset=17)
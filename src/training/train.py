# scripts/train.py - FIXED VERSION
import torch
import json
from pathlib import Path
from datetime import datetime
import sys
import os
import subprocess
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Change to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

def train_yolov5(epochs=5, batch_size=16):
    """Reproducible training script using YOLOv5's train.py"""
    
    print("\n" + "="*60)
    print("YOLOv5 TRAINING")
    print("="*60)
    
    # Method 1: Use YOLOv5's built-in training script (RECOMMENDED)
    print("\n1. Checking if YOLOv5 repo exists...")
    yolov5_dir = project_root / 'yolov5'
    
    if not yolov5_dir.exists():
        print("   Cloning YOLOv5 repository...")
        subprocess.run([
            'git', 'clone', 'https://github.com/ultralytics/yolov5.git'
        ], check=True)
    
    # Change to YOLOv5 directory
    os.chdir(yolov5_dir)
    
    # Install requirements if needed
    print("\n2. Installing YOLOv5 requirements...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
    ], check=True)
    
    # Run training
    print(f"\n3. Training for {epochs} epochs...")
    print(f"   Using: python train.py --epochs {epochs} --batch-size {batch_size} --data coco128.yaml --weights yolov5s.pt")
    
    result = subprocess.run([
        sys.executable, 'train.py',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--data', 'coco128.yaml',
        '--weights', 'yolov5s.pt',
        '--project', str(project_root / 'runs/train'),
        '--name', f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        '--exist-ok'
    ], capture_output=True, text=True)
    
    # Change back to project root
    os.chdir(project_root)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE")
        print("="*60)
        
        # Save metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'batch_size': batch_size,
            'status': 'success',
            'output': result.stdout[-500:] if result.stdout else ''
        }
    else:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED")
        print("="*60)
        print(f"Error: {result.stderr[:500]}")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'batch_size': batch_size,
            'status': 'failed',
            'error': result.stderr[:200] if result.stderr else 'Unknown error'
        }
    
    # Save metrics
    Path('results').mkdir(exist_ok=True)
    with open('results/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n✓ Metrics saved to: results/training_metrics.json")
    
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=16, dest='batch_size')
    args = parser.parse_args()
    
    train_yolov5(epochs=args.epochs, batch_size=args.batch_size)
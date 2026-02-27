# infer.py - WITH METRICS SAVING
import torch
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

def run_inference(weights='yolov5s.pt', source='bus.jpg', conf_thres=0.25):
    # Load model
    print(f"Loading model: {weights}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, trust_repo=True)
    model.conf = conf_thres
    
    print(f"Running inference on: {source}")
    
    # Run inference
    start_time = time.time()
    results = model(source)
    inference_time = time.time() - start_time
    
    # Print results
    results.print()
    save_path = results.save()
    
    # Extract detailed timing from results
    # results.t contains [preprocessing, inference, nms] times in milliseconds
    speed_info = results.speed if hasattr(results, 'speed') else {}
    
    # Get detection summary
    detections_summary = str(results).split('\n')[0] if len(str(results)) > 0 else "No detections"
    
    # Calculate metrics
    metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': weights,
        'source': source,
        'inference_time_total_ms': round(inference_time * 1000, 2),
        'inference_time_total_s': round(inference_time, 3),
        'fps': round(1 / inference_time, 2),
        'preprocessing_ms': speed_info.get('preprocess', 0) if speed_info else 0,
        'inference_ms': speed_info.get('inference', 0) if speed_info else 0,
        'nms_ms': speed_info.get('postprocess', 0) if speed_info else 0,
        'confidence_threshold': conf_thres,
        'detections': detections_summary,
        'save_path': str(save_path[0]) if save_path else 'runs/detect/exp'
    }
    
    # Save metrics to JSON file
    metrics_dir = Path('results')
    metrics_dir.mkdir(exist_ok=True)
    
    metrics_file = metrics_dir / 'pytorch_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n{'='*50}")
    print(f"Inference time: {inference_time:.3f}s ({inference_time*1000:.2f}ms)")
    print(f"FPS: {1/inference_time:.2f}")
    print(f"Results saved to: {save_path[0] if save_path else 'runs/detect/exp'}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"{'='*50}\n")
    
    return results, inference_time, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model weights path')
    parser.add_argument('--source', type=str, default='bus.jpg', help='image file')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    args = parser.parse_args()
    
    run_inference(weights=args.weights, source=args.source, conf_thres=args.conf_thres)
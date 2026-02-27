# scripts/optimize_onnx.py - WINDOWS COMPATIBLE
import onnxruntime as ort
import numpy as np
import cv2
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

def optimize_and_benchmark(onnx_path='yolov5s.onnx', image_path='bus.jpg'):
    """Apply graph optimizations and benchmark"""
    
    print("\n" + "="*60)
    print("="*60)
    
    # Load test image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    
    # Models to benchmark
    models_to_test = {
        'Baseline (No Opt)': onnx_path,
        'Basic Optimizations': None,  # Will create
        'All Optimizations': None,    # Will create
        'Extended Optimizations': None # Will create
    }
    
    results = []
    
    # 1. Baseline
    baseline_time = benchmark_model(onnx_path, img)
    results.append({
        'optimization': 'Baseline',
        'model': onnx_path,
        'time_ms': baseline_time,
        'fps': 1000/baseline_time,
        'speedup': 1.0
    })
    
    # 2. Create optimized versions
    opt_levels = [
        ('Basic Optimizations', ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, 'yolov5s_opt_basic.onnx'),
        ('All Optimizations', ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 'yolov5s_opt_all.onnx'),
        ('Extended Optimizations', ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, 'yolov5s_opt_extended.onnx')
    ]
    
    for name, level, save_path in opt_levels:
        try:
            # Create session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = level
            sess_options.optimized_model_filepath = save_path
            
            session = ort.InferenceSession(onnx_path, sess_options)
            
            # Benchmark
            time_ms = benchmark_model(session, img)
            speedup = baseline_time / time_ms
            
            results.append({
                'optimization': name,
                'model': save_path,
                'time_ms': time_ms,
                'fps': 1000/time_ms,
                'speedup': speedup
            })
            
            print(f"  ✓ Created: {save_path}")
            
        except Exception as e:
            print(f"  ✗ Failed: {name} - {str(e)[:50]}")
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'optimization_benchmark.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print comparison table
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Optimization':<25} {'Time (ms)':<12} {'FPS':<10} {'Speedup':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['optimization']:<25} {r['time_ms']:<12.2f} {r['fps']:<10.2f} {r['speedup']:<10.2f}x")
    
    # Find best
    best = max(results, key=lambda x: x['speedup'])
    print("\n" + "="*60)
    print(f"🏆 BEST OPTIMIZATION: {best['optimization']}")
    print(f"   Speedup: {best['speedup']:.2f}x")
    print(f"   Time: {best['time_ms']:.2f}ms → FPS: {best['fps']:.2f}")
    print("="*60)
    
    return results

def benchmark_model(model_or_path, img, iterations=50):
    """Benchmark inference time"""
    
    if isinstance(model_or_path, str):
        session = ort.InferenceSession(model_or_path)
    else:
        session = model_or_path
    
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: img})
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = session.run(None, {input_name: img})
        times.append((time.time() - start) * 1000)
    
    return np.mean(times[10:])  # Skip first 10

if __name__ == '__main__':
    # First, ensure ONNX model exists
    if not Path('yolov5s.onnx').exists():
        print("ONNX model not found! Running export.py first...")
        from export import export_onnx
        export_onnx()
    
    # Run optimizations
    optimize_and_benchmark()
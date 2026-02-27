import onnxruntime as ort
import numpy as np
import cv2
import time
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime


def load_class_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_rgb = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0
    return img, img_rgb


def percentile_stats(times):
    return {
        "mean": float(np.mean(times)),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }


def test_onnx_inference(onnx_path, image_path, yaml_path):
    print(f"Loading ONNX model: {onnx_path}")

    # Load class names
    class_names = load_class_names(yaml_path)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path,
        options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    original_img, img_input = preprocess(image_path)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: img_input})

    # Timed runs
    times = []
    for _ in range(20):
        start = time.time()
        outputs = session.run(None, {input_name: img_input})
        times.append((time.time() - start) * 1000)

    inference_time = np.mean(times)
    fps = 1000 / inference_time

    preds = outputs[0][0]  # (num_boxes, 85)

    conf_threshold = 0.25
    iou_threshold = 0.45

    boxes = []
    scores = []
    class_ids = []

    for pred in preds:
        obj_conf = pred[4]
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        conf = obj_conf * class_conf

        if conf < conf_threshold:
            continue

        x, y, w, h = pred[:4]
        x1 = x - w / 2
        y1 = y - h / 2

        boxes.append([x1, y1, w, h])
        scores.append(float(conf))
        class_ids.append(class_id)

    boxes = np.array(boxes)
    scores = np.array(scores)

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        conf_threshold,
        iou_threshold
    )

    h_orig, w_orig = original_img.shape[:2]

    if len(indices) > 0:
        indices = indices.flatten()

        for i in indices:
            x, y, w, h = boxes[i]

            # Scale back to original image
            x_scale = w_orig / 640
            y_scale = h_orig / 640

            x1 = int(x * x_scale)
            y1 = int(y * y_scale)
            w1 = int(w * x_scale)
            h1 = int(h * y_scale)

            class_id = class_ids[i]
            label = f"{class_names[class_id]} {scores[i]:.2f}"

            cv2.rectangle(original_img,
                          (x1, y1),
                          (x1 + w1, y1 + h1),
                          (255, 0, 0),
                          2)

            cv2.putText(original_img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2)

    # Save output image
    Path("results").mkdir(exist_ok=True)
    output_path = "results/onnx_output.jpg"
    cv2.imwrite(output_path, original_img)

    # Save metrics
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": onnx_path,
        "inference_time_ms": round(float(inference_time), 2),
        "fps": round(float(fps), 2),
        "num_detections": int(len(indices)) if len(boxes) > 0 else 0
    }

    with open("results/onnx_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n========================================")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Saved output image: {output_path}")
    print("========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5s.onnx")
    parser.add_argument("--source", type=str, default="src/data/images/bus.jpg")
    parser.add_argument("--data", type=str, default="src/data/coco128.yaml")

    args = parser.parse_args()

    test_onnx_inference(args.weights, args.source, args.data)
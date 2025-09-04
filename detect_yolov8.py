#!/usr/bin/env python3
"""
YOLOv8 Vehicle Detection Inference Script
Perform inference with a trained YOLOv8 model

Usage:
    python detect_yolov8_fixed.py --model runs/train/exp17/weights/best.pt --source test_images/car2.jpg --save
"""

import argparse
import os

# Set environment variables BEFORE importing torch/ultralytics to completely disable CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices to force CPU usage
os.environ['TORCH_USE_CUDA_DSA'] = '0'

from pathlib import Path
import torch

# Force CPU usage completely
torch.backends.cudnn.enabled = False
torch.set_default_tensor_type('torch.FloatTensor')  # Force CPU tensors

from ultralytics import YOLO
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Vehicle Detection Inference')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 model (.pt file)')
    parser.add_argument('--source', type=str, default='test_images/',
                        help='Source: image file, video file, directory, webcam (0)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu only (GPU disabled for memory issues)')
    parser.add_argument('--save', action='store_true',
                        help='Save inference results')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save results as .txt files')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Save results to project/name')
    parser.add_argument('--name', type=str, default='vehicle_detection',
                        help='Save results to project/name')
    parser.add_argument('--show', action='store_true',
                        help='Display results')

    args = parser.parse_args()

    # Force CPU usage
    args.device = 'cpu'

    # Validate model file
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    # Load YOLO model
    print(f"Loading YOLOv8 model: {args.model}")
    try:
        # Set number of threads to avoid multiprocessing issues
        torch.set_num_threads(1)

        # Load model with CPU only
        model = YOLO(args.model)
        model.to('cpu')

        print(f"Model loaded successfully on CPU: {args.model}")

        # Get model info safely
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            print(f"Model classes: {list(model.model.names.values())}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Vehicle class names (should match your dataset)
    class_names = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Bicycle']

    print(f"Starting inference with the following parameters:")
    print(f"  Model: {args.model}")
    print(f"  Source: {args.source}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Device: CPU (forced)")
    print(f"  Save results: {args.save}")

    # Handle different source types
    if os.path.isfile(args.source):
        sources = [args.source]
    elif os.path.isdir(args.source):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        sources = [os.path.join(args.source, f) for f in os.listdir(args.source)
                  if f.lower().endswith(image_extensions)]
    else:
        sources = [args.source]

    if not sources:
        print(f"No valid image files found in {args.source}")
        return

    print(f"Processing {len(sources)} image(s)...")

    # Process each source
    total_detections = 0
    class_counts = {name: 0 for name in class_names}

    for i, source in enumerate(sources):
        print(f"Processing {i+1}/{len(sources)}: {source}")

        try:
            # Use the simplest possible inference call
            results = model.predict(
                source=source,
                conf=args.conf,
                iou=args.iou,
                device='cpu',
                verbose=False,
                save=args.save,
                project=args.project,
                name=args.name,
                imgsz=320,  # Smaller image size for less memory usage
                workers=0   # Disable multiprocessing
            )

            # Process results
            if results:
                result = results[0] if isinstance(results, list) else results

                if hasattr(result, 'boxes') and result.boxes is not None:
                    detections = len(result.boxes)
                    total_detections += detections
                    print(f"  Found {detections} detections")

                    # Count by class if possible
                    if hasattr(result.boxes, 'cls'):
                        for cls_id in result.boxes.cls:
                            cls_idx = int(cls_id.item()) if hasattr(cls_id, 'item') else int(cls_id)
                            if 0 <= cls_idx < len(class_names):
                                class_counts[class_names[cls_idx]] += 1
                else:
                    print(f"  No detections found")

        except Exception as e:
            print(f"  Error processing {source}: {e}")
            continue

    print(f"\nInference completed!")
    print(f"Total detections: {total_detections}")
    print("Detections by class:")
    for class_name, count in class_counts.items():
        if count > 0:
            print(f"  {class_name}: {count}")

    if args.save:
        print(f"Results saved to: {args.project}/{args.name}")

    return total_detections

if __name__ == '__main__':
    main()

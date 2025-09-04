#!/usr/bin/env python3
"""
Simple YOLOv8 Vehicle Detection Inference
Quick inference script for vehicle detection with YOLOv8

Usage:
    python infer_yolov8.py                          # Use webcam
    python infer_yolov8.py --source test_images/    # Process folder
    python infer_yolov8.py --source image.jpg       # Process single image
    python infer_yolov8.py --source video.mp4       # Process video
"""

import argparse
from ultralytics import YOLO
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Vehicle Detection Inference')
    parser.add_argument('--model', type=str, default='runs/train/vehicle_yolov8/weights/best.pt',
                        help='Path to trained YOLOv8 model')
    parser.add_argument('--source', type=str, default='0',
                        help='Source: webcam (0), image file, video file, or folder')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for inference')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--save', action='store_true',
                        help='Save inference results')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save results as .txt files')
    parser.add_argument('--device', type=str, default='',
                        help='Device to run inference on')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Use the default YOLOv8 model or train your own model first")
        args.model = 'yolov8s.pt'  # Fallback to pretrained model
        print(f"🔄 Using fallback model: {args.model}")

    # Load model
    print(f"📦 Loading model: {args.model}")
    model = YOLO(args.model)

    # Vehicle class names
    class_names = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Bicycle']

    print(f"🎯 Inference Settings:")
    print(f"   Model: {args.model}")
    print(f"   Source: {args.source}")
    print(f"   Confidence: {args.conf}")
    print(f"   IoU: {args.iou}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Device: {args.device if args.device else 'auto'}")

    # Run inference
    print(f"🚀 Starting inference...")

    if args.source == '0':  # Webcam
        print("📹 Using webcam - Press 'q' to quit")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        show=args.show,
        save=args.save,
        save_txt=args.save_txt,
        verbose=True,
        stream=True if args.source == '0' else False,
    )

    # Process results
    total_detections = 0
    class_counts = {name: 0 for name in class_names}

    for result in results:
        if result.boxes is not None:
            detections = len(result.boxes)
            total_detections += detections

            # Count detections by class
            for box in result.boxes:
                if box.cls is not None:
                    class_id = int(box.cls.item())
                    if 0 <= class_id < len(class_names):
                        class_counts[class_names[class_id]] += 1

    print(f"\n✅ Inference completed!")
    print(f"📊 Total detections: {total_detections}")
    if total_detections > 0:
        print("📈 Detections by vehicle type:")
        for class_name, count in class_counts.items():
            if count > 0:
                percentage = (count / total_detections) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")

    if args.save:
        print(f"💾 Results saved to: runs/detect/predict")

if __name__ == '__main__':
    main()

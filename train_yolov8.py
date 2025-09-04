#!/usr/bin/env python3
"""
YOLOv8 Vehicle Detection - Advanced Training Configuration
İleri seviye performans optimizasyonu için

Usage:
    python train_yolov8_advanced.py --model yolov8m.pt --epochs 150 --batch 8
"""

import argparse
import os
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

# GPU optimizasyonları
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    parser = argparse.ArgumentParser(description='Advanced YOLOv8 Vehicle Detection Training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='YOLOv8 model variant')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=832,  # Daha büyük image size
                        help='Image size for training')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.001,  # Daha düşük final LR
                        help='Final learning rate factor')
    
    # Advanced augmentation
    parser.add_argument('--hsv_h', type=float, default=0.02,
                        help='Image HSV-Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=0.8,
                        help='Image HSV-Saturation augmentation') 
    parser.add_argument('--hsv_v', type=float, default=0.5,
                        help='Image HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=5.0,
                        help='Image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.15,
                        help='Image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='Image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=2.0,
                        help='Image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0005,
                        help='Image perspective (+/- fraction)')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.15,  # Mixup eklendi
                        help='Image mixup (probability)')
    parser.add_argument('--copy_paste', type=float, default=0.3,  # Copy-paste artırıldı
                        help='Copy paste augmentation (probability)')
    
    # Training settings
    parser.add_argument('--optimizer', type=str, default='AdamW',  # AdamW kullan
                        choices=['SGD', 'Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--cos_lr', action='store_true', default=True,
                        help='Use cosine learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=float, default=5.0,
                        help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='EarlyStopping patience')
    parser.add_argument('--save_period', type=int, default=20,
                        help='Save checkpoint every x epochs')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads for data loading')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Automatic Mixed Precision training')
    
    # Output settings
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Save results to project/name')
    parser.add_argument('--name', type=str, default='vehicle_advanced',
                        help='Save results to project/name')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Existing project/name ok, do not increment')
    
    args = parser.parse_args()
    
    # Force GPU usage
    args.device = '0'
    
    print(f"🚀 Advanced YOLOv8 Vehicle Detection Training")
    print(f"🔧 Model: {args.model}")
    print(f"🔧 Epochs: {args.epochs}")
    print(f"🔧 Batch Size: {args.batch}")
    print(f"🔧 Image Size: {args.imgsz}")
    print(f"🔧 Optimizer: {args.optimizer}")
    print(f"🔧 Learning Rate: {args.lr0} -> {args.lr0 * args.lrf}")
    
    # GPU bilgileri
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # Load model
    print(f"📦 Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Print augmentation settings
    print(f"\n🔄 Advanced Data Augmentation:")
    print(f"   HSV: H={args.hsv_h}, S={args.hsv_s}, V={args.hsv_v}")
    print(f"   Geometric: degrees={args.degrees}, translate={args.translate}, scale={args.scale}")
    print(f"   Shear: {args.shear}, Perspective: {args.perspective}")
    print(f"   Flip LR: {args.fliplr}")
    print(f"   Mosaic: {args.mosaic}, Mixup: {args.mixup}, Copy-paste: {args.copy_paste}")
    
    # Start training
    print(f"\n🚀 Starting advanced training...")
    results = model.train(
        data='dataset.yaml',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        
        # Optimizer settings
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=args.cos_lr,
        
        # Data augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=0.0,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        
        # Training settings
        amp=args.amp,
        patience=args.patience,
        save_period=args.save_period,
        
        # Validation settings
        val=True,
        plots=True,
        save=True,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        
        # Advanced settings
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        
        # Model optimization
        close_mosaic=10,  # Son 10 epoch'ta mosaic'i kapat
    )
    
    # Print final results
    print(f"\n📈 Advanced Training Completed!")
    print(f"Results saved to: {args.project}/{args.name}")
    
    return results

if __name__ == '__main__':
    main()

"""Train YOLOv11 model on the newly created video dataset.

Usage:
    python scripts/train_model.py [--epochs 15] [--batch 16] [--imgsz 640]
"""
import argparse
import os
import shutil
from ultralytics import YOLO

def train(epochs=15, batch=16, imgsz=640, device="cpu", lr0=0.005):
    data_yaml = r"D:\hoclamAI\giaothong\datasets\helmet_video\dataset.yaml"
    pretrained_model = r"D:\hoclamAI\giaothong\models\best_ver3.pt"
    save_project = r"D:\hoclamAI\giaothong\runs"
    exp_name = "train_video_v4"
    target_best_model = r"D:\hoclamAI\giaothong\models\best_video_v4.pt"

    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Data config:        {data_yaml}")
    print(f"Pretrained weights: {pretrained_model}")
    print(f"Epochs:             {epochs}")
    print(f"Batch size:         {batch}")
    print(f"Image size:         {imgsz}")
    print(f"Device:             {device}")
    print(f"Initial LR:         {lr0}")
    print("=" * 60)

    # Load model
    model = YOLO(pretrained_model)
    print("Loaded architecture:", model.task, model.names)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=4,
        project=save_project,
        name=exp_name,
        exist_ok=True,
        lr0=lr0,
        plots=True,
        save=True,
        verbose=True,
        val=True
    )

    best_weight = os.path.join(save_project, exp_name, "weights", "best.pt")
    if os.path.exists(best_weight):
        print(f"\nTraining completed! Copying {best_weight} -> {target_best_model}")
        shutil.copy2(best_weight, target_best_model)
    else:
        last_weight = os.path.join(save_project, exp_name, "weights", "last.pt")
        if os.path.exists(last_weight):
            shutil.copy2(last_weight, target_best_model)

    print("\nRunning Validation Evaluation...")
    val_model = YOLO(target_best_model if os.path.exists(target_best_model) else best_weight)
    val_res = val_model.val(data=data_yaml, imgsz=imgsz, device=device)

    print("\n" + "=" * 60)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 60)
    print(f"Overall mAP50:    {val_res.box.map50:.4f}")
    print(f"Overall mAP50-95: {val_res.box.map:.4f}")
    print(f"Precision:        {val_res.box.mp:.4f}")
    print(f"Recall:           {val_res.box.mr:.4f}")
    print("-" * 60)
    names = val_model.names
    for i, cls_name in names.items():
        if i < len(val_res.box.maps):
            map50 = val_res.box.maps[i]
            print(f"Class [{i}] {cls_name:15s}: mAP50-95 = {map50:.4f}")
    print("=" * 60)
    print(f"Model saved to: {target_best_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr0", type=float, default=0.005)
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device, lr0=args.lr0)

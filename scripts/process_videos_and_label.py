"""Extract frames from videos, auto-label using teacher YOLO model,
merge existing dataset, split into train/val, and generate Ultralytics yaml.

Usage:
    python scripts/process_videos_and_label.py
"""
import glob
import os
import random
import shutil
from collections import Counter
import cv2
from ultralytics import YOLO

VIDEO_DIR = r"D:\hoclamAI\giaothong\video"
EXISTING_RAW_DIR = r"D:\hoclamAI\giaothong\datasets\helmet\raw"
OUTPUT_DIR = r"D:\hoclamAI\giaothong\datasets\helmet_video"
TEACHER_MODEL_PATH = r"D:\hoclamAI\giaothong\models\best_ver3.pt"
CONF_THRESHOLD = 0.35
FRAME_INTERVAL = 15
VAL_FRACTION = 0.15
RANDOM_SEED = 42


def auto_label_videos():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

    print(f"Loading teacher model: {TEACHER_MODEL_PATH}")
    model = YOLO(TEACHER_MODEL_PATH)
    print(f"Model classes: {model.names}")

    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) +
                         glob.glob(os.path.join(VIDEO_DIR, "*.avi")) +
                         glob.glob(os.path.join(VIDEO_DIR, "*.mov")))
    image_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.jpg")) +
                         glob.glob(os.path.join(VIDEO_DIR, "*.png")) +
                         glob.glob(os.path.join(VIDEO_DIR, "*.jpeg")))

    print(f"Found {len(video_files)} videos and {len(image_files)} standalone images in {VIDEO_DIR}")

    total_saved = 0
    class_counter = Counter()

    # 1. Process standalone images in video/
    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        res = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        if len(res.boxes) == 0:
            continue

        lines = []
        for b in res.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            # Clamp coordinates
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw <= 0.005 or bh <= 0.005:
                continue
            cx = (x1 + x2) / (2.0 * w)
            cy = (y1 + y2) / (2.0 * h)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counter[cls] += 1

        if lines:
            out_img = os.path.join(OUTPUT_DIR, "images", f"{stem}.jpg")
            out_lbl = os.path.join(OUTPUT_DIR, "labels", f"{stem}.txt")
            cv2.imwrite(out_img, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            total_saved += 1

    # 2. Process all videos
    for vid_idx, vid_path in enumerate(video_files):
        stem_base = os.path.splitext(os.path.basename(vid_path))[0]
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Cannot open video: {vid_path}")
            continue

        frame_count = 0
        vid_saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if frame_count % FRAME_INTERVAL != 0:
                continue

            h, w = frame.shape[:2]
            res = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            if len(res.boxes) == 0:
                continue

            lines = []
            for b in res.boxes:
                cls = int(b.cls[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                if bw <= 0.005 or bh <= 0.005:
                    continue
                cx = (x1 + x2) / (2.0 * w)
                cy = (y1 + y2) / (2.0 * h)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                class_counter[cls] += 1

            if lines:
                name = f"{stem_base}_{frame_count:06d}"
                out_img = os.path.join(OUTPUT_DIR, "images", f"{name}.jpg")
                out_lbl = os.path.join(OUTPUT_DIR, "labels", f"{name}.txt")
                cv2.imwrite(out_img, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                with open(out_lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                total_saved += 1
                vid_saved += 1

        cap.release()
        print(f"[{vid_idx+1}/{len(video_files)}] {os.path.basename(vid_path)}: saved {vid_saved} frames ({frame_count} total frames)")

    print(f"\nAuto-labeling done from videos! Extracted {total_saved} images.")
    print("Class counts from video extraction:", {model.names[k]: v for k, v in class_counter.items()})

    # 3. Merge existing vetted samples from EXISTING_RAW_DIR
    if os.path.exists(EXISTING_RAW_DIR):
        print("\nMerging existing vetted samples from:", EXISTING_RAW_DIR)
        merged_count = 0
        for split in ("train", "val"):
            img_dir = os.path.join(EXISTING_RAW_DIR, "images", split)
            lbl_dir = os.path.join(EXISTING_RAW_DIR, "labels", split)
            if not os.path.exists(img_dir):
                continue
            for img_name in os.listdir(img_dir):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    base = os.path.splitext(img_name)[0]
                    lbl_name = base + ".txt"
                    src_img = os.path.join(img_dir, img_name)
                    src_lbl = os.path.join(lbl_dir, lbl_name)
                    if os.path.exists(src_lbl):
                        dst_img = os.path.join(OUTPUT_DIR, "images", f"existing_{img_name}")
                        dst_lbl = os.path.join(OUTPUT_DIR, "labels", f"existing_{lbl_name}")
                        shutil.copy2(src_img, dst_img)
                        shutil.copy2(src_lbl, dst_lbl)
                        merged_count += 1
                        with open(src_lbl, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_counter[int(parts[0])] += 1
        print(f"Merged {merged_count} existing images.")

    # 4. Split into train / val
    img_dir = os.path.join(OUTPUT_DIR, "images")
    lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    valid_imgs = [f for f in all_imgs if os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))]

    random.Random(RANDOM_SEED).shuffle(valid_imgs)
    n_val = max(1, int(len(valid_imgs) * VAL_FRACTION))
    val_set = set(valid_imgs[:n_val])
    train_set = [f for f in valid_imgs if f not in val_set]

    for sub in ("train", "val"):
        os.makedirs(os.path.join(img_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(lbl_dir, sub), exist_ok=True)

    for f in valid_imgs:
        sub = "val" if f in val_set else "train"
        stem = os.path.splitext(f)[0]
        src_i = os.path.join(img_dir, f)
        dst_i = os.path.join(img_dir, sub, f)
        src_l = os.path.join(lbl_dir, stem + ".txt")
        dst_l = os.path.join(lbl_dir, sub, stem + ".txt")
        shutil.move(src_i, dst_i)
        shutil.move(src_l, dst_l)

    # 5. Class frequency summary
    split_freq = Counter()
    for sub in ("train", "val"):
        for f in os.listdir(os.path.join(lbl_dir, sub)):
            if f.endswith(".txt"):
                with open(os.path.join(lbl_dir, sub, f), "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if parts:
                            split_freq[(sub, int(parts[0]))] += 1

    print(f"\nFinal Dataset Split: Train = {len(train_set)} images | Val = {len(val_set)} images")
    print("Class distributions:")
    for sub in ("train", "val"):
        dist = {model.names[cls_id]: count for (s, cls_id), count in split_freq.items() if s == sub}
        print(f"  {sub}: {dist}")

    # 6. Generate dataset YAML
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    out_dir_clean = OUTPUT_DIR.replace("\\", "/")
    yaml_content = f"""path: {out_dir_clean}
train: images/train
val: images/val

names:
  0: moto
  1: helmet
  2: no_helmet
  3: license_plate
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\nCreated dataset YAML: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    auto_label_videos()

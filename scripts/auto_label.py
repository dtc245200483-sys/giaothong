"""Auto-label frames from a video using a trained YOLO model (YOLO label format).

Usage:
    python auto_label.py --video <file.mp4> --out <dataset_dir> [--interval 15]

Output:
    <out>/images/<stem>_<frame>.jpg
    <out>/labels/<stem>_<frame>.txt   (class cx cy w h normalized)
"""
import argparse
import os
import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=r"D:\hoclamAI\giaothong\models\best_ver3.pt")
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    model = YOLO(args.model)
    print("classes:", model.names)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("cannot open video:", args.video)
        return

    n = 0
    saved = 0
    stem_base = os.path.splitext(os.path.basename(args.video))[0]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % args.interval != 0:
            continue
        if args.max_frames and saved >= args.max_frames:
            break

        h, w = frame.shape[:2]
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        if len(r.boxes) == 0:
            continue

        stem = f"{stem_base}_{n:06d}"
        cv2.imwrite(os.path.join(args.out, "images", stem + ".jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        lines = []
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cx = (x1 + x2) / (2.0 * w)
            cy = (y1 + y2) / (2.0 * h)
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(args.out, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        saved += 1
        if saved % 20 == 0:
            print(f"  saved {saved} frames")

    cap.release()
    print(f"done: {saved} labeled frames from {n} frames")


if __name__ == "__main__":
    main()

"""Enhanced Visual Demo: Fullscreen/Resizable Window, High-Contrast Labels & Zoom PiP.

Usage:
    # 1. Preview Hanoi video (window opens big & fits screen):
    python scripts/demo_video.py --input "D:/hoclamAI/giaothong/video/hanoi_bach_mai_traffic.mp4" --show

    # 2. Preview 480p video (auto-upscaled to 720p HD display):
    python scripts/demo_video.py --input "D:/hoclamAI/giaothong/video/1790209379225_9011771178293999674_9011771178293999674.mp4" --show

    # 3. Save to MP4:
    python scripts/demo_video.py --input "D:/hoclamAI/giaothong/video/hanoi_bach_mai_traffic.mp4" --save "runs/hanoi_annotated.mp4"
"""
import argparse
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# High-contrast color palette
COLOR_PALETTE = {
    0: {"box": (230, 140, 20),  "badge": (230, 140, 20), "text": (255, 255, 255)}, # moto: Ocean Blue
    1: {"box": (40, 180, 40),   "badge": (40, 180, 40),   "text": (255, 255, 255)}, # helmet: Green
    2: {"box": (30, 30, 240),   "badge": (30, 30, 240),   "text": (255, 255, 255)}, # no_helmet: Red (VIOLATION)
    3: {"box": (0, 200, 240),   "badge": (0, 200, 240),   "text": (0, 0, 0)},       # license_plate: Amber Yellow (black text)
}

# 100% clean ASCII labels (no '???' encoding errors in OpenCV)
LABEL_CLEAN = {
    0: "Xe may",
    1: "Co mu",
    2: "KHONG MU",
    3: "Bien so",
}


def enhance_crop(crop, target_w=180, target_h=120):
    if crop is None or crop.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    upscaled = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)
    return sharpened


def run_demo(input_source, model_path, conf_thresh, imgsz, save_path=None, show_window=False, fullscreen=False, max_frames=0):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    if str(input_source).isdigit():
        source = int(input_source)
    else:
        source = os.path.normpath(input_source)
        if not os.path.exists(source):
            print(f"Error: Input file does not exist: {source}")
            return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate optimal display & render resolution
    # If video is smaller than 720p (like 854x480), upscale to 1280x720 so it fills the screen cleanly
    if src_w < 1280 and src_w > 0:
        render_w = 1280
        render_h = int(1280 * (src_h / src_w))
    else:
        render_w = src_w
        render_h = src_h

    window_name = "Giam Sat Giao Thong - Nhan Dien Vi Pham (Full Screen: Phim F | Tam dung: Space | Thoat: Q)"

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(render_w, 1600), min(render_h, 900))
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (render_w, render_h))
        print(f"Saving output to: {save_path}")

    frame_idx = 0
    t0 = time.time()
    total_violations_seen = 0
    is_paused = False
    is_fullscreen = fullscreen

    print("\n--- CONTROLS ---")
    print("  [F]     : Bat/Tat Fullscreen (Toan man hinh)")
    print("  [SPACE] : Tam dung / Tiep tuc")
    print("  [Q/ESC] : Thoat chuong trinh")
    print("----------------\n")

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames > 0 and frame_idx > max_frames:
                print(f"Dat gioi han {max_frames} khung hinh yeu cau.")
                break

            # Resize frame to clean HD render resolution if needed
            if (src_w, src_h) != (render_w, render_h):
                frame = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_LINEAR)

            # Predict
            results = model.predict(frame, conf=conf_thresh, imgsz=imgsz, verbose=False)[0]

            counts = {"moto": 0, "helmet": 0, "no_helmet": 0, "license_plate": 0}
            latest_plate_crop = None
            latest_violation_crop = None

            # Draw detections
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_key = model.names.get(cls_id, str(cls_id))
                counts[cls_key] = counts.get(cls_key, 0) + 1
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(render_w, x2), min(render_h, y2)

                style = COLOR_PALETTE.get(cls_id, {"box": (200, 200, 200), "badge": (200, 200, 200), "text": (0, 0, 0)})
                box_thick = 3 if cls_id == 2 else 2

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), style["box"], box_thick)

                # Label badge
                lbl_text = f"{LABEL_CLEAN.get(cls_id, cls_key)} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)

                badge_y1 = max(0, y1 - th - 8)
                badge_y2 = y1
                cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 8, badge_y2), style["badge"], -1)
                cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 8, badge_y2), (20, 20, 20), 1)

                cv2.putText(frame, lbl_text, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, style["text"], 1, cv2.LINE_AA)

                # Collect crops for PiP zoom
                if cls_id == 3 and latest_plate_crop is None:
                    pw, ph = x2 - x1, y2 - y1
                    if pw > 6 and ph > 6:
                        px1 = max(0, int(x1 - 0.15 * pw))
                        py1 = max(0, int(y1 - 0.15 * ph))
                        px2 = min(render_w, int(x2 + 0.15 * pw))
                        py2 = min(render_h, int(y2 + 0.15 * ph))
                        latest_plate_crop = frame[py1:py2, px1:px2]

                if cls_id == 2 and latest_violation_crop is None:
                    hw, hh = x2 - x1, y2 - y1
                    hx1 = max(0, int(x1 - 0.3 * hw))
                    hy1 = max(0, int(y1 - 0.3 * hh))
                    hx2 = min(render_w, int(x2 + 0.3 * hw))
                    hy2 = min(render_h, int(y2 + 0.3 * hh))
                    latest_violation_crop = frame[hy1:hy2, hx1:hx2]

            if counts["no_helmet"] > 0:
                total_violations_seen += 1

            # Dashboard (Top-Left)
            dash_w, dash_h = 340, 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (12, 12), (12 + dash_w, 12 + dash_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            is_viol = counts["no_helmet"] > 0
            border_col = (0, 0, 240) if is_viol else (0, 180, 0)
            cv2.rectangle(frame, (12, 12), (12 + dash_w, 12 + dash_h), border_col, 2)

            sec = frame_idx / fps
            cv2.putText(frame, f"Khung hinh: {frame_idx}/{total_frames} ({sec:.1f}s)", (24, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1)
            cv2.putText(frame, f"Xe may: {counts['moto']}  |  Bien so: {counts['license_plate']}", (24, 64),
                        cv2.FONT_HERSHEY_DUPLEX, 0.50, (220, 220, 220), 1)

            v_text = f"VI PHAM (KHONG MU): {counts['no_helmet']}"
            v_col = (50, 50, 255) if is_viol else (80, 220, 80)
            cv2.putText(frame, v_text, (24, 92),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, v_col, 2 if is_viol else 1)

            # Picture-in-Picture: Zooms in bottom-right
            pip_w, pip_h = 180, 120
            if latest_plate_crop is not None and latest_plate_crop.size > 0:
                zoom_plate = enhance_crop(latest_plate_crop, pip_w, pip_h)
                px = render_w - pip_w - 15
                py = render_h - pip_h - 15
                frame[py:py+pip_h, px:px+pip_w] = zoom_plate
                cv2.rectangle(frame, (px, py), (px + pip_w, py + pip_h), (0, 200, 240), 2)
                cv2.rectangle(frame, (px, py), (px + 120, py + 20), (0, 200, 240), -1)
                cv2.putText(frame, "ZOOM BIEN SO", (px + 6, py + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 0, 0), 1)

            if latest_violation_crop is not None and latest_violation_crop.size > 0:
                zoom_viol = enhance_crop(latest_violation_crop, pip_w, pip_h)
                vx = render_w - (pip_w * 2) - 30
                vy = render_h - pip_h - 15
                frame[vy:vy+pip_h, vx:vx+pip_w] = zoom_viol
                cv2.rectangle(frame, (vx, vy), (vx + pip_w, vy + pip_h), (0, 0, 240), 2)
                cv2.rectangle(frame, (vx, vy), (vx + 130, vy + 20), (0, 0, 240), -1)
                cv2.putText(frame, "CAN CANH LOI", (vx + 6, vy + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1)

            if writer:
                writer.write(frame)

        if show_window:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1 if not is_paused else 50) & 0xFF
            if key == ord('q') or key == 27: # q or Esc
                print("Dung chuong trinh boi nguoi dung.")
                break
            elif key == ord(' '): # Space
                is_paused = not is_paused
                print("Tam dung..." if is_paused else "Tiep tuc...")
            elif key == ord('f'): # Fullscreen toggle
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"\nDone! Tong so frame da xu ly: {frame_idx}. So frame vi pham: {total_violations_seen}")
    if save_path:
        print(f"File video ket qua da luu tai: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"D:\hoclamAI\giaothong\video\hanoi_bach_mai_traffic.mp4")
    parser.add_argument("--model", default=r"D:\hoclamAI\giaothong\models\best_video_v4.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="So frame toi da can chay (0 = chay het)")
    args = parser.parse_args()

    run_demo(args.input, args.model, args.conf, args.imgsz, args.save, args.show, args.fullscreen, args.max_frames)

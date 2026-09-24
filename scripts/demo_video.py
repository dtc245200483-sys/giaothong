"""Traffic Violation Monitoring System (Full Screen, High-Recall Dual Detector, Tracker & Violation Registry Table).

Features:
- Dual High-Recall Detection: Fuses COCO (motorcycle, person) & Custom Model (helmet, no_helmet, license_plate).
- Object Tracking: Tracks every motorcycle across frames with unique ID (XE-01, XE-02...).
- Cumulative Violation Counter: Total violation count strictly increments (+1) whenever a new violating bike/rider is spotted and NEVER resets to 0.
- On-Screen Violation Registry Table: Live semi-transparent table listing every violation with (STT, Time, Bike ID, License Plate, Violation Type).
- Auto-export: Saves table to runs/violation_registry.csv and evidence snapshot crops.
- Fullscreen & Resizable: Full window scaling, press [F] for Fullscreen, [Space] to pause, [Q/Esc] to exit.
"""
import argparse
import csv
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

# High-contrast color palette
COLOR_PALETTE = {
    "moto":          {"box": (230, 140, 20),  "badge": (230, 140, 20), "text": (255, 255, 255)}, # Ocean Blue
    "person":        {"box": (180, 180, 180), "badge": (100, 100, 100), "text": (255, 255, 255)}, # Gray
    "helmet":        {"box": (40, 180, 40),   "badge": (40, 180, 40),   "text": (255, 255, 255)}, # Green
    "no_helmet":     {"box": (30, 30, 240),   "badge": (30, 30, 240),   "text": (255, 255, 255)}, # Bold Red
    "license_plate": {"box": (0, 200, 240),   "badge": (0, 200, 240),   "text": (0, 0, 0)},       # Amber Yellow
}

LABEL_CLEAN = {
    "moto": "Xe may",
    "person": "Nguoi",
    "helmet": "Co mu",
    "no_helmet": "KHONG MU",
    "license_plate": "Bien so",
}


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, scores, iou_thresh=0.45):
    if not boxes:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        if len(idxs) == 1:
            break
        cur_box = boxes[cur]
        other_boxes = boxes[idxs[1:]]
        ious = np.array([box_iou(cur_box, ob) for ob in other_boxes])
        idxs = idxs[1:][ious < iou_thresh]
    return keep


class BikeTracker:
    def __init__(self, max_lost=20, iou_thresh=0.20):
        self.next_id = 1
        self.tracks = {} # id: dict
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh

    def update(self, detected_bikes):
        # detected_bikes: list of dict {'box': [x1,y1,x2,y2], 'conf': float}
        matched_tracks = set()
        matched_dets = set()

        # Match existing tracks with detected bikes
        for tid, tinfo in self.tracks.items():
            best_iou = 0.0
            best_didx = -1
            for didx, d in enumerate(detected_bikes):
                if didx in matched_dets:
                    continue
                iou = box_iou(tinfo['box'], d['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_didx = didx
            if best_iou >= self.iou_thresh and best_didx != -1:
                tinfo['box'] = detected_bikes[best_didx]['box']
                tinfo['conf'] = detected_bikes[best_didx]['conf']
                tinfo['lost'] = 0
                matched_tracks.add(tid)
                matched_dets.add(best_didx)
            else:
                tinfo['lost'] += 1

        # Delete expired tracks
        expired = [tid for tid, tinfo in self.tracks.items() if tinfo['lost'] > self.max_lost]
        for tid in expired:
            del self.tracks[tid]

        # Create new tracks for unmatched detections
        for didx, d in enumerate(detected_bikes):
            if didx not in matched_dets:
                tid = f"XE-{self.next_id:02d}"
                self.next_id += 1
                self.tracks[tid] = {
                    'box': d['box'],
                    'conf': d['conf'],
                    'lost': 0,
                    'violated': False,
                    'plate': None,
                    'first_seen': d.get('time', 0.0)
                }

        return self.tracks


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


def run_system(input_source, model_path, conf_thresh, imgsz, save_path=None, show_window=False, fullscreen=False, max_frames=0):
    print("=" * 60)
    print("KHOI DONG HE THONG GIAM SAT GIAO THONG & DANH SACH VI PHAM")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Dang tai mo hinh nhan dien...")
    custom_model = YOLO(model_path)
    coco_model = YOLO("yolo11n.pt") # High-recall detector for motorcycle and person

    source = os.path.normpath(input_source) if not str(input_source).isdigit() else int(input_source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate HD render resolution so video fills full screen
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
        print(f"Luu video ket qua vao: {save_path}")

    # Violation records directory & CSV
    log_dir = r"D:\hoclamAI\giaothong\runs"
    os.makedirs(log_dir, exist_ok=True)
    snap_dir = os.path.join(log_dir, "violation_snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, "violation_registry.csv")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["STT", "Thoi_Gian_Giay", "Ma_Xe", "Bien_So", "Hanh_Vi_Vi_Pham", "Do_Tin_Cay", "Anh_Bang_Chung"])

    tracker = BikeTracker(max_lost=25, iou_thresh=0.20)
    violation_registry = [] # List of violation dicts
    total_violations = 0
    frame_idx = 0
    t0 = time.time()
    is_paused = False
    is_fullscreen = fullscreen

    print("\n--- PHIM TAT DIEU KHIEN ---")
    print("  [F]     : Bat / Tat Toan man hinh (Fullscreen)")
    print("  [SPACE] : Tam dung / Tiep tuc phat")
    print("  [Q/ESC] : Thoat chuong trinh")
    print("----------------------------\n")

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames > 0 and frame_idx > max_frames:
                print(f"Da dat gioi han {max_frames} khung hinh yeu cau.")
                break

            if (src_w, src_h) != (render_w, render_h):
                frame = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_LINEAR)

            curr_sec = frame_idx / fps

            # 1. Dual High-Recall Detections
            # COCO: Motorcycle (3) & Person (0)
            r_coco = coco_model.predict(frame, conf=0.22, imgsz=imgsz, verbose=False)[0]
            detected_bikes_raw = []
            detected_people = []
            for b in r_coco.boxes:
                cls_id = int(b.cls[0])
                c = float(b.conf[0])
                bx = [int(v) for v in b.xyxy[0]]
                if cls_id == 3: # motorcycle
                    detected_bikes_raw.append({"box": bx, "conf": c, "time": curr_sec})
                elif cls_id == 0: # person
                    detected_people.append({"box": bx, "conf": c})

            # Custom model: Helmet (1), No_Helmet (2), License_Plate (3), Moto (0)
            r_custom = custom_model.predict(frame, conf=0.18, imgsz=imgsz, verbose=False)[0]
            detected_helmets = []
            detected_no_helmets = []
            detected_plates = []
            for b in r_custom.boxes:
                cls_id = int(b.cls[0])
                c = float(b.conf[0])
                bx = [int(v) for v in b.xyxy[0]]
                if cls_id == 0: # custom moto
                    detected_bikes_raw.append({"box": bx, "conf": c, "time": curr_sec})
                elif cls_id == 1: # helmet
                    detected_helmets.append({"box": bx, "conf": c})
                elif cls_id == 2: # no_helmet
                    detected_no_helmets.append({"box": bx, "conf": c})
                elif cls_id == 3: # license_plate
                    detected_plates.append({"box": bx, "conf": c})

            # NMS merge for motorcycles to avoid duplicate boxes
            if detected_bikes_raw:
                bike_boxes = [d["box"] for d in detected_bikes_raw]
                bike_scores = [d["conf"] for d in detected_bikes_raw]
                keep_idxs = nms_boxes(bike_boxes, bike_scores, iou_thresh=0.45)
                detected_bikes = [detected_bikes_raw[k] for k in keep_idxs]
            else:
                detected_bikes = []

            # 2. Update Tracker for Motorcycles
            active_tracks = tracker.update(detected_bikes)

            # 3. Associate License Plate & Violation to Tracked Bikes
            latest_plate_crop = None
            latest_violation_crop = None

            for tid, tinfo in active_tracks.items():
                if tinfo['lost'] > 0:
                    continue
                bx1, by1, bx2, by2 = tinfo['box']
                bw, bh = bx2 - bx1, by2 - by1

                # Extended search region around motorcycle for rider & plate
                ex1 = max(0, int(bx1 - 0.15 * bw))
                ey1 = max(0, int(by1 - 0.35 * bh)) # Look higher for rider head
                ex2 = min(render_w, int(bx2 + 0.15 * bw))
                ey2 = min(render_h, int(by2 + 0.15 * bh))

                # Find license plate on this motorcycle
                for p in detected_plates:
                    px1, py1, px2, py2 = p["box"]
                    # Center of plate inside bike search region
                    pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                    if ex1 <= pcx <= ex2 and ey1 <= pcy <= ey2:
                        tinfo['plate'] = p
                        if latest_plate_crop is None:
                            pw, ph = px2 - px1, py2 - py1
                            if pw > 6 and ph > 6:
                                latest_plate_crop = frame[max(0, py1-5):min(render_h, py2+5), max(0, px1-5):min(render_w, px2+5)]

                # Check violations (no_helmet) on this motorcycle
                bike_has_violation = False
                viol_conf = 0.0
                viol_box = None

                for nh in detected_no_helmets:
                    nx1, ny1, nx2, ny2 = nh["box"]
                    ncx, ncy = (nx1 + nx2) // 2, (ny1 + ny2) // 2
                    if ex1 <= ncx <= ex2 and ey1 <= ncy <= ey2:
                        bike_has_violation = True
                        viol_conf = max(viol_conf, nh["conf"])
                        viol_box = nh["box"]

                # If this bike violated and HAS NOT BEEN RECORDED YET -> +1 VIOLATION!
                if bike_has_violation and not tinfo['violated']:
                    tinfo['violated'] = True
                    total_violations += 1 # CUMULATIVE COUNTER +1

                    # Plate text formatting
                    if tinfo['plate'] is not None:
                        plate_str = f"Bien: {tinfo['plate']['conf']:.2f}"
                    else:
                        plate_str = "Chua thay bien sau"

                    snap_name = f"violation_{total_violations:03d}_{tid}_{frame_idx}.jpg"
                    snap_path = os.path.join(snap_dir, snap_name)

                    # Save evidence image (crop bike + rider)
                    snap_crop = frame[ey1:ey2, ex1:ex2]
                    if snap_crop.size > 0:
                        cv2.imwrite(snap_path, snap_crop)

                    # Register in table
                    rec = {
                        "stt": total_violations,
                        "time": f"{curr_sec:.1f}s",
                        "track_id": tid,
                        "plate": plate_str,
                        "type": "KHONG DOI MU",
                        "conf": f"{viol_conf * 100:.0f}%",
                        "snap": snap_name
                    }
                    violation_registry.append(rec)
                    print(f"==> PHAT HIEN VI PHAM MOI (+1) | #{total_violations} | Ma xe: {tid} | Thoi diem: {curr_sec:.1f}s | Bien so: {plate_str}")

                    # Append to CSV immediately
                    with open(csv_file, "a", newline="", encoding="utf-8") as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([rec["stt"], rec["time"], rec["track_id"], rec["plate"], rec["type"], rec["conf"], snap_name])

                if viol_box is not None and latest_violation_crop is None:
                    vx1, vy1, vx2, vy2 = viol_box
                    vw, vh = vx2 - vx1, vy2 - vy1
                    latest_violation_crop = frame[max(0, vy1 - int(0.2*vh)):min(render_h, vy2 + int(0.2*vh)),
                                                  max(0, vx1 - int(0.2*vw)):min(render_w, vx2 + int(0.2*vw))]

            # 4. Draw All Bounding Boxes & IDs
            # Draw Bikes (with Track ID)
            for tid, tinfo in active_tracks.items():
                if tinfo['lost'] > 0:
                    continue
                bx1, by1, bx2, by2 = tinfo['box']
                is_viol = tinfo['violated']
                col = (0, 0, 240) if is_viol else (230, 140, 20)
                thick = 3 if is_viol else 2
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, thick)

                lbl = f"{tid} {'[VI PHAM]' if is_viol else 'Xe may'}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                cv2.rectangle(frame, (bx1, max(0, by1 - 20)), (bx1 + tw + 6, max(0, by1)), col, -1)
                cv2.putText(frame, lbl, (bx1 + 3, max(15, by1 - 4)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw People
            for p in detected_people:
                px1, py1, px2, py2 = p["box"]
                cv2.rectangle(frame, (px1, py1), (px2, py2), (180, 180, 180), 1)

            # Draw Helmets
            for h in detected_helmets:
                hx1, hy1, hx2, hy2 = h["box"]
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (40, 180, 40), 2)
                cv2.putText(frame, f"Co mu {h['conf']:.2f}", (hx1, max(15, hy1 - 4)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.40, (40, 180, 40), 1, cv2.LINE_AA)

            # Draw No-Helmets (Red Alert)
            for nh in detected_no_helmets:
                nx1, ny1, nx2, ny2 = nh["box"]
                cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 0, 240), 3)
                (tw, th), _ = cv2.getTextSize("KHONG MU", cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                cv2.rectangle(frame, (nx1, max(0, ny1 - 20)), (nx1 + tw + 6, max(0, ny1)), (0, 0, 240), -1)
                cv2.putText(frame, f"KHONG MU {nh['conf']:.2f}", (nx1 + 3, max(15, ny1 - 4)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw License Plates
            for pl in detected_plates:
                lx1, ly1, lx2, ly2 = pl["box"]
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 200, 240), 2)
                cv2.rectangle(frame, (lx1, max(0, ly1 - 18)), (lx1 + 65, max(0, ly1)), (0, 200, 240), -1)
                cv2.putText(frame, "Bien so", (lx1 + 2, max(14, ly1 - 4)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

            # 5. Dashboard Card (Top-Left)
            dash_w, dash_h = 360, 95
            overlay = frame.copy()
            cv2.rectangle(overlay, (12, 12), (12 + dash_w, 12 + dash_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            badge_col = (0, 0, 240) if total_violations > 0 else (0, 180, 0)
            cv2.rectangle(frame, (12, 12), (12 + dash_w, 12 + dash_h), badge_col, 2)

            cv2.putText(frame, f"Khung hinh: {frame_idx}/{total_frames} ({curr_sec:.1f}s)", (22, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1)
            cv2.putText(frame, f"Xe may tren duong: {len(detected_bikes)}  |  Nguoi: {len(detected_people)}", (22, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (220, 220, 220), 1)

            # Cumulative counter display
            cv2.putText(frame, f"TONG VI PHAM TICH LUY: {total_violations}", (22, 88),
                        cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 80, 255) if total_violations > 0 else (80, 220, 80), 2)

            # 6. ON-SCREEN VIOLATION REGISTRY TABLE (Bottom-Left Sidebar)
            # Renders a sleek table showing latest recorded violations
            tbl_x, tbl_y = 12, 120
            tbl_w, tbl_h = 440, min(240, 36 + max(1, min(6, len(violation_registry))) * 28)
            t_overlay = frame.copy()
            cv2.rectangle(t_overlay, (tbl_x, tbl_y), (tbl_x + tbl_w, tbl_y + tbl_h), (20, 20, 20), -1)
            cv2.addWeighted(t_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (tbl_x, tbl_y), (tbl_x + tbl_w, tbl_y + tbl_h), (0, 0, 220), 2)

            # Table Header
            cv2.rectangle(frame, (tbl_x, tbl_y), (tbl_x + tbl_w, tbl_y + 28), (0, 0, 180), -1)
            cv2.putText(frame, "DANH SACH VI PHAM: KHONG DOI MU BAO HIEM", (tbl_x + 10, tbl_y + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1)

            if not violation_registry:
                cv2.putText(frame, "Chua phat hien vi pham nao tren tuyen duong.", (tbl_x + 15, tbl_y + 55),
                            cv2.FONT_HERSHEY_DUPLEX, 0.42, (160, 160, 160), 1)
            else:
                # Show last 6 violations
                display_recs = violation_registry[-6:]
                row_y = tbl_y + 52
                for r in display_recs:
                    row_str = f"#{r['stt']:02d} | {r['time']:>5s} | {r['track_id']:<5s} | {r['plate']:<16s} | KHONG MU"
                    cv2.putText(frame, row_str, (tbl_x + 8, row_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)
                    row_y += 28

            # 7. Picture-in-Picture (PiP Zoom at Bottom-Right)
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
            if key == ord('q') or key == 27:
                print("Dung boi nguoi dung.")
                break
            elif key == ord(' '):
                is_paused = not is_paused
                print("Tam dung..." if is_paused else "Tiep tuc...")
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("HOAN TAT PHAN TICH VIDEO GIAO THONG")
    print("=" * 60)
    print(f"Tong so khung hinh xu ly: {frame_idx}")
    print(f"TONG SO VI PHAM TICH LUY:  {total_violations}")
    print(f"Nhat ky vi pham luu tai:   {csv_file}")
    print(f"Anh bang chung luu tai:    {snap_dir}")
    if save_path:
        print(f"Video ket qua xuat ra tai: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"D:\hoclamAI\giaothong\video\hanoi_bach_mai_traffic.mp4")
    parser.add_argument("--model", default=r"D:\hoclamAI\giaothong\models\best_video_v4.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    run_system(args.input, args.model, args.conf, args.imgsz, args.save, args.show, args.fullscreen, args.max_frames)

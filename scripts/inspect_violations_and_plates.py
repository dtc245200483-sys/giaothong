"""Quét tìm các frame có biển số và người vi phạm trên cả 8 video:
- Lấy mẫu đều các video
- Lưu ảnh xe và ảnh biển số vào runs/violator_inspection/
- Chạy qua PlateCharRecognizer để kiểm tra kết quả nhận diện
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
from ultralytics import YOLO
from plate_char_recognizer import PlateCharRecognizer

ROOT = r"D:\hoclamAI\giaothong"
OUT_DIR = os.path.join(ROOT, "runs", "violator_inspection")
os.makedirs(OUT_DIR, exist_ok=True)

m_bike = YOLO(os.path.join(ROOT, "models", "yolo11s_motobike_openvino_model"), task='detect')
m_stage2 = YOLO(os.path.join(ROOT, "models", "yolo11s_helmet_lp_openvino_model"), task='detect')
recognizer = PlateCharRecognizer(os.path.join(ROOT, "models", "lp_char_classifier.pt"))

video_dir = os.path.join(ROOT, "video")
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]

print(f">> Quét 8 video ({len(video_files)} files) để tìm biển số xe và người vi phạm...")
saved_records = []

for vid_idx, vid_path in enumerate(video_files):
    vid_name = os.path.basename(vid_path)
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        continue
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Chọn 40 mốc thời gian đều nhau trong video
    step = max(10, total_frames // 40)
    print(f"\n[Video {vid_idx+1}] {vid_name[:24]} ({total_frames} frames, step={step})...", flush=True)

    found_in_video = 0
    for f_no in range(0, total_frames, step):
        if found_in_video >= 3:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        res_b = m_bike(frame, conf=0.35, imgsz=640, verbose=False)[0]

        for b_box in res_b.boxes:
            bx1, by1, bx2, by2 = map(int, b_box.xyxy[0].tolist())
            bw, bh = bx2 - bx1, by2 - by1
            if bw < 50 or bh < 60:
                continue

            crop_bike = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            if crop_bike.size == 0:
                continue

            res_s2 = m_stage2(crop_bike, conf=0.20, imgsz=640, verbose=False)[0]
            
            has_no_helmet = False
            has_plate = False
            plate_local = None

            for s_box in res_s2.boxes:
                c = int(s_box.cls[0].item())
                s_xyxy = s_box.xyxy[0].tolist()
                if c == 2:
                    has_no_helmet = True
                elif c == 0:
                    has_plate = True
                    plate_local = s_xyxy

            # Ưu tiên bắt xe có vi phạm HOẶC có biển số xe rõ ràng
            if has_no_helmet or (has_plate and bh > 120):
                found_in_video += 1
                case_type = "VI_PHAM" if has_no_helmet else "BIEN_SO_RO"
                
                # Trích xuất ảnh biển số
                if plate_local is not None:
                    px1 = max(0, int(bx1 + plate_local[0]))
                    py1 = max(0, int(by1 + plate_local[1]))
                    px2 = min(w, int(bx1 + plate_local[2]))
                    py2 = min(h, int(by1 + plate_local[3]))
                    plate_crop = frame[py1:py2, px1:px2]
                else:
                    f_py1 = max(0, by1 + int(0.60 * bh))
                    f_py2 = min(h, by2)
                    f_px1 = max(0, bx1 + int(0.20 * bw))
                    f_px2 = min(w, bx2 - int(0.20 * bw))
                    plate_crop = frame[f_py1:f_py2, f_px1:f_px2]

                pw, ph = (plate_crop.shape[1], plate_crop.shape[0]) if (plate_crop is not None and plate_crop.size > 0) else (0, 0)
                pred_text = recognizer.recognize_plate(plate_crop) if (plate_crop is not None and plate_crop.size > 0) else "KHONG_CO"

                img_prefix = f"case_{vid_idx+1}_{found_in_video}_{case_type}"
                bike_file = os.path.join(OUT_DIR, f"{img_prefix}_bike.jpg")
                plate_file = os.path.join(OUT_DIR, f"{img_prefix}_plate.jpg")
                cv2.imwrite(bike_file, crop_bike)
                if plate_crop is not None and plate_crop.size > 0:
                    cv2.imwrite(plate_file, plate_crop)

                record = {
                    'video': vid_name[:22],
                    'frame': f_no,
                    'type': case_type,
                    'bike_w_h': f"{bw}x{bh}",
                    'plate_w_h': f"{pw}x{ph}",
                    'plate_text': pred_text,
                    'plate_img': f"{img_prefix}_plate.jpg",
                    'bike_img': f"{img_prefix}_bike.jpg"
                }
                saved_records.append(record)
                print(f"  [+] Frame {f_no} [{case_type}] Xe: {bw}x{bh} px | Biển: {pw}x{ph} px -> Đoán: '{pred_text}'", flush=True)

    cap.release()

print("\n" + "=" * 80, flush=True)
print(f"KẾT QUẢ KIỂM TRA THỰC TẾ TRÊN 8 VIDEO: (Tìm thấy {len(saved_records)} mẫu)", flush=True)
print("=" * 80, flush=True)
for r in saved_records:
    print(f"Video {r['video']} | F:{r['frame']:5d} | {r['type']:10s} | Biển: {r['plate_w_h']:7s} | Text: {r['plate_text']}", flush=True)

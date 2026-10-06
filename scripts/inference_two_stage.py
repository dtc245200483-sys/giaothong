"""Two-Stage Traffic Violation Monitoring System:
Stage 1: Detect Motorbike (YOLO11s)
Stage 2: Detect Helmet, No-Helmet & License Plate on Motorbike Crop (YOLO11s)
Stage 3: License Plate Super-Resolution & OCR (PlateEnhancer + PaddleOCR)

Usage:
    python scripts/inference_two_stage.py --video video/1790209379270_9011771178293999674_9011771178293999674.mp4 \
        --model-bike models/yolo11s_motobike.pt \
        --model-helmet-lp models/yolo11s_helmet_lp.pt
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import re
from ultralytics import YOLO


def run_two_stage(
    video_path,
    model_bike_path,
    model_helmet_lp_path,
    conf_bike=0.35,
    conf_helmet_lp=0.25,
    imgsz_bike=960,
    imgsz_sub=640,
    save_path=None
):
    print("=" * 70)
    print("KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT GIAO THÔNG 2 TẦNG (TWO-STAGE YOLO11)")
    print("=" * 70)

    # 1. Load models
    # Fallback to general model if dedicated 2-stage models are not yet trained
    if not os.path.exists(model_bike_path):
        print(f"Lưu ý: Chưa tìm thấy {model_bike_path}, dùng tạm checkpoint hiện tại.")
        model_bike = YOLO(r"D:\hoclamAI\giaothong\models\best_video_v6.pt")
    else:
        print(f"Nạp Model Tầng 1 (Xe máy): {model_bike_path}")
        model_bike = YOLO(model_bike_path)

    if not os.path.exists(model_helmet_lp_path):
        print(f"Lưu ý: Chưa tìm thấy {model_helmet_lp_path}, dùng tạm checkpoint hiện tại.")
        model_sub = YOLO(r"D:\hoclamAI\giaothong\models\best_video_v6.pt")
    else:
        print(f"Nạp Model Tầng 2 (Mũ & Biển số): {model_helmet_lp_path}")
        model_sub = YOLO(model_helmet_lp_path)

    # Initialize OCR
    ocr_engine = None
    try:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(
            det_model_dir=r"D:\hoclamAI\giaothong\repos\lpr_nmthanh\model\ocr\det",
            rec_model_dir=r"D:\hoclamAI\giaothong\repos\lpr_nmthanh\model\ocr\rec",
            rec_char_dict_path=r"D:\hoclamAI\giaothong\repos\lpr_nmthanh\model\ocr\en_dict.txt",
            show_log=False, use_angle_cls=False
        )
        print("Hệ thống OCR sẵn sàng.")
    except Exception as e:
        print(f"Lỗi khởi tạo OCR: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {os.path.basename(video_path)} | {w}x{h} | {fps:.1f} FPS | {total_frames} frames")
    print("\nBắt đầu xử lý...")

    frame_count = 0
    start_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ==========================================================
        # TẦNG 1: Phát hiện xe máy trên toàn ảnh (imgsz cao: 960/1280)
        # ==========================================================
        res_bikes = model_bike.predict(frame, conf=conf_bike, imgsz=imgsz_bike, verbose=False)[0]

        bikes = []
        for b in res_bikes.boxes:
            cid = int(b.cls[0])
            # If model is motobike-only (cid=0), or if general model (cid=0 is moto)
            if cid == 0:
                bx = [int(v) for v in b.xyxy[0]]
                bikes.append({"box": bx, "conf": float(b.conf[0])})

        # ==========================================================
        # TẦNG 2: Cắt từng vùng xe máy và chạy Model Mũ & Biển số
        # ==========================================================
        for idx, bike in enumerate(bikes):
            bx1, by1, bx2, by2 = bike["box"]
            bw, bh = bx2 - bx1, by2 - by1

            # Mở rộng biên an toàn 15% để bắt trọn đầu người lái phía trên
            crop_y1 = max(0, by1 - int(0.40 * bh))
            crop_y2 = min(h, by2 + int(0.15 * bh))
            crop_x1 = max(0, bx1 - int(0.10 * bw))
            crop_x2 = min(w, bx2 + int(0.10 * bw))

            bike_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if bike_crop.size == 0:
                continue

            # Dự đoán trên vùng crop (độ phân giải hiệu dụng cao)
            res_sub = model_sub.predict(bike_crop, conf=conf_helmet_lp, imgsz=imgsz_sub, verbose=False)[0]

            helmets = []
            no_helmets = []
            plates = []

            for b in res_sub.boxes:
                sub_cid = int(b.cls[0])
                sub_c = float(b.conf[0])
                sx1, sy1, sx2, sy2 = [int(v) for v in b.xyxy[0]]

                # Chuyển đổi tọa độ từ crop về khung hình gốc
                orig_box = [crop_x1 + sx1, crop_y1 + sy1, crop_x1 + sx2, crop_y1 + sy2]

                # Map class theo cấu hình dataset mới:
                # Trong helmet_lincense_plate: 0: LP, 1: helmet, 2: no helmet
                # Trong best_video_v6 cũ: 0: moto, 1: helmet, 2: no_helmet, 3: license_plate
                if sub_cid == 1:
                    helmets.append(orig_box)
                elif sub_cid == 2:
                    no_helmets.append(orig_box)
                elif sub_cid in [0, 3]:  # LP / license_plate
                    plates.append(orig_box)

            # ==========================================================
            # TẦNG 3: Nhận diện biển số (Làm nét + OCR)
            # ==========================================================
            for px1, py1, px2, py2 in plates:
                plate_crop = frame[max(0, py1-2):min(h, py2+2), max(0, px1-2):min(w, px2+2)]
                if plate_crop.size > 0 and ocr_engine is not None:
                    try:
                        ocr_res = ocr_engine.ocr(plate_crop, cls=False)
                        if ocr_res and ocr_res[0]:
                            tokens = [line[1][0] for line in ocr_res[0] if line[1][1] > 0.40]
                            if tokens:
                                cleaned_plate = " ".join(tokens)
                                print(f"[Frame {frame_count}] Xe #{idx+1} Biển số: {cleaned_plate}")
                    except Exception:
                        pass

        if frame_count % 30 == 0:
            elapsed = time.time() - start_t
            fps_proc = frame_count / elapsed
            print(f"Đã xử lý {frame_count}/{total_frames} frames ({fps_proc:.1f} FPS)...")

    cap.release()
    total_time = time.time() - start_t
    print(f"\nHoàn tất! Tổng thời gian: {total_time:.1f}s | Tốc độ trung bình: {frame_count/total_time:.1f} FPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"D:\hoclamAI\giaothong\video\1790209379270_9011771178293999674_9011771178293999674.mp4")
    parser.add_argument("--model-bike", default=r"D:\hoclamAI\giaothong\models\yolo11s_motobike.pt")
    parser.add_argument("--model-helmet-lp", default=r"D:\hoclamAI\giaothong\models\yolo11s_helmet_lp.pt")
    args = parser.parse_args()

    run_two_stage(args.video, args.model_bike, args.model_helmet_lp)

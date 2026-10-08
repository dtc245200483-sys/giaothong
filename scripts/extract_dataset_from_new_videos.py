"""Trích xuất và tự động gán nhãn dữ liệu xe máy, mũ bảo hiểm, không đội mũ, biển số
từ 4 video đường phố Hà Nội mới (videoplayback_part1_5p đến part4_6p)
Tạo bộ dữ liệu dataset_v3_street để huấn luyện mô hình YOLO11s.
"""
import os
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = r"D:\hoclamAI\giaothong"
OUT_DIR = os.path.join(ROOT, "dataset_v3_street")
TRAIN_IMG = os.path.join(OUT_DIR, "train", "images")
TRAIN_LBL = os.path.join(OUT_DIR, "train", "labels")
VAL_IMG = os.path.join(OUT_DIR, "val", "images")
VAL_LBL = os.path.join(OUT_DIR, "val", "labels")

for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

# Nạp model OpenVINO siêu tốc
model_bike = YOLO(os.path.join(ROOT, "models", "yolo11s_motobike_openvino_model"))
model_stage2 = YOLO(os.path.join(ROOT, "models", "yolo11s_helmet_lp_openvino_model"))

# Tìm 4 video mới
new_videos = sorted(glob.glob(os.path.join(ROOT, "video", "videoplayback_part*.mp4")))
print(f">> Tìm thấy {len(new_videos)} video mới để trích xuất:")
for v in new_videos:
    print(f"   - {os.path.basename(v)}")

TARGET_TOTAL = 500
sample_idx = 0
val_ratio = 0.15

for vid_i, vid_path in enumerate(new_videos):
    if sample_idx >= TARGET_TOTAL:
        break
    cap = cv2.VideoCapture(vid_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[Video {vid_i+1}/{len(new_videos)}] {os.path.basename(vid_path)} ({total_f} frames)")
    
    # Bước nhảy đều khung hình
    step = max(15, total_f // 140)
    curr_f = 20
    
    while curr_f < total_f - 20 and sample_idx < TARGET_TOTAL:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_f)
        ret, frame = cap.read()
        if not ret:
            break
            
        orig_h, orig_w = frame.shape[:2]
        res_b = model_bike.predict(frame, conf=0.30, imgsz=640, verbose=False)[0]
        
        for bb in res_b.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in bb.xyxy[0]]
            bw, bh = bx2 - bx1, by2 - by1
            
            # Chỉ lấy các xe có kích thước đủ rõ ràng (bh > 90, bw > 45)
            if bh < 90 or bw < 45:
                continue
            # Lọc xe sát mép trên/dưới
            if by1 < int(orig_h * 0.10) or by2 > int(orig_h * 0.95):
                continue
                
            crop_y1 = max(0, by1 - int(0.18 * bh))
            crop_y2 = min(orig_h, by2 + int(0.08 * bh))
            crop_x1 = max(0, bx1 - int(0.08 * bw))
            crop_x2 = min(orig_w, bx2 + int(0.08 * bw))
            
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            ch, cw = crop.shape[:2]
            if ch < 80 or cw < 40:
                continue
                
            # Chạy model 2 dự đoán LP, helmet, no-helmet
            res2 = model_stage2.predict(crop, conf=0.20, imgsz=416, verbose=False)[0]
            
            annots = []
            has_valid_annot = False
            for b2 in res2.boxes:
                cls_id = int(b2.cls[0])
                c_conf = float(b2.conf[0])
                sx1, sy1, sx2, sy2 = [float(v) for v in b2.xyxy[0]]
                
                # Bắt nhãn tin cậy
                if c_conf >= 0.22:
                    xc = ((sx1 + sx2) / 2.0) / cw
                    yc = ((sy1 + sy2) / 2.0) / ch
                    w_norm = (sx2 - sx1) / cw
                    h_norm = (sy2 - sy1) / ch
                    annots.append(f"{cls_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                    has_valid_annot = True
                    
            if has_valid_annot:
                sample_idx += 1
                is_val = (sample_idx % int(1.0 / val_ratio) == 0)
                img_dest = VAL_IMG if is_val else TRAIN_IMG
                lbl_dest = VAL_LBL if is_val else TRAIN_LBL
                
                s_name = f"street_v{vid_i+1}_f{curr_f:05d}_{sample_idx:04d}"
                cv2.imwrite(os.path.join(img_dest, f"{s_name}.jpg"), crop)
                with open(os.path.join(lbl_dest, f"{s_name}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(annots) + "\n")
                    
                if sample_idx % 50 == 0:
                    print(f"   >> Đã trích xuất: {sample_idx}/{TARGET_TOTAL} mẫu...")
                    
                if sample_idx >= TARGET_TOTAL:
                    break
                    
        curr_f += step
    cap.release()

# Tạo data.yaml
yaml_content = f"""path: '{OUT_DIR}'
train: 'train/images'
val: 'val/images'

names:
  0: 'LP'
  1: 'helmet'
  2: 'no helmet'
"""
with open(os.path.join(OUT_DIR, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n=======================================================")
print(f"HOÀN TẤT TRÍCH XUẤT {sample_idx} MẪU HUẤN LUYỆN TẠI: {OUT_DIR}")
print(f"Train images: {len(os.listdir(TRAIN_IMG))}")
print(f"Val images:   {len(os.listdir(VAL_IMG))}")
print(f"File data.yaml đã sẵn sàng!")
print(f"=======================================================")

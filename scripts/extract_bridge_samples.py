import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO

# Thu muc dau ra
OUT_DIR = "D:/hoclamAI/giaothong/bridge_finetune"
IMG_DIR = os.path.join(OUT_DIR, "images")
LBL_DIR = os.path.join(OUT_DIR, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# Nap model stage 1 de tim xe may va stage 2 de pre-annotate
m_bike = YOLO("D:/hoclamAI/giaothong/models/yolo11s_motobike.pt")
m_stage2 = YOLO("D:/hoclamAI/giaothong/models/yolo11s_helmet_lp.pt")

vids = glob.glob("D:/hoclamAI/giaothong/video/*.mp4")
print(f"Tim thay {len(vids)} video.")

sample_count = 0
MAX_SAMPLES = 100  # Trich xuat khoang 100 mau xe may tieu bieu

# Lay mau deu dan tu cac video
for vid_idx, vid_path in enumerate(vids):
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[Video {vid_idx+1}/{len(vids)}] {os.path.basename(vid_path)} ({total_frames} frames)")
    
    # Moi video lay khoang 12-15 frame cach nhau
    step = max(30, total_frames // 25)
    frame_idx = 30
    
    while frame_idx < total_frames - 30 and sample_count < MAX_SAMPLES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        orig_h, orig_w = frame.shape[:2]
        
        # Phat hien xe may (Stage 1)
        res_bike = m_bike(frame, conf=0.35, imgsz=640, verbose=False)[0]
        
        for bb in res_bike.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in bb.xyxy[0]]
            bw, bh = bx2 - bx1, by2 - by1
            
            # Chi lay xe o khu vuc giua khung hinh (du to, du net, khong qua xa)
            if bh < 110 or bw < 45 or by1 < int(orig_h * 0.25) or by2 > int(orig_h * 0.88):
                continue
                
            # Cat dung ti le chuan nhu trong viewer
            crop_y1 = max(0, by1 - int(0.18 * bh))
            crop_y2 = min(orig_h, by2 + int(0.08 * bh))
            crop_cx1 = max(0, bx1 - int(0.06 * bw))
            crop_cx2 = min(orig_w, bx2 + int(0.06 * bw))
            
            crop = frame[crop_y1:crop_y2, crop_cx1:crop_cx2]
            ch, cw = crop.shape[:2]
            if ch < 80 or cw < 40:
                continue
            
            sample_count += 1
            sample_name = f"bridge_v{vid_idx+1}_{frame_idx:05d}_b{sample_count:03d}"
            img_path = os.path.join(IMG_DIR, f"{sample_name}.jpg")
            lbl_path = os.path.join(LBL_DIR, f"{sample_name}.txt")
            
            # Ghi anh crop
            cv2.imwrite(img_path, crop)
            
            # Pre-annotate bang Stage 2
            res2 = m_stage2(crop, conf=0.15, imgsz=640, verbose=False)[0]
            annotations = []
            for b2 in res2.boxes:
                cls_id = int(b2.cls[0])
                conf = float(b2.conf[0])
                sx1, sy1, sx2, sy2 = [float(v) for v in b2.xyxy[0]]
                
                # YOLO format: cls_id x_center y_center width height (normalized)
                xc = ((sx1 + sx2) / 2.0) / cw
                yc = ((sy1 + sy2) / 2.0) / ch
                w_norm = (sx2 - sx1) / cw
                h_norm = (sy2 - sy1) / ch
                
                annotations.append(f"{cls_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(annotations) + "\n")
                
            if sample_count >= MAX_SAMPLES:
                break
                
        frame_idx += step
    cap.release()

print(f"\nDa trich xuat thanh cong {sample_count} mau xe may chuan goc cau vuot tai: {IMG_DIR}")

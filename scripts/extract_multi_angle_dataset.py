import cv2
import os
import glob
from ultralytics import YOLO

# Thu muc luu dataset da goc re
OUT_DIR = "D:/hoclamAI/giaothong/dataset_cau_vuot_multi_angle"
IMG_DIR = os.path.join(OUT_DIR, "images")
LBL_DIR = os.path.join(OUT_DIR, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

m_bike = YOLO("D:/hoclamAI/giaothong/models/yolo11s_motobike.pt")
m_stage2 = YOLO("D:/hoclamAI/giaothong/models/yolo11s_helmet_lp.pt")

vids = glob.glob("D:/hoclamAI/giaothong/video/*.mp4")
print(f"Tim thay {len(vids)} video.")

sample_count = 0
TARGET_SAMPLES = 300

for vid_idx, vid_path in enumerate(vids):
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[Video {vid_idx+1}/{len(vids)}] {os.path.basename(vid_path)}")
    
    # Quet mat do cao hon o cac video co nhieu pha re goc nhu Video 7, Video 2, Video 6
    step = max(20, total_frames // 45)
    frame_idx = 40
    
    while frame_idx < total_frames - 30 and sample_count < TARGET_SAMPLES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        orig_h, orig_w = frame.shape[:2]
        res_bike = m_bike(frame, conf=0.30, imgsz=640, verbose=False)[0]
        
        for bb in res_bike.boxes:
            bx1, by1, bx2, by2 = [int(v) for v in bb.xyxy[0]]
            bw, bh = bx2 - bx1, by2 - by1
            
            # Lay rong ca cac xe re mep le (x sat bien trai/phai)
            if bh < 100 or bw < 40 or by1 < int(orig_h * 0.20) or by2 > int(orig_h * 0.90):
                continue
                
            crop_y1 = max(0, by1 - int(0.20 * bh))
            crop_y2 = min(orig_h, by2 + int(0.10 * bh))
            crop_cx1 = max(0, bx1 - int(0.08 * bw))
            crop_cx2 = min(orig_w, bx2 + int(0.08 * bw))
            
            crop = frame[crop_y1:crop_y2, crop_cx1:crop_cx2]
            ch, cw = crop.shape[:2]
            if ch < 80 or cw < 40:
                continue
            
            sample_count += 1
            name = f"multi_v{vid_idx+1}_{frame_idx:05d}_{sample_count:04d}"
            cv2.imwrite(os.path.join(IMG_DIR, f"{name}.jpg"), crop)
            
            # Pre-annotate
            res2 = m_stage2(crop, conf=0.20, imgsz=640, verbose=False)[0]
            annots = []
            for b2 in res2.boxes:
                cls_id = int(b2.cls[0])
                sx1, sy1, sx2, sy2 = [float(v) for v in b2.xyxy[0]]
                xc = ((sx1 + sx2) / 2.0) / cw
                yc = ((sy1 + sy2) / 2.0) / ch
                w_norm = (sx2 - sx1) / cw
                h_norm = (sy2 - sy1) / ch
                annots.append(f"{cls_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                
            with open(os.path.join(LBL_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(annots) + "\n")
                
            if sample_count >= TARGET_SAMPLES:
                break
                
        frame_idx += step
    cap.release()

print(f"\nDa trich xuat {sample_count} mau da goc re tai: {IMG_DIR}")

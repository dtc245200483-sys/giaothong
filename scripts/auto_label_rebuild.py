"""Kịch bản gán nhãn tự động chuẩn (Assisted Auto-Labeling) cho tập dữ liệu tái thiết
Nguồn: 812 ảnh trích xuất từ 8 video giao thông thực tế.
Các lớp chuẩn (4 classes):
  0: moto
  1: helmet
  2: no_helmet
  3: license_plate
"""
import os
import sys
import glob
import cv2
import numpy as np
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\hoclamAI\giaothong"
RAW_DIR = os.path.join(ROOT, "dataset_rebuild", "raw_frames")
OUT_DATASET = os.path.join(ROOT, "dataset_rebuild", "dataset_v4_clean")

TRAIN_IMG = os.path.join(OUT_DATASET, "train", "images")
TRAIN_LBL = os.path.join(OUT_DATASET, "train", "labels")
VAL_IMG = os.path.join(OUT_DATASET, "val", "images")
VAL_LBL = os.path.join(OUT_DATASET, "val", "labels")
GALLERY_NO_HELMET = os.path.join(ROOT, "dataset_rebuild", "gallery_no_helmet")
GALLERY_HELMET = os.path.join(ROOT, "dataset_rebuild", "gallery_helmet")

for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL, GALLERY_NO_HELMET, GALLERY_HELMET]:
    os.makedirs(d, exist_ok=True)

# Nạp các model hỗ trợ dán nhãn
print(">> Nạp mô hình hỗ trợ dán nhãn...")
m_unified = YOLO(os.path.join(ROOT, "repos", "vietnam_helmet_violation", "AI", "best_ver3.pt"))
m_bike = YOLO(os.path.join(ROOT, "models", "yolo11s_motobike.pt"))
m_stage2 = YOLO(os.path.join(ROOT, "models", "yolo11s_helmet_lp.pt"))

raw_images = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
print(f">> Tìm thấy {len(raw_images)} ảnh gốc để gán nhãn tự động.")

stats = {"moto": 0, "helmet": 0, "no_helmet": 0, "license_plate": 0}
val_step = 7 # ~14% ảnh vào tập val

def box_iou(b1, b2):
    # b1, b2: [x1, y1, x2, y2]
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

for idx, img_path in enumerate(raw_images):
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    h, w = frame.shape[:2]
    
    # 1. Dự đoán bằng mô hình unified chuyên giao thông Việt Nam
    res_u = m_unified(frame, conf=0.22, verbose=False)[0]
    
    # 2. Dự đoán phụ trợ xe máy để không sót xe
    res_b = m_bike(frame, conf=0.30, verbose=False)[0]
    
    # 3. Dự đoán phụ trợ mũ và biển số
    res_s2 = m_stage2(frame, conf=0.22, verbose=False)[0]
    
    detected_boxes = [] # [(cls_id, [x1, y1, x2, y2], conf)]
    
    # Lấy xe máy từ unified (cls 0: moto)
    for b in res_u.boxes:
        c = int(b.cls[0])
        conf = float(b.conf[0])
        xyxy = [float(v) for v in b.xyxy[0]]
        # 0: moto, 1: helmet, 2: no_helmet, 3: license_plate
        detected_boxes.append((c, xyxy, conf))
        
    # Bổ sung xe máy từ model bike nếu chưa có
    for b in res_b.boxes:
        xyxy = [float(v) for v in b.xyxy[0]]
        conf = float(b.conf[0])
        # Kiểm tra trùng với moto đã có
        is_dup = False
        for ex_c, ex_xy, _ in detected_boxes:
            if ex_c == 0 and box_iou(xyxy, ex_xy) > 0.45:
                is_dup = True
                break
        if not is_dup:
            detected_boxes.append((0, xyxy, conf))
            
    # Bổ sung nhãn từ model stage 2 (0: LP -> 3, 1: helmet -> 1, 2: no helmet -> 2)
    s2_map = {0: 3, 1: 1, 2: 2}
    for b in res_s2.boxes:
        c_raw = int(b.cls[0])
        mapped_c = s2_map.get(c_raw, 1)
        conf = float(b.conf[0])
        xyxy = [float(v) for v in b.xyxy[0]]
        is_dup = False
        for ex_c, ex_xy, _ in detected_boxes:
            if ex_c == mapped_c and box_iou(xyxy, ex_xy) > 0.40:
                is_dup = True
                break
        if not is_dup and conf >= 0.28:
            detected_boxes.append((mapped_c, xyxy, conf))
            
    # Lọc và chuẩn hóa nhãn cho ảnh
    clean_labels = []
    # Lấy danh sách các xe máy
    motos = [b for b in detected_boxes if b[0] == 0]
    
    for cls_id, xyxy, conf in detected_boxes:
        x1, y1, x2, y2 = xyxy
        bw, bh = x2 - x1, y2 - y1
        
        # Lọc kích thước quá bé hoặc bất thường
        if bw < 10 or bh < 10:
            continue
            
        # Lớp 1 (helmet) và Lớp 2 (no_helmet): kiểm tra xem có nằm ở vùng hợp lý (trên xe máy hoặc người)
        if cls_id in (1, 2):
            # Không lấy mũ/đầu ở quá thấp (chân xe máy)
            # Kiểm tra xem có xe máy nào ở gần không
            near_moto = any(
                (m_xy[0] - 30 <= x1 <= m_xy[2] + 30) and (m_xy[1] - 40 <= y1 <= m_xy[1] + (m_xy[3]-m_xy[1])*0.75)
                for _, m_xy, _ in motos
            )
            # Nếu không có xe máy nào gần trong phạm vi, nhưng conf cao thì vẫn giữ nếu ở nửa dưới camera
            if not near_moto and y2 < int(h * 0.25):
                continue # Loại bỏ bóng cây hoặc vật thể trên trời/tòa nhà
                
            # Lưu mẫu vào gallery để tiện kiểm tra bằng mắt
            if cls_id == 2: # No helmet
                cx1, cy1 = max(0, int(x1-5)), max(0, int(y1-5))
                cx2, cy2 = min(w, int(x2+5)), min(h, int(y2+5))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(GALLERY_NO_HELMET, f"nh_{base_name}_{int(x1)}_{int(y1)}.jpg"), crop)
            elif cls_id == 1 and idx % 20 == 0: # Helmet (lưu mẫu đại diện)
                cx1, cy1 = max(0, int(x1-5)), max(0, int(y1-5))
                cx2, cy2 = min(w, int(x2+5)), min(h, int(y2+5))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    cv2.imwrite(os.path.join(GALLERY_HELMET, f"h_{base_name}_{int(x1)}_{int(y1)}.jpg"), crop)

        # Chuyển sang định dạng chuẩn YOLO: <class> <x_center> <y_center> <width> <height>
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        norm_w = bw / w
        norm_h = bh / h
        
        # Giới hạn giá trị 0..1
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        norm_w = max(0.001, min(1.0, norm_w))
        norm_h = max(0.001, min(1.0, norm_h))
        
        clean_labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        c_names = ["moto", "helmet", "no_helmet", "license_plate"]
        stats[c_names[cls_id]] += 1
        
    # Phân bổ train / val
    is_val = (idx % val_step == 0)
    target_img_dir = VAL_IMG if is_val else TRAIN_IMG
    target_lbl_dir = VAL_LBL if is_val else TRAIN_LBL
    
    # Lưu ảnh và file nhãn txt
    cv2.imwrite(os.path.join(target_img_dir, img_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(target_lbl_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(clean_labels) + "\n")
        
    if (idx + 1) % 100 == 0 or (idx + 1) == len(raw_images):
        print(f"   -> Đã xử lý {idx+1}/{len(raw_images)} ảnh...")

# Tạo file data.yaml chuẩn
yaml_content = f"""path: {OUT_DATASET}
train: train/images
val: val/images

nc: 4
names:
  0: moto
  1: helmet
  2: no_helmet
  3: license_plate
"""

with open(os.path.join(OUT_DATASET, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

train_count = len(os.listdir(TRAIN_IMG))
val_count = len(os.listdir(VAL_IMG))
no_helmet_crops = len(os.listdir(GALLERY_NO_HELMET))

print("\n=======================================================")
print("HOÀN TẤT GÁN NHÃN BỘ DỮ LIỆU TÁI THIẾT (DATASET v4 CLEAN)!")
print(f"Tổng ảnh Train: {train_count} | Tổng ảnh Val: {val_count}")
print(f"Thống kê số lượng đối tượng gán nhãn:")
print(f"  - Xe máy (moto):         {stats['moto']}")
print(f"  - Mũ bảo hiểm (helmet):  {stats['helmet']}")
print(f"  - KHÔNG ĐỘI MŨ (no_helmet): {stats['no_helmet']} (Đã trích xuất {no_helmet_crops} mẫu vào gallery_no_helmet)")
print(f"  - Biển số xe (license_plate): {stats['license_plate']}")
print(f"Đường dẫn dataset: {OUT_DATASET}")
print(f"File cấu hình: {os.path.join(OUT_DATASET, 'data.yaml')}")
print("=======================================================")

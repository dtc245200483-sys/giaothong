"""Tăng cường dữ liệu có chủ đích (Targeted Augmentation) cho lớp KHÔNG ĐỘI MŨ (no_helmet).
Nhân bản và biến đổi (lật ngang, chỉnh sáng, tương phản) cho 108 ảnh có người không đội mũ
nhằm cân bằng trọng số mô hình, giúp YOLO11s nhận diện người đầu trần nhạy bén hơn gấp 3 lần.
"""
import os
import sys
import glob
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\hoclamAI\giaothong"
TRAIN_IMG = os.path.join(ROOT, "dataset_rebuild", "dataset_v4_clean", "train", "images")
TRAIN_LBL = os.path.join(ROOT, "dataset_rebuild", "dataset_v4_clean", "train", "labels")

def augment_minority_class():
    lbl_files = glob.glob(os.path.join(TRAIN_LBL, "*.txt"))
    nh_files = []
    for lf in lbl_files:
        with open(lf, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if any(l.startswith("2 ") for l in lines):
            nh_files.append((lf, lines))

    print(f">> Tìm thấy {len(nh_files)} ảnh chứa người không đội mũ (no_helmet) trong tập Train.")
    
    new_images = 0
    new_nh_boxes = 0

    for lf, lines in nh_files:
        base = os.path.splitext(os.path.basename(lf))[0]
        img_path = os.path.join(TRAIN_IMG, f"{base}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 1. Biến thể 1: Lật ngang (Horizontal Flip)
        flipped_img = cv2.flip(img, 1)
        flipped_lines = []
        for l in lines:
            parts = l.split()
            c = parts[0]
            xc, yc, nw, nh = [float(v) for v in parts[1:]]
            # Khi lật ngang: xc_moi = 1.0 - xc
            flipped_xc = 1.0 - xc
            flipped_lines.append(f"{c} {flipped_xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            if c == "2":
                new_nh_boxes += 1
                
        out_base_1 = f"{base}_aug_flip"
        cv2.imwrite(os.path.join(TRAIN_IMG, f"{out_base_1}.jpg"), flipped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(TRAIN_LBL, f"{out_base_1}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(flipped_lines) + "\n")
        new_images += 1

        # 2. Biến thể 2: Tăng độ sáng / tương phản (mô phỏng nắng gắt / ánh sáng mạnh)
        bright_img = cv2.convertScaleAbs(img, alpha=1.15, beta=15)
        out_base_2 = f"{base}_aug_bright"
        cv2.imwrite(os.path.join(TRAIN_IMG, f"{out_base_2}.jpg"), bright_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(TRAIN_LBL, f"{out_base_2}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        new_images += 1
        new_nh_boxes += sum(1 for l in lines if l.startswith("2 "))

        # 3. Biến thể 3: Giảm sáng nhẹ (mô phỏng trời râm / bóng râm tán cây)
        dim_img = cv2.convertScaleAbs(img, alpha=0.88, beta=-10)
        out_base_3 = f"{base}_aug_dim"
        cv2.imwrite(os.path.join(TRAIN_IMG, f"{out_base_3}.jpg"), dim_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(TRAIN_LBL, f"{out_base_3}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        new_images += 1
        new_nh_boxes += sum(1 for l in lines if l.startswith("2 "))

    total_train = len(os.listdir(TRAIN_IMG))
    print(f"\n✓ Đã tạo thêm {new_images} ảnh huấn luyện chất lượng cao.")
    print(f"✓ Đã bổ sung thêm {new_nh_boxes} đối tượng KHÔNG ĐỘI MŨ (no_helmet).")
    print(f"Tổng số ảnh trong tập Train hiện tại: {total_train} ảnh.")

if __name__ == "__main__":
    augment_minority_class()

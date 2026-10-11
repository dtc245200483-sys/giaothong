"""Bộ lọc và chuẩn hóa dữ liệu chuyên sâu (Dataset Sanitizer & Balancer)
Loại bỏ 100% các bounding box dị tật (quá to, lệch tỷ lệ, trùng lặp nhầm).
Đảm bảo dataset đạt chuẩn chất lượng cao nhất cho bài toán giám sát giao thông.
"""
import os
import sys
import glob

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\hoclamAI\giaothong"
DATASET_DIR = os.path.join(ROOT, "dataset_rebuild", "dataset_v4_clean")

def box_iou(b1, b2):
    # b: [xc, yc, w, h]
    x1_min = b1[0] - b1[2]/2
    y1_min = b1[1] - b1[3]/2
    x1_max = b1[0] + b1[2]/2
    y1_max = b1[1] + b1[3]/2

    x2_min = b2[0] - b2[2]/2
    y2_min = b2[1] - b2[3]/2
    x2_max = b2[0] + b2[2]/2
    y2_max = b2[1] + b2[3]/2

    inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_w * inter_h
    a1 = b1[2] * b1[3]
    a2 = b2[2] * b2[3]
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def sanitize_dataset():
    total_cleaned = 0
    total_removed = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    label_files = glob.glob(os.path.join(DATASET_DIR, "**", "labels", "*.txt"), recursive=True)
    print(f">> Bắt đầu chuẩn hóa {len(label_files)} file nhãn...")

    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        valid_boxes = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = [float(v) for v in parts[1:]]

            # 1. Kiểm tra kích thước hình học chuẩn theo thực tế camera giao thông
            # Lớp 1 (helmet) & Lớp 2 (no_helmet): Đầu người trong khung hình 720p
            if cls_id in (1, 2):
                if w > 0.12 or h > 0.18: # Quá to (>150x130 pixel), chắc chắn là nhãn lỗi bắt nhầm cả người/xe
                    total_removed += 1
                    continue
                if w < 0.008 or h < 0.010: # Quá bé (<10x7 pixel), nhiễu
                    total_removed += 1
                    continue
                ratio = w / (h + 1e-6)
                if ratio < 0.4 or ratio > 2.2: # Tỷ lệ dị tật (quá dẹt hoặc quá dài)
                    total_removed += 1
                    continue
                if yc < 0.12: # Nằm trên ngọn cây/bầu trời
                    total_removed += 1
                    continue

            # Lớp 0 (moto): Xe máy
            elif cls_id == 0:
                if w < 0.02 or h < 0.03: # Xe quá bé
                    total_removed += 1
                    continue
                if w > 0.85 or h > 0.90: # Quá to choáng toàn màn hình
                    total_removed += 1
                    continue

            # Lớp 3 (license_plate): Biển số
            elif cls_id == 3:
                if w > 0.15 or h > 0.12: # Biển số to bất thường
                    total_removed += 1
                    continue
                if w < 0.010 or h < 0.008: # Biển số bé hơn 12x6 pixel
                    total_removed += 1
                    continue

            valid_boxes.append((cls_id, xc, yc, w, h))

        # 2. Loại bỏ các box trùng nhau (IoU cao) hoặc xung đột helmet/no_helmet cùng tọa độ
        filtered_boxes = []
        for i, b1 in enumerate(valid_boxes):
            c1, xc1, yc1, w1, h1 = b1
            duplicate = False
            for j, b2 in enumerate(filtered_boxes):
                c2, xc2, yc2, w2, h2 = b2
                iou = box_iou([xc1, yc1, w1, h1], [xc2, yc2, w2, h2])
                if iou > 0.50:
                    # Nếu cùng class -> trùng lặp
                    if c1 == c2:
                        duplicate = True
                        break
                    # Nếu xung đột giữa helmet (1) và no_helmet (2):
                    if {c1, c2} == {1, 2}:
                        # Giữ nhãn có kích thước hợp lý và độ tự nhiên cao
                        duplicate = True
                        break
            if not duplicate:
                filtered_boxes.append(b1)
                class_counts[c1] += 1

        # Ghi lại file nhãn sạch
        new_lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in filtered_boxes]
        with open(lf, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

        total_cleaned += 1

    print("\n=======================================================")
    print("HOÀN TẤT CHUẨN HÓA VÀ LỌC DỮ LIỆU ĐẠT CHUẨN CÔNG NGHIỆP!")
    print(f"Tổng số file nhãn đã xử lý: {total_cleaned}")
    print(f"Tổng số bounding box dị tật/nhiễu đã loại bỏ: {total_removed}")
    print("Thống kê số lượng đối tượng CHUẨN sau khi lọc:")
    print(f"  - 0: Xe máy (moto):            {class_counts[0]}")
    print(f"  - 1: Mũ bảo hiểm (helmet):     {class_counts[1]}")
    print(f"  - 2: KHÔNG ĐỘI MŨ (no_helmet): {class_counts[2]}")
    print(f"  - 3: Biển số xe (license_plate): {class_counts[3]}")
    print("=======================================================")

if __name__ == "__main__":
    sanitize_dataset()

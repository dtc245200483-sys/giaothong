"""Trích xuất khung hình thông minh từ 8 video giao thông thực tế.
Lọc bỏ các frame bị rung lắc/mờ, chọn các frame có mật độ phương tiện tốt.
Lưu vào dataset_rebuild/raw_frames/
"""
import os
import sys
import glob
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

VIDEO_DIR = r"D:\hoclamAI\giaothong\video"
OUTPUT_DIR = r"D:\hoclamAI\giaothong\dataset_rebuild\raw_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_blurry(image, threshold=60.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def extract_frames():
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f">> Tìm thấy {len(videos)} video giao thông gốc:")
    
    total_extracted = 0
    target_per_video = {
        0: 180, # Video 1 (dài nhất, nhiều xe)
        1: 180, # Video 2
        2: 90,  # Video 3
        3: 40,  # Video 4 (ngắn)
        4: 60,  # Video 5
        5: 160, # Video 6
        6: 120, # Video 7 (nhiều góc rẽ)
        7: 60   # Video 8
    }

    for idx, vpath in enumerate(videos):
        vname = os.path.basename(vpath)
        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        target = target_per_video.get(idx, 100)
        # Tính bước nhảy để phân bổ đều khắp thời lượng video
        step = max(int(fps * 1.2), total_frames // (target + 5))
        
        print(f"\n[{idx+1}/{len(videos)}] Xử lý: {vname} ({total_frames} frames, FPS: {fps:.1f})")
        print(f"   -> Bước nhảy: {step} frames (~{step/fps:.1f}s), Mục tiêu: ~{target} ảnh")

        vid_extracted = 0
        curr_frame = int(fps * 2.0) # Bỏ qua 2 giây đầu lúc camera vừa khởi động
        
        prev_gray = None
        while curr_frame < (total_frames - int(fps * 2.0)) and vid_extracted < target:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Kiểm tra độ nét
            if is_blurry(frame):
                curr_frame += max(10, step // 2)
                continue
                
            # Kiểm tra khác biệt với frame trước (tránh ảnh đứng yên khi dừng đèn đỏ)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 90))
            if prev_gray is not None:
                diff = cv2.absdiff(gray_small, prev_gray)
                mean_diff = np.mean(diff)
                if mean_diff < 4.0: # Quá giống frame trước
                    curr_frame += step
                    continue
            prev_gray = gray_small
            
            # Lưu ảnh
            out_name = f"v{idx+1:02d}_f{curr_frame:06d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            vid_extracted += 1
            total_extracted += 1
            curr_frame += step

        cap.release()
        print(f"   ✓ Đã trích xuất {vid_extracted} ảnh nét từ video {idx+1}")

    print(f"\n=======================================================")
    print(f"TỔNG CỘNG ĐÃ TRÍCH XUẤT THÀNH CÔNG: {total_extracted} ẢNH CHẤT LƯỢNG CAO")
    print(f"Thư mục lưu: {OUTPUT_DIR}")
    print(f"=======================================================")

if __name__ == "__main__":
    extract_frames()

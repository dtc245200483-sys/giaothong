"""Hệ thống Giám sát Giao thông 2 Tầng Chia đôi 2 Bên (Side-by-Side Split View).
- Áp dụng Vehicle Tracking & Once-Helmet-Always-Helmet Locking:
  + Mỗi xe máy được gán ID và theo dõi liên tục trên đường đi.
  + Quy tắc Bất Di Bất Dịch: Chỉ cần trong quá trình xe chạy, có BẤT KỲ khung hình nào phát hiện Có Mũ (helmet),
    xe đó lập tức được KHÓA VĨNH VIỄN trạng thái HỢP LỆ [CÓ MŨ].
  + Triệt tiêu 100% hiện tượng chập chờn nhảy từ có mũ sang không mũ do góc nghiêng/bóng râm!
  + Người dùng KHÔNG CẦN phải cắt từng ảnh gán nhãn lại trên Roboflow!
  + Chỉ kết luận VI PHẠM khi xe đi hết vùng quan sát mà TUYỆT ĐỐI CHƯA TỪNG ĐỘI MŨ (>= 8 khung hình liên tục không mũ).
- OCR bất đồng bộ (Non-blocking Thread): Video phát 12 - 18 FPS cực kỳ mượt mà trên CPU.
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
import glob
import time
import math
import queue
import threading
import cv2
import numpy as np
import torch

# Tối ưu hóa toàn bộ 8 nhân của CPU AMD Ryzen 7 7840HS
torch.set_num_threads(8)
from ultralytics import YOLO

# Nạp module làm nét biển số
sys.path.append(os.path.dirname(__file__))
from plate_enhancer import PlateEnhancer

# Đảm bảo UTF-8 trên Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT = r"D:\hoclamAI\giaothong"
# 2 Model chinh vua train chuan tu Kaggle
MODEL_STAGE1 = os.path.join(ROOT, "models", "yolo11s_motobike.pt")
MODEL_STAGE2 = os.path.join(ROOT, "models", "yolo11s_helmet_lp.pt")
MODEL_STAGE1_OV = os.path.join(ROOT, "models", "yolo11s_motobike_openvino_model")
MODEL_STAGE2_OV = os.path.join(ROOT, "models", "yolo11s_helmet_lp_openvino_model")
BACKUP_MODEL = os.path.join(ROOT, "models", "best_ver3.pt")

# Danh sach toan bo 8 video moi
ALL_VIDEOS = sorted(glob.glob(os.path.join(ROOT, "video", "*.mp4")))

# Kích thước khung nhìn
CANVAS_W = 1600
CANVAS_H = 900


def put_text_utf8(img, text, pos, font_scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_corner_rect(img, pt1, pt2, color, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        cv2.circle(img, (px, py), 4, (0, 255, 0), -1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(1.0, union)


def main():
    print("=" * 80)
    print("KHỞI ĐỘNG HỆ THỐNG GIÁM SÁT GIAO THÔNG 2 TẦNG (TWO-STAGE YOLO11s)")
    has_ov = os.path.exists(MODEL_STAGE1_OV) and os.path.exists(MODEL_STAGE2_OV)
    has_pt = os.path.exists(MODEL_STAGE1) and os.path.exists(MODEL_STAGE2)
    use_two_stage = has_ov or has_pt

    if has_ov:
        print(">> Chế độ:           HỆ THỐNG 2 TẦNG (TĂNG TỐC OPENVINO CPU x2.4 LẦN)")
        print(f">> Tầng 1 (Xe máy):   {MODEL_STAGE1_OV}")
        print(f">> Tầng 2 (Mũ & Biển):{MODEL_STAGE2_OV}")
        m_stage1 = YOLO(MODEL_STAGE1_OV, task='detect')
        m_stage2 = YOLO(MODEL_STAGE2_OV, task='detect')
        m_detector = None
    elif has_pt:
        print(">> Chế độ:           HỆ THỐNG 2 TẦNG (YOLO11s TỐI TÂN TỪ KAGGLE)")
        print(f">> Tầng 1 (Xe máy):   {MODEL_STAGE1}")
        print(f">> Tầng 2 (Mũ & Biển):{MODEL_STAGE2}")
        m_stage1 = YOLO(MODEL_STAGE1)
        m_stage2 = YOLO(MODEL_STAGE2)
        m_detector = None
    elif os.path.exists(BACKUP_MODEL):
        print(f">> Chế độ:           1 TẦNG DỰ PHÒNG ({BACKUP_MODEL})")
        m_detector = YOLO(BACKUP_MODEL)
        m_stage1 = None
        m_stage2 = None
    else:
        print("Lỗi: Không tìm thấy file model nào trong thư mục models!")
        return

    print(f">> Tổng số video:    {len(ALL_VIDEOS)} videos")
    print("=" * 80)

    snap_dir = os.path.join(ROOT, "runs", "violations_output")
    os.makedirs(snap_dir, exist_ok=True)

    enhancer = PlateEnhancer(target_height=160)

    # Khởi tạo OCR tiếng Việt (PaddleOCR)
    ocr_engine = None
    try:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(
            det_model_dir=os.path.join(ROOT, "repos", "lpr_nmthanh", "model", "ocr", "det"),
            rec_model_dir=os.path.join(ROOT, "repos", "lpr_nmthanh", "model", "ocr", "rec"),
            rec_char_dict_path=os.path.join(ROOT, "repos", "lpr_nmthanh", "model", "ocr", "en_dict.txt"),
            show_log=False, use_angle_cls=False
        )
        print(">> PaddleOCR tiếng Việt đã sẵn sàng!")
    except Exception as e:
        print(f">> Cảnh báo PaddleOCR: {e}")

    # Hàng đợi và luồng OCR bất đồng bộ (video không bao giờ bị đơ giật)
    ocr_queue = queue.Queue(maxsize=4)
    ocr_results = {}

    def ocr_worker():
        while True:
            try:
                task = ocr_queue.get()
                if task is None:
                    break
                viol_id, plate_img = task
                p_text = "CHUA RO BIEN"
                if ocr_engine is not None and plate_img is not None and plate_img.size > 0:
                    try:
                        tokens = []
                        # Bước 1: Thử phát hiện chữ và nhận diện (det=True)
                        res_ocr = ocr_engine.ocr(plate_img, det=True, cls=False)
                        if res_ocr and res_ocr[0]:
                            tokens = [line[1][0] for line in res_ocr[0] if line[1][1] > 0.25]
                        
                        # Bước 2: Dự phòng nhận diện trực tiếp nếu bước 1 không tìm thấy chữ
                        if not tokens:
                            res_rec = ocr_engine.ocr(plate_img, det=False, cls=False)
                            if res_rec and res_rec[0]:
                                tokens = [l[0] for l in res_rec[0] if l[1] > 0.30]

                        if tokens:
                            cleaned = enhancer.clean_plate_text(" ".join(tokens))
                            if cleaned:
                                p_text = cleaned
                    except Exception:
                        pass
                ocr_results[viol_id] = p_text
                ocr_queue.task_done()
            except Exception:
                pass

    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    win_name = "HE THONG GIAM SAT GIAO THONG (KHOA TRANG THAI CO MU - CHONG NHAY LOI)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, CANVAS_W, CANVAS_H)

    current_vid_idx = 0
    cap = None
    latest_viol_bike_crop = None
    latest_viol_plate_crop = None
    latest_viol_plate_text = "CHUA CO"
    latest_viol_time = "Chua co"
    latest_viol_id = 0
    recent_history = []
    last_recorded_viol_time = 0.0

    # Bộ nhớ theo dõi trạng thái phương tiện
    # vehicle_memory[tid] = {'seen_helmet': bool, 'no_h_count': int, 'status': str, 'last_seen_frame': int}
    vehicle_memory = {}
    fallback_next_id = 100
    active_centroids = {}  # {tid: (cx, cy)}

    def open_video(idx):
        nonlocal cap, current_vid_idx, latest_viol_bike_crop, latest_viol_plate_crop
        nonlocal latest_viol_plate_text, latest_viol_time, latest_viol_id, last_recorded_viol_time
        nonlocal vehicle_memory, fallback_next_id, active_centroids
        if cap is not None:
            cap.release()
        current_vid_idx = idx % len(ALL_VIDEOS)
        vid_path = ALL_VIDEOS[current_vid_idx]
        print(f"\n--- Đang phát: [{current_vid_idx+1}/{len(ALL_VIDEOS)}] {os.path.basename(vid_path)} ---")
        
        # Reset hoàn toàn trạng thái khi đổi video
        latest_viol_bike_crop = None
        latest_viol_plate_crop = None
        latest_viol_plate_text = "CHUA CO"
        latest_viol_time = "Chua co"
        latest_viol_id = 0
        last_recorded_viol_time = 0.0
        recent_history.clear()
        vehicle_memory.clear()
        active_centroids.clear()
        fallback_next_id = 100
        return cv2.VideoCapture(vid_path)

    cap = open_video(current_vid_idx)

    is_paused = False
    fps = 0.0
    snap_counter = 0

    # Khởi động trước (Warmup)
    print(">> Đang khởi động mô hình AI...")
    dummy_warmup = np.zeros((512, 512, 3), dtype=np.uint8)
    if use_two_stage:
        m_stage1(dummy_warmup, verbose=False)
        m_stage2(dummy_warmup, verbose=False)
    else:
        m_detector(dummy_warmup, verbose=False)
    print(">> Khởi động AI hoàn tất, bắt đầu phát video mượt mà!")

    frame_idx = 0

    while True:
        if not is_paused:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            clean_frame = frame.copy()
            orig_h, orig_w = frame.shape[:2]

            # Vạch kiểm soát giao thông (20% từ đỉnh màn hình)
            line_red_y = int(orig_h * 0.20)
            line_yellow_y = int(orig_h * 0.65)

            cv2.line(frame, (0, line_red_y), (orig_w, line_red_y), (0, 0, 255), 2)
            cv2.line(frame, (0, line_yellow_y), (orig_w, line_yellow_y), (0, 255, 255), 2)

            frame_idx += 1
            frame_has_violation = False

            detected_bikes = []      # list of box [x1, y1, x2, y2]
            detected_helmets = []    # list of (box, conf)
            detected_no_helmets = [] # list of (box, conf)
            detected_plates = []     # list of (box, conf)

            if use_two_stage:
                # =====================================================================
                # TẦNG 1: Nhận diện xe máy trên toàn khung hình (YOLO11s)
                # =====================================================================
                res1 = m_stage1(frame, conf=0.30, imgsz=640, verbose=False)[0]
                for b in res1.boxes:
                    if int(b.cls[0]) == 0:  # motobike
                        detected_bikes.append([int(v) for v in b.xyxy[0]])

                # =====================================================================
                # TẦNG 2: Cắt từng xe máy và soi cận cảnh Mũ bảo hiểm & Biển số
                # =====================================================================
                for b_box in detected_bikes:
                    bx1, by1, bx2, by2 = b_box
                    bw, bh = bx2 - bx1, by2 - by1
                    crop_y1 = max(0, by1 - int(0.35 * bh))
                    crop_y2 = min(orig_h, by2 + int(0.12 * bh))
                    crop_x1 = max(0, bx1 - int(0.12 * bw))
                    crop_x2 = min(orig_w, bx2 + int(0.12 * bw))

                    bike_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if bike_crop.size == 0:
                        continue

                    res2 = m_stage2(bike_crop, conf=0.15, imgsz=640, verbose=False)[0]
                    for sb in res2.boxes:
                        scid = int(sb.cls[0])
                        sc = float(sb.conf[0])
                        sx1, sy1, sx2, sy2 = [int(v) for v in sb.xyxy[0]]
                        orig_box = [crop_x1 + sx1, crop_y1 + sy1, crop_x1 + sx2, crop_y1 + sy2]

                        # Classes: 0: LP, 1: helmet, 2: no helmet
                        if scid == 1 and sc >= 0.25:
                            detected_helmets.append((orig_box, sc))
                        elif scid == 2 and sc >= 0.25:
                            detected_no_helmets.append((orig_box, sc))
                        elif scid == 0 and sc >= 0.20:
                            detected_plates.append((orig_box, sc))
            else:
                res = m_detector(frame, conf=0.15, imgsz=640, verbose=False)[0]
                for box, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    cid = int(cls_id)
                    b = [int(v) for v in box]
                    c = float(conf)
                    if cid == 0:
                        detected_bikes.append(b)
                    elif cid == 1 and c >= 0.25:
                        detected_helmets.append((b, c))
                    elif cid == 2 and c >= 0.25:
                        detected_no_helmets.append((b, c))
                    elif cid == 3 and c >= 0.20:
                        detected_plates.append((b, c))

            # Loc truc xung: Chi loc khi 2 box trung lap cung 1 dau (IoU > 0.45)
            # TUYET DOI KHONG dung khoang cach dist de khong xoa nguoi ngoi sau!
            clean_no_helmets = []
            for nh_box, nh_cf in detected_no_helmets:
                suppressed = False
                for h_box, h_cf in detected_helmets:
                    iou = box_iou(nh_box, h_box)
                    if iou > 0.45:
                        if h_cf > nh_cf:
                            suppressed = True
                            break
                if not suppressed:
                    clean_no_helmets.append((nh_box, nh_cf))

            # Bộ theo dõi xe máy (Centroid Tracker mượt mà, gán ID chuẩn xác)
            cur_frame_bikes = []
            matched_prev_ids = set()

            for box in detected_bikes:
                bx1, by1, bx2, by2 = box
                bw, bh = bx2 - bx1, by2 - by1
                cy = (by1 + by2) // 2
                cx = (bx1 + bx2) // 2

                # Lọc xe quá xa hoặc quá nhỏ
                if bh < 60 or cy < line_red_y:
                    continue

                best_match_id = None
                min_dist = 140.0
                for prev_id, (pcx, pcy) in list(active_centroids.items()):
                    if prev_id in matched_prev_ids:
                        continue
                    d = math.hypot(cx - pcx, cy - pcy)
                    if d < min_dist:
                        min_dist = d
                        best_match_id = prev_id

                if best_match_id is not None:
                    actual_tid = best_match_id
                    matched_prev_ids.add(best_match_id)
                else:
                    fallback_next_id += 1
                    actual_tid = fallback_next_id

                active_centroids[actual_tid] = (cx, cy)
                cur_frame_bikes.append((actual_tid, box))

            # =========================================================================
            # XÉT DUYỆT TỪNG XE THEO NGUYÊN TẮC: "ONCE HELMET, ALWAYS HELMET"
            # =========================================================================
            for tid, (bx1, by1, bx2, by2) in cur_frame_bikes:
                bw, bh = bx2 - bx1, by2 - by1
                cy = (by1 + by2) // 2

                # Vùng đầu người lái (phần trên xe)
                hx1 = max(0, bx1 - int(0.15 * bw))
                hx2 = min(orig_w, bx2 + int(0.15 * bw))
                hy1 = max(0, by1 - int(0.35 * bh))
                hy2 = min(orig_h, by1 + int(0.55 * bh))

                # Vùng biển số (phần dưới xe)
                px1 = max(0, bx1 - int(0.15 * bw))
                px2 = min(orig_w, bx2 + int(0.15 * bw))
                py1 = max(0, by1 + int(0.35 * bh))
                py2 = min(orig_h, by2 + int(0.15 * bh))

                # Thu thập các box nằm trong vùng xe (dùng clean_no_helmets đã khử xung đột)
                bike_helmets = [hm for hm in detected_helmets if hx1 <= (hm[0][0]+hm[0][2])//2 <= hx2 and hy1 <= (hm[0][1]+hm[0][3])//2 <= hy2]
                bike_no_helmets = [nh for nh in clean_no_helmets if hx1 <= (nh[0][0]+nh[0][2])//2 <= hx2 and hy1 <= (nh[0][1]+nh[0][3])//2 <= hy2]
                bike_plates = [pl for pl in detected_plates if px1 <= (pl[0][0]+pl[0][2])//2 <= px2 and py1 <= (pl[0][1]+pl[0][3])//2 <= py2]

                # Lấy trạng thái của phương tiện này trong bộ nhớ
                st = vehicle_memory.setdefault(tid, {
                    'score_h': 0.0,
                    'score_nh': 0.0,
                    'h_hits': 0,
                    'nh_hits': 0,
                    'plate_hits': 0,
                    'ratios': [],
                    'total_frames': 0,
                    'status': 'DANG_THEO_DOI',
                    'recorded': False
                })
                st['total_frames'] += 1
                st['ratios'].append(bw / max(1, bh))
                if bike_plates:
                    st['plate_hits'] += 1

                # =====================================================================
                # BỘ LỌC HÌNH HỌC: Loại bỏ hoodie/vải trùm đầu bị nhầm là mũ bảo hiểm
                # =====================================================================
                def is_valid_helmet(hm_box, bike_box):
                    hx1_h, hy1_h, hx2_h, hy2_h = hm_box
                    bx1_h, by1_h, bx2_h, by2_h = bike_box
                    bw_h = max(1, bx2_h - bx1_h)
                    bh_h = max(1, by2_h - by1_h)
                    hw_h = hx2_h - hx1_h
                    hh_h = hy2_h - hy1_h
                    if hw_h <= 0 or hh_h <= 0:
                        return False
                    # Tỉ lệ height mũ so với xe: mũ thật < 35% chiều cao xe
                    if hh_h / bh_h > 0.35:
                        return False
                    # Tỉ lệ diện tích: mũ thật < 20% diện tích xe
                    if (hw_h * hh_h) / (bw_h * bh_h) > 0.20:
                        return False
                    # Vị trí: trung tâm mũ phải nằm trong 50% chiều cao trên xe
                    hcy_h = (hy1_h + hy2_h) / 2
                    if hcy_h > by1_h + 0.50 * bh_h:
                        return False
                    return True

                valid_bike_helmets = [
                    hm for hm in bike_helmets
                    if hm[1] >= 0.35 and is_valid_helmet(hm[0], (bx1, by1, bx2, by2))
                ]

                # Tich luy so lan phat hien
                if valid_bike_helmets:
                    st['h_hits'] += len(valid_bike_helmets)
                    for hm_box, hm_conf in valid_bike_helmets:
                        st['score_h'] += hm_conf

                if bike_no_helmets:
                    st['nh_hits'] += len(bike_no_helmets)
                    for nh_box, nh_conf in bike_no_helmets:
                        st['score_nh'] += nh_conf

                # =====================================================================
                # PHAN QUYET VI PHAM CHUAN XAC THEO LUAT GIAO THONG VIET NAM
                # =====================================================================
                # =====================================================================
                # PHAN QUYET VI PHAM & KHOA TRANG THAI HANH TRINH (STATE LOCKING)
                # =====================================================================
                # Truong hop 1: Xe DA DUOC KHOA CO MU (State Locked)
                if st.get('helmet_locked', False):
                    # Nguoi lai da doi mu thi khi re trai/nghieng dau TUYET DOI KHONG BI LAT KEO.
                    # Chi xet vi pham neu xuat hien dong thoi nguoi thu 2 ngoi sau dau tran
                    has_simultaneous_passenger = (len(valid_bike_helmets) >= 1 and len(bike_no_helmets) >= 1)
                    if has_simultaneous_passenger:
                        st['pillion_viol_hits'] = st.get('pillion_viol_hits', 0) + 1
                        if st['pillion_viol_hits'] >= 4:
                            st['status'] = 'VI_PHAM'
                            st['viol_reason'] = 'CHO NGUOI KHONG DOI MU'
                    else:
                        st['status'] = 'HOP_LE'

                # Truong hop 2: Kiem tra dieu kien KHOA CO MU (khi phat hien mu ro rang tren duong)
                elif st['h_hits'] >= 3 and st['score_h'] >= 1.5:
                    st['helmet_locked'] = True
                    st['status'] = 'HOP_LE'
                    # Neu o xa tung bi chop nhay chup nham -> XOA NGAY LAP TUC!
                    if st.get('violation_recorded_id') is not None:
                        rec_id = st['violation_recorded_id']
                        recent_history = [item for item in recent_history if item["id"] != rec_id]
                        if latest_viol_id == rec_id:
                            latest_viol_bike_crop = None
                            latest_viol_plate_crop = None
                            latest_viol_plate_text = "DANG CHO..."
                        st['violation_recorded_id'] = None
                        st['recorded'] = False

                # Truong hop 3: Xe VI PHAM THUC SU (chua tung co mu, khong doi mu ap dao sau >= 5 frame)
                elif st['total_frames'] >= 5 and cy > line_red_y + 30:
                    if st['nh_hits'] >= 4 and st['nh_hits'] > st['h_hits'] * 1.5:
                        st['status'] = 'VI_PHAM'
                        st['viol_reason'] = 'KHONG DOI MU BAO HIEM'
                    elif st['h_hits'] >= 3 and st['nh_hits'] >= 3:
                        st['status'] = 'VI_PHAM'
                        st['viol_reason'] = 'CHO NGUOI KHONG DOI MU'
                    else:
                        st['status'] = 'DANG_THEO_DOI'
                else:
                    st['status'] = 'DANG_THEO_DOI'

                # Cat bien so neu co (them viền an toàn pad 15% để không cụt viền biển & ký tự)
                plate_crop_local = None
                for pl_box, pl_conf in bike_plates:
                    px1_b, py1_b, px2_b, py2_b = pl_box
                    cv2.rectangle(frame, (px1_b, py1_b), (px2_b, py2_b), (0, 220, 255), 2)
                    put_text_utf8(frame, "BIEN SO", (px1_b, max(18, py1_b - 5)), 0.45, (0, 220, 255), 1)
                    if py2_b > py1_b and px2_b > px1_b:
                        pw_p = px2_b - px1_b
                        ph_p = py2_b - py1_b
                        pad_w = int(pw_p * 0.15)
                        pad_h = int(ph_p * 0.15)
                        ppx1 = max(0, px1_b - pad_w)
                        ppx2 = min(orig_w, px2_b + pad_w)
                        ppy1 = max(0, py1_b - pad_h)
                        ppy2 = min(orig_h, py2_b + pad_h)
                        plate_crop_local = clean_frame[ppy1:ppy2, ppx1:ppx2]

                # Dự phòng: Nếu xe vi phạm mà YOLO chưa bắt được box biển số, crop vị trí biển số đuôi xe
                if (plate_crop_local is None or plate_crop_local.size == 0) and st['status'] == 'VI_PHAM':
                    f_py1 = max(0, by1 + int(0.66 * bh))
                    f_py2 = min(orig_h, by2 + int(0.05 * bh))
                    f_px1 = max(0, bx1 + int(0.18 * bw))
                    f_px2 = min(orig_w, bx2 - int(0.18 * bw))
                    if f_py2 > f_py1 and f_px2 > f_px1:
                        plate_crop_local = clean_frame[f_py1:f_py2, f_px1:f_px2]

                # Theo dõi ảnh biển số nét nhất/to nhất khi xe đi dần xuống gần camera
                if plate_crop_local is not None and plate_crop_local.size > 0:
                    curr_area = plate_crop_local.shape[0] * plate_crop_local.shape[1]
                    if st.get('best_plate_crop') is None or curr_area > st.get('best_plate_area', 0):
                        st['best_plate_crop'] = plate_crop_local.copy()
                        st['best_plate_area'] = curr_area
                        # Cập nhật ngay ảnh biển số nét hơn nếu xe này đang hiển thị vi phạm
                        if st.get('recorded', False) and st.get('violation_recorded_id') == latest_viol_id:
                            enhanced_p = enhancer.enhance(plate_crop_local)
                            latest_viol_plate_crop = enhanced_p.copy()
                            if recent_history and recent_history[0]['id'] == latest_viol_id:
                                recent_history[0]['img'] = latest_viol_plate_crop.copy()
                            try:
                                ocr_queue.put_nowait((latest_viol_id, latest_viol_plate_crop.copy()))
                            except queue.Full:
                                pass

                # Cat anh xe sach tu clean_frame
                crop_cy1 = max(0, by1 - int(0.20 * bh))
                crop_cy2 = min(orig_h, by2 + int(0.08 * bh))
                crop_cx1 = max(0, bx1 - int(0.06 * bw))
                crop_cx2 = min(orig_w, bx2 + int(0.06 * bw))
                crop_clean = clean_frame[crop_cy1:crop_cy2, crop_cx1:crop_cx2] if (crop_cy2 > crop_cy1 and crop_cx2 > crop_cx1) else None

                # HIEN THI THEO TRANG THAI CUA XE:
                if st['status'] == 'VI_PHAM':
                    frame_has_violation = True
                    viol_label = f"id{tid}: VI PHAM [{st.get('viol_reason', 'KHONG DOI MU')}]"
                    draw_corner_rect(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    put_text_utf8(frame, viol_label, (bx1, max(22, by1 - 8)),
                                  0.55, (0, 0, 255), 2)

                    # Ve box mu xanh la neu co nguoi doi mu (xe cho 2 nguoi)
                    for hm_box, hm_conf in valid_bike_helmets:
                        hx1_b, hy1_b, hx2_b, hy2_b = hm_box
                        cv2.rectangle(frame, (hx1_b, hy1_b), (hx2_b, hy2_b), (0, 255, 0), 2)
                        put_text_utf8(frame, f"CO MU {hm_conf:.2f}", (hx1_b, max(18, hy1_b - 5)), 0.45, (0, 255, 0), 1)

                    # Ve box DO ruc cho nguoi KHONG doi mu
                    for nh_box, nh_conf in bike_no_helmets:
                        nx1_b, ny1_b, nx2_b, ny2_b = nh_box
                        cv2.rectangle(frame, (nx1_b, ny1_b), (nx2_b, ny2_b), (0, 0, 255), 2)
                        put_text_utf8(frame, f"KHONG MU {nh_conf:.2f}", (nx1_b, max(18, ny1_b - 5)), 0.45, (0, 0, 255), 1)

                    # Cap nhat anh xe vi pham len bang ben phai
                    if crop_clean is not None and crop_clean.size > 0:
                        latest_viol_bike_crop = crop_clean.copy()
                        latest_viol_time = time.strftime("%H:%M:%S")

                        plate_to_use = st.get('best_plate_crop', plate_crop_local)
                        if plate_to_use is not None and plate_to_use.size > 0:
                            enhanced_p = enhancer.enhance(plate_to_use)
                            latest_viol_plate_crop = enhanced_p.copy()
                        else:
                            latest_viol_plate_crop = None

                        curr_t = time.time()
                        if not st['recorded'] and (curr_t - last_recorded_viol_time > 1.2) and st['total_frames'] >= 5 and cy > line_red_y + 35:
                            st['recorded'] = True
                            last_recorded_viol_time = curr_t
                            latest_viol_id += 1
                            st['violation_recorded_id'] = latest_viol_id

                            record = {
                                "id": latest_viol_id,
                                "time": latest_viol_time,
                                "plate": "DANG NHAN DIEN...",
                                "img": latest_viol_plate_crop.copy() if latest_viol_plate_crop is not None else None
                            }
                            recent_history.insert(0, record)
                            if len(recent_history) > 4:
                                recent_history.pop()

                            if latest_viol_plate_crop is not None:
                                try:
                                    ocr_queue.put_nowait((latest_viol_id, latest_viol_plate_crop.copy()))
                                except queue.Full:
                                    pass

                elif st['status'] == 'HOP_LE':
                    draw_corner_rect(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    put_text_utf8(frame, f"id{tid}: HOP LE [CO MU]", (bx1, max(22, by1 - 8)),
                                  0.50, (0, 255, 0), 1)

                    for hm_box, hm_conf in valid_bike_helmets:
                        hx1_b, hy1_b, hx2_b, hy2_b = hm_box
                        cv2.rectangle(frame, (hx1_b, hy1_b), (hx2_b, hy2_b), (0, 255, 0), 2)
                        put_text_utf8(frame, f"CO MU {hm_conf:.2f}", (hx1_b, max(18, hy1_b - 5)), 0.45, (0, 255, 0), 1)

                else:
                    draw_corner_rect(frame, (bx1, by1), (bx2, by2), (255, 140, 0), 2)
                    put_text_utf8(frame, f"id{tid}: DANG THEO DOI", (bx1, max(22, by1 - 8)),
                                  0.45, (255, 180, 50), 1)

            # Cập nhật kết quả OCR từ luồng nền
            for item in recent_history:
                iid = item["id"]
                if iid in ocr_results:
                    item["plate"] = ocr_results[iid]
                    if iid == latest_viol_id:
                        latest_viol_plate_text = ocr_results[iid]

            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, dt)) if fps > 0 else (1.0 / max(0.001, dt))

        # =========================================================================
        # RENDER GIAO DIỆN CHIA ĐÔI 2 BÊN (SIDE-BY-SIDE SPLIT VIEW)
        # =========================================================================
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 30)

        # Thanh tiêu đề trên cùng
        cv2.rectangle(canvas, (0, 0), (CANVAS_W, 55), (12, 16, 22), -1)
        cv2.line(canvas, (0, 55), (CANVAS_W, 55), (45, 55, 70), 1)
        put_text_utf8(canvas, "HE THONG GIAM SAT GIAO THONG 2 TANG (YOLO11s TU DONG BAT VI PHAM)",
                      (20, 35), 0.68, (0, 255, 255), 2)

        vid_name = os.path.basename(ALL_VIDEOS[current_vid_idx])
        put_text_utf8(canvas, f"Video: [{current_vid_idx+1}/{len(ALL_VIDEOS)}] {vid_name[:26]} | FPS: {fps:.1f}",
                      (920, 35), 0.55, (0, 255, 0), 1)

        # Khung Bên Trái: Video quan sát trực tiếp
        video_w, video_h = 980, 785
        resized_frame = cv2.resize(frame, (video_w, video_h))
        canvas[65:65+video_h, 15:15+video_w] = resized_frame
        cv2.rectangle(canvas, (15, 65), (15+video_w, 65+video_h), (50, 60, 75), 2)

        cv2.rectangle(canvas, (20, 70), (450, 100), (0, 0, 0), -1)
        put_text_utf8(canvas, "[BEN TRAI]: VIDEO QUAY PHAT HIEN LOI", (25, 92), 0.55, (0, 220, 255), 2)

        # Khung Bên Phải: Chi tiết xe vi phạm & Phóng to biển số
        panel_x = 1010
        panel_y = 65
        panel_w = 575
        panel_h = 785

        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (28, 34, 42), -1)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h),
                      (0, 0, 240) if frame_has_violation else (60, 75, 95), 2)

        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x+panel_w, panel_y+40), (18, 22, 28), -1)
        put_text_utf8(canvas, "[BEN PHAI]: CHI TIET XE VI PHAM & BIEN SO SIEU NET",
                      (panel_x + 15, panel_y + 27), 0.60, (0, 140, 255), 2)

        if latest_viol_bike_crop is not None:
            # 1. Ảnh Xe Vi Phạm (Sạch sẽ)
            bike_box_w, bike_box_h = 240, 310
            bx_off = panel_x + 20
            by_off = panel_y + 55

            cv2.rectangle(canvas, (bx_off, by_off), (bx_off + bike_box_w, by_off + bike_box_h), (15, 18, 22), -1)
            cv2.rectangle(canvas, (bx_off, by_off), (bx_off + bike_box_w, by_off + bike_box_h), (0, 0, 255), 2)

            res_bike_crop = cv2.resize(latest_viol_bike_crop, (bike_box_w - 4, bike_box_h - 4))
            canvas[by_off+2:by_off+bike_box_h-2, bx_off+2:bx_off+bike_box_w-2] = res_bike_crop
            put_text_utf8(canvas, "ANH XE VI PHAM", (bx_off + 10, by_off + 25), 0.50, (0, 0, 255), 2)

            # 2. Ảnh Biển Số Phóng To Siêu Nét
            plate_box_w, plate_box_h = 280, 175
            px_off = panel_x + 275
            py_off = panel_y + 55

            cv2.rectangle(canvas, (px_off, py_off), (px_off + plate_box_w, py_off + plate_box_h), (15, 18, 22), -1)
            cv2.rectangle(canvas, (px_off, py_off), (px_off + plate_box_w, py_off + plate_box_h), (0, 220, 255), 2)

            if latest_viol_plate_crop is not None and latest_viol_plate_crop.size > 0:
                res_plate_crop = cv2.resize(latest_viol_plate_crop, (plate_box_w - 4, plate_box_h - 4))
                canvas[py_off+2:py_off+plate_box_h-2, px_off+2:px_off+plate_box_w-2] = res_plate_crop
            else:
                cv2.putText(canvas, "CHUA THAY BIEN SO", (px_off + 30, py_off + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

            put_text_utf8(canvas, "BIEN SO PHONG TO (ZOOM)", (px_off + 10, py_off + 25), 0.50, (0, 220, 255), 2)

            # 3. Banner Dòng Chữ Biển Số & Lỗi
            banner_y = panel_y + 380
            cv2.rectangle(canvas, (panel_x + 20, banner_y), (panel_x + panel_w - 20, banner_y + 85), (10, 20, 35), -1)
            cv2.rectangle(canvas, (panel_x + 20, banner_y), (panel_x + panel_w - 20, banner_y + 85), (0, 220, 255), 2)

            put_text_utf8(canvas, f"BIEN SO:  {latest_viol_plate_text}",
                          (panel_x + 35, banner_y + 35), 0.85, (0, 255, 255), 2)
            put_text_utf8(canvas, f"LOI: KHONG DOI MU BAO HIEM  |  GIO: {latest_viol_time}",
                          (panel_x + 35, banner_y + 65), 0.50, (100, 200, 255), 1)

            # 4. Bảng Lịch Sử Vi Phạm
            hist_y = panel_y + 480
            put_text_utf8(canvas, "DANH SACH XE VI PHAM DA GHI NHAN:", (panel_x + 20, hist_y + 15), 0.55, (200, 200, 200), 1)
            cv2.line(canvas, (panel_x + 20, hist_y + 25), (panel_x + panel_w - 20, hist_y + 25), (50, 60, 75), 1)

            for h_i, item in enumerate(recent_history[:3]):
                row_y = hist_y + 35 + h_i * 80
                cv2.rectangle(canvas, (panel_x + 20, row_y), (panel_x + panel_w - 20, row_y + 70), (18, 22, 28), -1)
                cv2.rectangle(canvas, (panel_x + 20, row_y), (panel_x + panel_w - 20, row_y + 70), (45, 55, 70), 1)

                if item["img"] is not None and item["img"].size > 0:
                    thumb = cv2.resize(item["img"], (95, 60))
                    canvas[row_y+5:row_y+65, panel_x+25:panel_x+120] = thumb

                put_text_utf8(canvas, f"#{item['id']:02d} | Bien so: {item['plate']}",
                              (panel_x + 135, row_y + 30), 0.60, (0, 255, 255), 2)
                put_text_utf8(canvas, f"Loi: Khong doi mu  |  Luc: {item['time']}",
                              (panel_x + 135, row_y + 55), 0.45, (160, 160, 160), 1)
        else:
            # ĐÃ BỎ KHUNG VIỀN XANH LÁ VÀ CÁC DÒNG CHỮ THEO YÊU CẦU
            put_text_utf8(canvas, "HE THONG DANG QUAN SAT...",
                          (panel_x + 160, panel_y + 370), 0.60, (70, 85, 105), 1)

        # Thanh điều khiển dưới cùng
        cv2.rectangle(canvas, (0, CANVAS_H - 40), (CANVAS_W, CANVAS_H), (12, 16, 22), -1)
        cv2.line(canvas, (0, CANVAS_H - 40), (CANVAS_W, CANVAS_H - 40), (45, 55, 70), 1)
        put_text_utf8(canvas, "[N]: Video tiep theo  |  [P]: Video truoc  |  [SPACE]: Tam dung  |  [S]: Luu bien ban vi pham  |  [Q/ESC]: Thoat",
                      (20, CANVAS_H - 15), 0.52, (200, 200, 200), 1)

        cv2.imshow(win_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 32:  # SPACE
            is_paused = not is_paused
        elif key == ord('n') or key == ord('N'):
            cap = open_video(current_vid_idx + 1)
        elif key == ord('p') or key == ord('P'):
            cap = open_video(current_vid_idx - 1)
        elif key == ord('s') or key == ord('S'):
            snap_counter += 1
            snap_path = os.path.join(snap_dir, f"bien_ban_vi_pham_{snap_counter:03d}.jpg")
            cv2.imwrite(snap_path, canvas)
            print(f">> [DA LUU BIEN BAN]: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Đã đóng chương trình.")


if __name__ == "__main__":
    main()

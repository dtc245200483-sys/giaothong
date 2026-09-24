"""End-to-end demo: detect moto/helmet/no_helmet/license_plate + read plate via PaddleOCR.

Usage:
    python run_pipeline.py --input <image_or_video> [--save-dir <dir>] [--video-interval 10]
"""
import argparse
import importlib.util
import os
import time

import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR


def load_nm_utility():
    spec = importlib.util.spec_from_file_location(
        "nm_utility", r"D:\hoclamAI\giaothong\repos\lpr_nmthanh\utility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def process_image(frame, det, ocr, nm_util, save_dir=None, name="frame"):
    t0 = time.time()
    r = det(frame, verbose=False, conf=0.30)[0]
    boxes = r.boxes
    names = det.names

    # group objects by moto box (center-inside + margin)
    motos = [b for b in boxes if int(b.cls[0]) == 0]
    events = []
    for mb in motos:
        mx1, my1, mx2, my2 = [float(v) for v in mb.xyxy[0]]
        mw = mx2 - mx1
        mh = my2 - my1
        x1e, y1e, x2e, y2e = mx1 - 0.1 * mw, my1 - 0.1 * mh, mx2 + 0.1 * mw, my2 + 0.1 * mh

        helmets = []
        no_helmets = []
        plates = []
        for ob in boxes:
            cls_id = int(ob.cls[0])
            ox, oy = float(ob.xyxy[0][0]), float(ob.xyxy[0][1])
            if x1e <= ox <= x2e and y1e <= oy <= y2e:
                if cls_id == 1:  # helmet
                    helmets.append(ob)
                elif cls_id == 2 and float(ob.conf[0]) >= 0.40:  # no_helmet with confident score
                    no_helmets.append(ob)
                elif cls_id == 3:  # license_plate
                    plates.append(ob)

        # Helmet Priority: if helmet is detected on this motorcycle, rider is compliant
        if helmets:
            no_helmets = []

        plate_text = None
        plate_conf = None
        for pb in sorted(plates, key=lambda b: -float(b.conf[0]))[:2]:
            x1, y1, x2, y2 = [int(v) for v in pb.xyxy[0]]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = nm_util.adjust_image(crop)
            crop = nm_util.unwrap_image(crop)
            res = ocr.ocr(crop, cls=False)
            if res and res[0]:
                parts = [line[1] for line in res[0] if isinstance(line[-1], tuple)]
                if parts:
                    plate_text = " ".join(p[0] for p in parts)
                    plate_conf = sum(p[1] for p in parts) / len(parts)
                    break

        events.append({
            "moto_conf": float(mb.conf[0]),
            "no_helmet_count": len(no_helmets),
            "no_helmet_conf": max((float(b.conf[0]) for b in no_helmets), default=None),
            "plate": plate_text,
            "plate_conf": plate_conf,
        })

    dt = time.time() - t0
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = frame.copy()
        for mb in motos:
            x1, y1, x2, y2 = [int(v) for v in mb.xyxy[0]]
            color = (0, 0, 255)
            for ev in events:
                if ev["no_helmet_count"] > 0:
                    color = (0, 0, 255)
                else:
                    color = (0, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        for ev in events:
            if ev["plate"]:
                cv2.putText(out, f"{ev['plate']} {ev['plate_conf']:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(save_dir, name + ".jpg"), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return events, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--video-interval", type=int, default=10)
    ap.add_argument("--model", default=r"D:\hoclamAI\giaothong\models\best_video_v4.pt")
    args = ap.parse_args()

    model_path = args.model if os.path.exists(args.model) else r"D:\hoclamAI\giaothong\models\best_ver3.pt"
    print(f"Using detector model: {model_path}")
    det = YOLO(model_path)
    ocr = PaddleOCR(det_model_dir=r"D:\hoclamAI\giaothong\repos\lpr_nmthanh\model\ocr\det",
                    rec_model_dir=r"D:\hoclamAI\giaothong\repos\lpr_nmthanh\model\ocr\rec",
                    rec_char_dict_path=r"D:\hoclamAI\giaothong\repos\lpr_nmthanh\model\ocr\en_dict.txt",
                    show_log=False, use_angle_cls=False)
    nm_util = load_nm_utility()

    input_path = os.path.abspath(os.path.normpath(args.input))
    if not os.path.exists(input_path):
        print(f"Error: input file does not exist: {input_path}")
        return

    if os.path.splitext(input_path)[1].lower() in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: cv2 could not read image from {input_path}")
            return
        events, dt = process_image(frame, det, ocr, nm_util, args.save_dir,
                                   os.path.splitext(os.path.basename(input_path))[0])
        print(f"{input_path}: {dt:.2f}s")
        for i, ev in enumerate(events):
            print(" ", i, ev)
    else:
        cap = cv2.VideoCapture(input_path)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % args.video_interval != 0:
                continue
            events, dt = process_image(frame, det, ocr, nm_util, args.save_dir,
                                       f"{os.path.splitext(os.path.basename(args.input))[0]}_{n:06d}")
            violations = [ev for ev in events if ev["no_helmet_count"] > 0]
            print(f"frame {n}: {len(events)} moto, {len(violations)} violation, {dt:.2f}s")
            for ev in violations[:5]:
                print("   ", ev)
        cap.release()


if __name__ == "__main__":
    main()

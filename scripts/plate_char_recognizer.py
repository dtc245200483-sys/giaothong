"""Module nhận diện ký tự biển số xe máy Việt Nam (LP Character Recognizer)
Sử dụng mạng CNN huấn luyện trực tiếp từ bộ dữ liệu: licenseplate_digits
Tự động tách ký tự (Character Segmentation) và phân loại theo chuẩn 2 dòng của biển số xe máy.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn

# 32 ký tự chuẩn biển số Việt Nam
CLASS_LIST = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X',
    'Y', 'Z'
]


class LPCharCNN(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4

            nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class PlateCharRecognizer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = r"D:\hoclamAI\giaothong\models\lp_char_classifier.pt"
        self.model_path = model_path
        self.model = None
        self.class_list = CLASS_LIST
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                ckpt = torch.load(self.model_path, map_location='cpu')
                self.model = LPCharCNN(num_classes=len(self.class_list))
                if 'model_state' in ckpt:
                    self.model.load_state_dict(ckpt['model_state'])
                else:
                    self.model.load_state_dict(ckpt)
                self.model.eval()
                print(f">> [PlateCharRecognizer] Loaded LP character model: {os.path.basename(self.model_path)}")
            except Exception as e:
                print(f">> [PlateCharRecognizer] Error loading model: {e}")
                self.model = None
        else:
            print(f">> [PlateCharRecognizer] Model not found: {self.model_path}")

    def predict_char(self, char_gray):
        """Dự đoán 1 ký tự từ ảnh grayscale đã cắt."""
        if self.model is None or char_gray is None or char_gray.size == 0:
            return ""
        # Resize về 32x32 và chuẩn hóa [-1, 1]
        im = cv2.resize(char_gray, (32, 32), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(im).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            out = self.model(tensor)
            idx = out.argmax(dim=1).item()
        return self.class_list[idx]

    def recognize_plate(self, plate_img):
        """Tách và nhận diện chuỗi biển số từ ảnh crop biển số xe máy (Hỗ trợ định dạng chuẩn VN)."""
        if plate_img is None or plate_img.size == 0:
            return "CHUA RO BIEN"

        h, w = plate_img.shape[:2]
        if h == 0 or w == 0:
            return "CHUA RO BIEN"

        # 1. Phóng to ảnh để phân tách ký tự chuẩn xác
        scale = 130.0 / h
        target_w = max(50, int(round(w * scale)))
        plate_up = cv2.resize(plate_img, (target_w, 130), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_up, cv2.COLOR_BGR2GRAY)
        h_up, w_up = gray.shape

        # 2. Tăng cường độ tương phản cục bộ (CLAHE) để chữ số đen nổi bật rõ trên nền biển
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        cl = clahe.apply(gray)

        # 3. Phân ngưỡng thích ứng
        thresh = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        char_candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = ch / float(cw) if cw > 0 else 0
            area = cw * ch
            # Lọc kích thước chuẩn của ký tự trên biển số
            if 0.9 < aspect < 4.5 and 14 < ch < 65 and 5 < cw < 45 and area > 100:
                if x > 2 and y > 2 and (x + cw) < (w_up - 2) and (y + ch) < (h_up - 2):
                    char_candidates.append((x, y, cw, ch))

        # Khử các box lồng nhau (inner holes của số 0, 8, 9, B, D) và khử trùng lặp (NMS)
        char_candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        keep_boxes = []
        for b1 in char_candidates:
            x1, y1, w1, h1 = b1
            overlap = False
            for b2 in keep_boxes:
                x2, y2, w2, h2 = b2
                # Kiểm tra b1 có nằm lọt trong b2 không
                if x1 >= x2 - 2 and y1 >= y2 - 2 and (x1 + w1) <= (x2 + w2 + 2) and (y1 + h1) <= (y2 + h2 + 2):
                    overlap = True
                    break
                # Kiểm tra IoU
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    if inter / min(w1 * h1, w2 * h2) > 0.35:
                        overlap = True
                        break
            if not overlap:
                keep_boxes.append(b1)

        if len(keep_boxes) < 2:
            return "CHUA RO BIEN"

        # 4. Phân chia 2 dòng biển số
        mid_y = h_up * 0.48
        r1 = sorted([b for b in keep_boxes if (b[1] + b[3] / 2) < mid_y], key=lambda b: b[0])
        r2 = sorted([b for b in keep_boxes if (b[1] + b[3] / 2) >= mid_y], key=lambda b: b[0])

        # 5. Dự đoán ký tự qua mô hình CNN
        t1 = "".join([self.predict_char(cl[y:y+ch, x:x+cw]) for x, y, cw, ch in r1])
        t2 = "".join([self.predict_char(cl[y:y+ch, x:x+cw]) for x, y, cw, ch in r2])

        # 6. Sửa lỗi dựa trên quy chuẩn biển số xe máy Việt Nam (Domain rules)
        L2D = {'D': '0', 'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'P': '9'}
        D2L = {'0': 'D', '1': 'L', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'}

        cr1 = list(t1)
        if len(cr1) >= 1 and cr1[0] in L2D: cr1[0] = L2D[cr1[0]]
        if len(cr1) >= 2 and cr1[1] in L2D: cr1[1] = L2D[cr1[1]]
        if len(cr1) >= 3 and cr1[2] in D2L: cr1[2] = D2L[cr1[2]]
        if len(cr1) >= 4 and cr1[3] in L2D: cr1[3] = L2D[cr1[3]]

        cr2 = [L2D.get(c, c) for c in t2]

        if len(cr1) >= 3:
            sub = "".join(cr1[2:])
            out1 = f"{cr1[0]}{cr1[1]}-{sub}"
        else:
            out1 = "".join(cr1)

        if len(cr2) == 5:
            out2 = f"{cr2[0]}{cr2[1]}{cr2[2]}.{cr2[3]}{cr2[4]}"
        elif len(cr2) == 4:
            out2 = f"{cr2[0]}{cr2[1]}.{cr2[2]}{cr2[3]}"
        else:
            out2 = "".join(cr2)

        res = f"{out1} {out2}".strip()
        return res if len(res) >= 4 else "CHUA RO BIEN"


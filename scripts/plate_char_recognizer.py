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
        """Tách và nhận diện chuỗi biển số từ ảnh crop biển số xe máy."""
        if plate_img is None or plate_img.size == 0:
            return "CHUA RO BIEN"

        h, w = plate_img.shape[:2]
        if h == 0 or w == 0:
            return "CHUA RO BIEN"

        # 1. Phóng to ảnh để tách ký tự chuẩn xác
        scale = 120.0 / h
        target_w = max(40, int(round(w * scale)))
        plate_up = cv2.resize(plate_img, (target_w, 120), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_up, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 2. Ngưỡng thích ứng (Adaptive Threshold) tách nét chữ đen trên nền biển sáng
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

        # 3. Tìm đường bao bằng RETR_LIST để lấy các ký tự bên trong biển
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        char_candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = ch / float(cw) if cw > 0 else 0
            # Lọc các contour có kích thước chuẩn của ký tự xe máy
            if 1.0 < aspect < 4.0 and 15 < ch < 55 and 5 < cw < 36:
                if x > 3 and y > 3 and (x + cw) < (w - 3) and (y + ch) < (h - 3):
                    char_candidates.append((x, y, cw, ch))

        # Khử các box trùng lặp (Non-Maximum Suppression)
        keep_boxes = []
        for b1 in char_candidates:
            x1, y1, w1, h1 = b1
            overlap = False
            for b2 in keep_boxes:
                x2, y2, w2, h2 = b2
                xi1, yi1 = max(x1, x2), max(y1, y2)
                xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    if inter / min(w1*h1, w2*h2) > 0.40:
                        overlap = True
                        break
            if not overlap:
                keep_boxes.append(b1)

        if len(keep_boxes) < 2:
            return "CHUA RO BIEN"

        # 4. Phân chia 2 dòng biển số
        mid_y = h * 0.48
        row1_boxes = [b for b in keep_boxes if (b[1] + b[3] / 2) < mid_y]
        row2_boxes = [b for b in keep_boxes if (b[1] + b[3] / 2) >= mid_y]

        row1_boxes.sort(key=lambda b: b[0])
        row2_boxes.sort(key=lambda b: b[0])

        # 5. Nhận diện từng ký tự qua mô hình đã huấn luyện
        text_row1 = "".join([self.predict_char(gray[y:y+ch, x:x+cw]) for x, y, cw, ch in row1_boxes])
        text_row2 = "".join([self.predict_char(gray[y:y+ch, x:x+cw]) for x, y, cw, ch in row2_boxes])

        # 6. Định dạng chuỗi chuẩn biển số xe máy Việt Nam
        if text_row1 and text_row2:
            return f"{text_row1} {text_row2}"
        elif text_row2:
            return text_row2
        elif text_row1:
            return text_row1
        elif text_row2:
            return text_row2
        elif text_row1:
            return text_row1

        return "CHUA RO BIEN"

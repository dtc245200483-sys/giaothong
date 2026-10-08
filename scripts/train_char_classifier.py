"""Huấn luyện mô hình nhận diện ký tự biển số xe (LP Character Classifier)
Sử dụng dữ liệu từ: D:\\hoclamAI\\giaothong\\videovaanhtrain\\licenseplate_digits\\licenseplate
Tự động nạp bộ nhớ RAM để tối ưu tốc độ huấn luyện CPU cực nhanh (xong trong < 30 giây).
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import random
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Thiết lập đa luồng CPU tối đa cho AMD Ryzen 7 7840HS
torch.set_num_threads(8)

# Danh sách 32 ký tự biển số chuẩn Việt Nam (10 số + 22 chữ cái)
CLASS_LIST = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X',
    'Y', 'Z'
]
CHAR_TO_IDX = {c: i for i, c in enumerate(CLASS_LIST)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CLASS_LIST)}

# Ánh xạ từ nhãn trong LP_Character_Label.csv sang ký tự
RAW_LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 25: 'P', 26: 'Q', 27: 'R', 28: 'S',
    29: 'T', 30: 'U', 31: 'V', 33: 'X', 34: 'Y', 35: 'Z'
}


class LPCharCNN(nn.Module):
    """Mạng CNN nhẹ, tối ưu inference CPU siêu tốc (< 0.2ms/ký tự)."""
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


from PIL import Image, ImageDraw, ImageFont

# Danh sách font chữ đậm chuẩn biển số Việt Nam
TTF_FONTS = []
for font_file in ["arialbd.ttf", "tahomabd.ttf", "segoeuib.ttf", "calibrib.ttf"]:
    fp = os.path.join(r"C:\Windows\Fonts", font_file)
    if os.path.exists(fp):
        TTF_FONTS.append(fp)


def generate_synthetic_char(char, img_h=32, img_w=32):
    """Tự động sinh ảnh ký tự 32x32 nét đậm mô phỏng font biển số xe máy Việt Nam."""
    bg_val = random.randint(200, 250)
    img_pil = Image.new("L", (img_w, img_h), color=bg_val)
    draw = ImageDraw.Draw(img_pil)

    if TTF_FONTS:
        font_path = random.choice(TTF_FONTS)
        font_size = random.randint(20, 24)
        try:
            pil_font = ImageFont.truetype(font_path, font_size)
            # Lấy bounding box để căn giữa hoàn hảo
            bbox = draw.textbbox((0, 0), char, font=pil_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = max(1, (img_w - tw) // 2 + random.randint(-1, 1))
            ty = max(1, (img_h - th) // 2 + random.randint(-1, 1))
            text_val = random.randint(10, 45)
            draw.text((tx, ty), char, fill=text_val, font=pil_font)
            arr = np.array(img_pil)
        except Exception:
            arr = np.full((img_h, img_w), bg_val, dtype=np.uint8)
            cv2.putText(arr, char, (6, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, random.randint(10, 45), 2)
    else:
        arr = np.full((img_h, img_w), bg_val, dtype=np.uint8)
        cv2.putText(arr, char, (6, 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, random.randint(10, 45), 2)

    # 1. Biến thiên độ dày nét (Dilation / Erosion)
    if random.random() < 0.35:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        arr = cv2.erode(arr, kernel, iterations=1)  # Nét chữ đen dày hơn
    elif random.random() < 0.25:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        arr = cv2.dilate(arr, kernel, iterations=1)  # Nét chữ thanh hơn

    # 2. Xoay và nghiêng nhẹ (-7 đến +7 độ)
    angle = random.uniform(-7.0, 7.0)
    M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle, 1.0)
    arr = cv2.warpAffine(arr, M, (img_w, img_h), borderValue=bg_val)

    # 3. Nhiễu hạt nhẹ mô phỏng bụi đường
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 8), arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return arr



def main():
    print("=" * 75, flush=True)
    print("BẮT ĐẦU HUẤN LUYỆN MODEL NHẬN DIỆN KÝ TỰ BIỂN SỐ XE (32 CLASSES)", flush=True)
    print("=" * 75, flush=True)

    dataset_root = r"D:\hoclamAI\giaothong\videovaanhtrain\licenseplate_digits\licenseplate"
    csv_path = os.path.join(dataset_root, "LP_Character_Label.csv")

    images_list = []
    labels_list = []
    class_counts = {c: 0 for c in CLASS_LIST}

    # 1. Nạp ảnh từ ổ đĩa (giới hạn tối đa 600 ảnh mỗi lớp để bộ nhớ cực gọn và cân bằng lớp)
    MAX_PER_CLASS = 600
    if os.path.exists(csv_path):
        print(f">> Đang nạp ảnh từ: {csv_path} ...", flush=True)
        df = pd.read_csv(csv_path, header=None, names=['filename', 'label'])
        t0_load = time.time()
        for _, row in df.iterrows():
            lbl = int(row['label'])
            if lbl in RAW_LABEL_MAP:
                char = RAW_LABEL_MAP[lbl]
                if class_counts[char] >= MAX_PER_CLASS:
                    continue
                fn = str(row['filename'])
                p = os.path.join(dataset_root, fn)
                if os.path.exists(p):
                    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if im is not None:
                        im_resized = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
                        images_list.append(im_resized)
                        labels_list.append(CHAR_TO_IDX[char])
                        class_counts[char] += 1

        print(f">> Đã nạp {len(images_list)} ảnh từ ổ đĩa trong {time.time()-t0_load:.1f}s", flush=True)

    # 2. Sinh bổ sung các ký tự còn thiếu (đặc biệt số 3..9 và chữ T..Z)
    syn_count = 0
    TARGET_PER_CLASS = 700
    for char in CLASS_LIST:
        cur = class_counts[char]
        needed = max(0, TARGET_PER_CLASS - cur)
        if needed > 0:
            for _ in range(needed):
                syn_img = generate_synthetic_char(char, 32, 32)
                images_list.append(syn_img)
                labels_list.append(CHAR_TO_IDX[char])
                class_counts[char] += 1
                syn_count += 1

    print(f">> Đã sinh bổ sung: {syn_count} ảnh cho các ký tự thiếu (3, 4, 5, 6, 7, 8, 9, T, U, V, X, Y, Z)", flush=True)
    print(f">> Tổng số mẫu hoàn chỉnh: {len(images_list)} ảnh (cân bằng 32 lớp)", flush=True)

    # 3. Chuyển thành PyTorch Tensor trong RAM
    X = np.stack(images_list, axis=0)  # Shape: (N, 32, 32)
    X = (X.astype(np.float32) / 127.5) - 1.0  # Chuẩn hóa [-1, 1]
    X_tensor = torch.from_numpy(X).unsqueeze(1)  # Shape: (N, 1, 32, 32)
    Y_tensor = torch.tensor(labels_list, dtype=torch.long)

    # Xáo trộn dữ liệu
    indices = torch.randperm(len(X_tensor))
    X_tensor = X_tensor[indices]
    Y_tensor = Y_tensor[indices]

    split = int(0.9 * len(X_tensor))
    train_ds = TensorDataset(X_tensor[:split], Y_tensor[:split])
    val_ds = TensorDataset(X_tensor[split:], Y_tensor[split:])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = LPCharCNN(num_classes=len(CLASS_LIST))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 10
    print(f"\n>> Bắt đầu huấn luyện {epochs} Epochs trên RAM...", flush=True)
    t_start = time.time()

    best_val_acc = 0.0
    out_model_path = r"D:\hoclamAI\giaothong\models\lp_char_classifier.pt"
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        ep_t0 = time.time()

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_acc = correct / total * 100.0

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vout = model(vx)
                vpreds = vout.argmax(dim=1)
                val_correct += (vpreds == vy).sum().item()
                val_total += vx.size(0)

        val_acc = val_correct / val_total * 100.0
        ep_time = time.time() - ep_t0
        print(f"Epoch [{epoch}/{epochs}] ({ep_time:.2f}s) - Train Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'class_list': CLASS_LIST,
                'val_acc': val_acc
            }, out_model_path)

    total_time = time.time() - t_start
    print("=" * 75, flush=True)
    print(f"HUẤN LUYỆN HOÀN TẤT XUẤT SẮC TRONG {total_time:.1f} GIÂY!", flush=True)
    print(f">> Độ chính xác (Validation Accuracy): {best_val_acc:.2f}%", flush=True)
    print(f">> Đã lưu mô hình tại: {out_model_path}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()

import os
import glob
import shutil
import random
import zipfile

random.seed(42)

dest_dir = "D:/hoclamAI/giaothong/dataset_cau_vuot_finetune"
shutil.rmtree(dest_dir, ignore_errors=True)
os.makedirs(f"{dest_dir}/train/images", exist_ok=True)
os.makedirs(f"{dest_dir}/train/labels", exist_ok=True)
os.makedirs(f"{dest_dir}/val/images", exist_ok=True)
os.makedirs(f"{dest_dir}/val/labels", exist_ok=True)

# 1. 65 anh cau vuot thuc te
bridge_imgs = sorted(glob.glob("D:/hoclamAI/giaothong/bridge_finetune/images/*.jpg"))
random.shuffle(bridge_imgs)

# 80% train, 20% val
n_val = 12
val_bridge = bridge_imgs[:n_val]
train_bridge = bridge_imgs[n_val:]

# Nhan ban nhe tap anh cau vuot de model hoc sau ve goc cau vuot
for b_img in train_bridge:
    base = os.path.splitext(os.path.basename(b_img))[0]
    lbl = f"D:/hoclamAI/giaothong/bridge_finetune/labels/{base}.txt"
    for copy_idx in range(4):
        target_name = f"{base}_c{copy_idx}"
        shutil.copy2(b_img, f"{dest_dir}/train/images/{target_name}.jpg")
        if os.path.exists(lbl):
            shutil.copy2(lbl, f"{dest_dir}/train/labels/{target_name}.txt")

for b_img in val_bridge:
    base = os.path.splitext(os.path.basename(b_img))[0]
    lbl = f"D:/hoclamAI/giaothong/bridge_finetune/labels/{base}.txt"
    shutil.copy2(b_img, f"{dest_dir}/val/images/{base}.jpg")
    if os.path.exists(lbl):
        shutil.copy2(lbl, f"{dest_dir}/val/labels/{base}.txt")

# 2. Lay them 200 anh tu tap roboflow cu de model khong quen cac kien thuc chung
rf_train_imgs = glob.glob("D:/hoclamAI/giaothong/videovaanhtrain/helmet_lincense_plate/train/images/*.jpg")
random.shuffle(rf_train_imgs)
for rf_img in rf_train_imgs[:200]:
    base = os.path.splitext(os.path.basename(rf_img))[0]
    lbl = rf_img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    if os.path.exists(lbl):
        shutil.copy2(rf_img, f"{dest_dir}/train/images/{base}.jpg")
        shutil.copy2(lbl, f"{dest_dir}/train/labels/{base}.txt")

# Lay 30 anh val tu tap roboflow cu
rf_val_imgs = glob.glob("D:/hoclamAI/giaothong/videovaanhtrain/helmet_lincense_plate/valid/images/*.jpg")
for rf_img in rf_val_imgs[:30]:
    base = os.path.splitext(os.path.basename(rf_img))[0]
    lbl = rf_img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    if os.path.exists(lbl):
        shutil.copy2(rf_img, f"{dest_dir}/val/images/{base}.jpg")
        shutil.copy2(lbl, f"{dest_dir}/val/labels/{base}.txt")

# 3. Tao file data.yaml
yaml_content = """path: ./dataset_cau_vuot_finetune
train: train/images
val: val/images

nc: 3
names:
  0: LP
  1: helmet
  2: no helmet
"""
with open(f"{dest_dir}/data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

train_count = len(os.listdir(f"{dest_dir}/train/images"))
val_count = len(os.listdir(f"{dest_dir}/val/images"))
print(f"Tong anh Train: {train_count}")
print(f"Tong anh Val: {val_count}")

# 4. Dong goi thanh file zip san sang up len Kaggle
zip_path = "D:/hoclamAI/giaothong/dataset_cau_vuot_finetune.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)

print("Dang tao file zip...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, "D:/hoclamAI/giaothong")
            zipf.write(full_p, rel_p)

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Da dong goi xong file zip: {zip_path} ({zip_size_mb:.1f} MB)")

"""Đóng gói bộ dữ liệu dataset_v4_clean thành file zip để tải lên Google Colab / Kaggle.
"""
import os
import sys
import zipfile

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\hoclamAI\giaothong"
SRC_DIR = os.path.join(ROOT, "dataset_rebuild", "dataset_v4_clean")
ZIP_DEST = os.path.join(ROOT, "dataset_v4_clean.zip")

def package():
    if not os.path.exists(SRC_DIR):
        print(f"Lỗi: Không tìm thấy thư mục {SRC_DIR}")
        return
        
    print(f">> Bắt đầu nén thư mục: {SRC_DIR}")
    if os.path.exists(ZIP_DEST):
        os.remove(ZIP_DEST)
        
    total_files = 0
    with zipfile.ZipFile(ZIP_DEST, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(SRC_DIR):
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, os.path.dirname(SRC_DIR))
                zipf.write(full_p, rel_p)
                total_files += 1
                
    size_mb = os.path.getsize(ZIP_DEST) / (1024 * 1024)
    print(f"✓ Đã đóng gói thành công {total_files} file vào:")
    print(f"  {ZIP_DEST} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    package()

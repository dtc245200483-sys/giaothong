@echo off
chcp 65001 >nul
title HE THONG GIAM SAT GIAO THONG (YOLO11s UNIFIED)
color 0A
echo ======================================================================
echo    HỆ THỐNG GIÁM SÁT GIAO THÔNG THÔNG MINH (YOLO11s UNIFIED)
echo    - Phát hiện: Xe máy, Mũ bảo hiểm, Không đội mũ, Biển số xe
echo    - Tự động ghi nhận xe vi phạm và nhận diện biển số thời gian thực
echo    - Vận hành tốc độ cao trên CPU (OpenVINO FP16 + RapidOCR ONNX)
echo ======================================================================
echo.
echo [Phím tắt điều khiển]:
echo   - Phím Space (Cách): Tạm dừng / Tiếp tục xem
echo   - Phím N / P       : Chuyển video tiếp theo / Quay lại video trước
echo   - Phím Q hoặc Esc  : Thoát chương trình
echo.
echo Đang khởi động giao diện...
cd /d "D:\hoclamAI\giaothong"
call .venv\Scripts\activate.bat
python scripts\demo_unified_viewer.py
if %errorlevel% neq 0 (
    echo.
    echo Co loi xay ra khi chay chuong trinh!
    pause
)

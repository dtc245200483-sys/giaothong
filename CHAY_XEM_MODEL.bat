@echo off
chcp 65001 >nul
title HE THONG GIAM SAT GIAO THONG (YOLO11s UNIFIED)
color 0A
echo ======================================================================
echo    HE THONG GIAM SAT GIAO THONG THONG MINH (YOLO11s UNIFIED)
echo    - Phat hien Xe may, Mu bao hiem, Khong doi mu, Bien so xe
echo    - Toc do cao, muot ma tren CPU (OpenVINO FP16)
echo    - Nhan dien bien so xe voi RapidOCR ONNX
echo ======================================================================
echo.
echo [Phim tat tien loi]:
echo   - Phim Space (Cach): Tam dung / Tiep tuc xem
echo   - Phim N / P: Chuyen video tiep theo / Quay lai video truoc
echo   - Phim Q hoac Esc: Thoat
echo.
echo Dang khoi dong giao dien...
cd /d "D:\hoclamAI\giaothong"
call .venv\Scripts\activate.bat
python scripts\demo_unified_viewer.py
if %errorlevel% neq 0 (
    echo.
    echo Co loi xay ra khi chay chuong trinh!
    pause
)

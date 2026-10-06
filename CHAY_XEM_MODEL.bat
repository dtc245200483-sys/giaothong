@echo off
title He Thong Giam Sat Giao Thong 2 Tang (Chia Doi 2 Ben)
color 0A
echo ======================================================================
echo    HE THONG GIAM SAT GIAO THONG (GIAO DIEN CHIA DOI 2 BEN)
echo    - Ben Trai: Video quay truc tiep phat hien loi (Xe may, Mu, Bien so)
echo    - Ben Phai: Hien ro anh xe vi pham + Phong to sieu net bien so xe
echo    - Models: Nap bo doi YOLO11s (2 Tang) vua huan luyen tu Kaggle!
echo    - Videos: Tu dong chay 8 video HD moi nhat cua ban!
echo ======================================================================
echo.
echo [Phim tat tien loi]:
echo   - Phim N: Chuyen sang video moi tiep theo
echo   - Phim P: Quay lai video truoc do
echo   - Phim Space (Cach): Tam dung / Tiep tuc xem
echo   - Phim C: CHUP LUU FRAME LOI (Tu dong luu anh de ban mo web gan nhan ngay!)
echo   - Phim S: Luu bien ban vi pham (kem anh xe va bien so zoom)
echo   - Phim Q hoac Esc: Thoat
echo.
echo Dang khoi dong giao dien...
cd /d "D:\hoclamAI\giaothong"
".venv\Scripts\python.exe" "scripts/demo_two_stage_viewer.py"
if %errorlevel% neq 0 (
    echo.
    echo Co loi xay ra khi chay chuong trinh!
    pause
)

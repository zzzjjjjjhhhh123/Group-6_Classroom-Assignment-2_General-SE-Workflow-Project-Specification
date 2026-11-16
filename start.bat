@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   SE Workflow Prototype
echo   Open:  http://127.0.0.1:8000
echo   Keep this window open while demoing.
echo ============================================
start "" /B cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"
python app.py --port 8000
echo.
echo Server stopped. Press any key to close.
pause >nul

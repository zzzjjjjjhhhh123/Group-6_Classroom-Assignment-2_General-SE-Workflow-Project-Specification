@echo off
chcp 65001 >nul
cd /d %~dp0
rem Demo mode: idea -> spec still uses the model (about 10s), but task
rem decomposition uses the deterministic rule engine (<0.05s, always 39/43/64
rem tasks) so the live demo never waits 20s or fails on a bad network.
set SE_WORKFLOW_RULE_DECOMPOSE=1
echo ============================================
echo   SE Workflow  -  DEMO MODE
echo   Idea to spec : LLM  (about 10s)
echo   Task decomp. : rule engine (instant)
echo   Open:  http://127.0.0.1:8000
echo   Keep this window open while demoing.
echo ============================================
start "" /B cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"
python app.py --port 8000
echo.
echo Server stopped. Press any key to close.
pause >nul

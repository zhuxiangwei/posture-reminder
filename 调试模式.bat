@echo off
rem ============================================================
rem  Posture Guard - launcher (debug / troubleshooting)
rem
rem  Same as the normal launcher, but keeps the console visible
rem  so you can read whatever it prints if the app fails to open.
rem  Content is ASCII only on purpose -- see the normal launcher.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0repos\p0-prototype"
"%~dp0envs\posture\Scripts\python.exe" "posture_app.py"
echo.
echo ---- exited with code %ERRORLEVEL% ----
pause

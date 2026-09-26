@echo off
rem ============================================================
rem  Posture Guard - launcher (demo mode, no camera)
rem
rem  Cycles through the good / warning / cannot-see-you states so
rem  you can see what the interface looks like without seating a
rem  child in front of the camera. Content is ASCII only.
rem ============================================================
cd /d "%~dp0repos\p0-prototype"
start "" "%~dp0envs\posture\Scripts\pythonw.exe" "posture_app.py" --demo

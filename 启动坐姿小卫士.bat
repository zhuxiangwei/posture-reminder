@echo off
rem ============================================================
rem  Posture Guard - launcher (normal use)
rem
rem  Starts with pythonw.exe so NO console window appears --
rem  the child just sees a normal app window.
rem  Logs go to the in-app "Parent settings > Run log" tab.
rem
rem  NOTE: this file deliberately contains ASCII only.
rem  cmd.exe reads .bat as ANSI (GBK on Chinese Windows), so
rem  UTF-8 Chinese comments here would turn into mojibake, and a
rem  UTF-8 BOM would break the very first line. Rename the FILE
rem  freely -- only the contents had to stay ASCII.
rem ============================================================
cd /d "%~dp0repos\p0-prototype"
start "" "%~dp0envs\posture\Scripts\pythonw.exe" "posture_app.py"

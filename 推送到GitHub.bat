@echo off
rem ================================================================
rem  Push local repo to GitHub
rem  (ASCII-only content on purpose: cmd.exe reads .bat as GBK, and a
rem   BOM or non-ASCII byte here would break the first line.)
rem ================================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set PY=envs\posture\Scripts\python.exe

if not exist "%PY%" (
  echo.
  echo [ERROR] Python venv not found:
  echo         %PY%
  echo.
  echo         Follow the setup steps in README.md first
  echo         ^(create envs\posture and install the packages^).
  echo.
  pause
  exit /b 1
)

"%PY%" tools\push_to_github.py %*

echo.
pause

@echo off
title BoneGuard AI - FastAPI Backend
echo ========================================================
echo   Starting BoneGuard AI FastAPI Backend Server...
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause

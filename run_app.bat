@echo off
title BoneGuard AI Launcher
echo ========================================================
echo   Launching BoneGuard AI (Backend + Frontend)
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "BoneGuard AI - Backend" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting React Frontend on http://localhost:5173 ...
start "BoneGuard AI - Frontend" cmd /k "cd frontend && npm.cmd run dev"

echo.
echo ========================================================
echo   Both services started!
echo   Frontend Dashboard : http://localhost:5173
echo   API Swagger Docs   : http://127.0.0.1:8000/docs
echo ========================================================
echo.
echo Opening BoneGuard AI in your browser in 3 seconds...
timeout /t 3 >nul
start http://localhost:5173

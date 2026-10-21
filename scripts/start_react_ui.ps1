# Space RAG - Launch Mission Control HUD (React + FastAPI)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   🛰️ SPACE RAG — MISSION CONTROL HUD (REACT + FASTAPI)   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$Root = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $Root "renv\Scripts\python.exe"

# 1. Start FastAPI backend server
Write-Host "[1/2] Launching Space RAG FastAPI Backend on port 8000..." -ForegroundColor Yellow
$BackendProcess = Start-Process -FilePath $PythonPath -ArgumentList "-m uvicorn app.server:app --host 127.0.0.1 --port 8000" -WorkingDirectory $Root -PassThru

Start-Sleep -Seconds 2

# 2. Start React Vite Dev Server
Write-Host "[2/2] Launching React Vite Dev Server on port 5173..." -ForegroundColor Green
Set-Location (Join-Path $Root "frontend")
npm run dev

# Space RAG - Unified Production Server (serves React HUD + API)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   🛰️ SPACE RAG — UNIFIED MISSION CONTROL SERVER          " -ForegroundColor Cyan
Write-Host "   Serving at: http://127.0.0.1:8000                     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$Root = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $Root "renv\Scripts\python.exe"

& $PythonPath -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload

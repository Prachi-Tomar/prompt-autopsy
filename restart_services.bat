@echo off
echo Stopping Streamlit and Uvicorn processes...

REM Kill any existing streamlit processes
taskkill /f /im streamlit.exe 2>nul
echo Streamlit processes terminated.

REM Kill any existing uvicorn processes
taskkill /f /im uvicorn.exe 2>nul
echo Uvicorn processes terminated.

echo.
echo Starting backend service...
start "Backend" cmd /k "uvicorn backend.app:app --reload"

echo Starting frontend service...
start "Frontend" cmd /k "streamlit run frontend/streamlit_app.py --server.headless=true"

echo Services restarted successfully!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
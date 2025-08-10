#!/bin/bash

echo "Stopping Streamlit and Uvicorn processes..."

# Kill any existing streamlit processes
pkill -f "streamlit run" 2>/dev/null
echo "Streamlit processes terminated."

# Kill any existing uvicorn processes
pkill -f "uvicorn backend.app:app" 2>/dev/null
echo "Uvicorn processes terminated."

echo ""
echo "Starting backend service..."
uvicorn backend.app:app --reload &

echo "Starting frontend service..."
streamlit run frontend/streamlit_app.py --server.headless=true &

echo "Services restarted successfully!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:8501"
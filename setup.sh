#!/usr/bin/env bash
# Adala Auto Annotator - Setup Script

set -e

echo "==================================="
echo "  Adala Auto Annotator Setup"
echo "==================================="

# Create virtualenv if not exists
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

echo "→ Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "→ Creating data directories..."
mkdir -p data/uploads data/models data/annotations data/exports

echo ""
echo "==================================="
echo "  Setup complete!"
echo "==================================="
echo ""
echo "To start the system:"
echo ""
echo "  1. Backend (Terminal 1):"
echo "     source venv/bin/activate"
echo "     uvicorn main:app --reload --port 8000"
echo ""
echo "  2. Frontend (Terminal 2):"
echo "     source venv/bin/activate"
echo "     streamlit run frontend/Home.py"
echo ""
echo "  API docs: http://localhost:8000/docs"
echo "  UI:       http://localhost:8501"
echo "==================================="

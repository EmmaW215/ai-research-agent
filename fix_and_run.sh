#!/bin/bash

# Fix and Run Script for AI Research Agent
# This script ensures all dependencies are installed and runs the application

set -e

echo "=========================================="
echo "AI Research Agent - Fix and Run"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./install_with_python312.sh first"
    exit 1
fi

# Deactivate any existing virtual environment first
if [ -n "$VIRTUAL_ENV" ]; then
    echo "   Deactivating existing virtual environment..."
    deactivate 2>/dev/null || true
fi

# Activate virtual environment
echo "1. Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Verify we're using the correct Python (should be in ai-research-agent/venv)
EXPECTED_PYTHON="$SCRIPT_DIR/venv/bin/python"
ACTUAL_PYTHON=$(which python)
if [[ "$ACTUAL_PYTHON" != "$EXPECTED_PYTHON"* ]]; then
    echo "   ⚠️  Warning: Using unexpected Python interpreter"
    echo "   Expected: $EXPECTED_PYTHON"
    echo "   Actual: $ACTUAL_PYTHON"
    echo "   Attempting to use correct Python..."
    # Force use of local venv Python
    export PATH="$SCRIPT_DIR/venv/bin:$PATH"
    ACTUAL_PYTHON=$(which python)
fi

echo "   ✅ Virtual environment activated"
echo "   Python: $ACTUAL_PYTHON"
echo "   Python version: $(python --version)"
echo "   Virtual env: $VIRTUAL_ENV"
echo ""

# Install/update all dependencies
echo "2. Installing/updating dependencies..."
echo "   This ensures all required packages are installed..."
pip install -r requirements.txt --quiet
echo "   ✅ Dependencies installed"
echo ""

# Verify critical imports
echo "3. Verifying imports..."
python -c "from pydantic_settings import BaseSettings; print('   ✅ pydantic_settings')" || exit 1
python -c "from src.infrastructure.config import get_settings; print('   ✅ config module')" || exit 1
python -c "from src.api.main import app; print('   ✅ FastAPI app')" || exit 1
echo ""

# Check if database exists
if [ ! -f "research_agent.db" ]; then
    echo "4. Database not found. Setting up database..."
    python scripts/setup_db.py
    echo ""
else
    echo "4. Database already exists (skipping setup)"
    echo ""
fi

# Check if port 8000 is already in use
echo "5. Checking port 8000..."
PORT_PID=$(lsof -ti:8000 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    echo "   ⚠️  Port 8000 is already in use (PID: $PORT_PID)"
    echo "   Stopping existing process..."
    kill -9 $PORT_PID 2>/dev/null
    sleep 1
    # Verify port is free
    if lsof -ti:8000 >/dev/null 2>&1; then
        echo "   ❌ Failed to free port 8000. Please manually stop the process:"
        echo "      lsof -ti:8000 | xargs kill -9"
        exit 1
    else
        echo "   ✅ Port 8000 is now free"
    fi
else
    echo "   ✅ Port 8000 is available"
fi
echo ""

# Start the server
echo "=========================================="
echo "Starting AI Research Agent..."
echo "=========================================="
echo ""
echo "📍 API will be available at:"
echo "   - Main API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Health: http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Use python -m uvicorn to ensure correct Python environment
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000


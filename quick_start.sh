#!/bin/bash

# AI Research Agent - Quick Start Script
# Note: For first-time setup, use install_with_python312.sh instead

echo "=========================================="
echo "AI Research Agent - Quick Start"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo "1. Checking Python version..."
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    PYTHON_VERSION=$(python3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "   Using Python 3.12: $PYTHON_VERSION"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "   Python version: $PYTHON_VERSION"
    
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MINOR" -eq 14 ]; then
        echo "   ⚠️  Python 3.14 detected. Some packages may not work."
        echo "   💡 For best results, use: ./install_with_python312.sh"
    elif [ "$MINOR" -lt 11 ]; then
        echo "   ⚠️  Warning: Python 3.11+ recommended"
        echo "   💡 For best results, use: ./install_with_python312.sh"
    fi
else
    echo "   ❌ Python not found!"
    exit 1
fi
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "2. Creating virtual environment..."
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "2. Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"
echo ""

# Install/upgrade dependencies
echo "4. Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo "   ✅ Dependencies installed"
echo ""

# Check if database needs setup
if [ ! -f "research_agent.db" ]; then
    echo "5. Setting up database..."
    python scripts/setup_db.py
    echo ""
else
    echo "5. Database already exists (skipping setup)"
    echo ""
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "6. Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "   ✅ .env file created from .env.example"
        echo "   ⚠️  Please edit .env file if needed"
    else
        echo "   ℹ️  No .env.example found, using default settings"
    fi
    echo ""
else
    echo "6. .env file already exists"
    echo ""
fi

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

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000


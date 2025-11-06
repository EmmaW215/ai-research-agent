#!/bin/bash

# AI Research Agent - Installation Script with Python 3.12
# This script ensures the project works properly with Python 3.12

set -e  # Exit on error

echo "=========================================="
echo "AI Research Agent - Python 3.12 Installer"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check for Python 3.12
echo "Step 1: Checking for Python 3.12..."
PYTHON312=""
if command -v python3.12 &> /dev/null; then
    PYTHON312="python3.12"
    PYTHON_VERSION=$(python3.12 --version 2>&1)
    echo -e "${GREEN}   ✅ Found: $PYTHON_VERSION${NC}"
elif command -v python3.12.x &> /dev/null; then
    PYTHON312="python3.12.x"
    echo -e "${GREEN}   ✅ Found Python 3.12${NC}"
else
    echo -e "${YELLOW}   ⚠️  Python 3.12 not found in PATH${NC}"
    echo ""
    echo "Please install Python 3.12:"
    echo "  macOS (Homebrew):"
    echo "    brew install python@3.12"
    echo ""
    echo "  Or download from: https://www.python.org/downloads/"
    echo ""
    read -p "Do you want to continue with system Python? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    # Try to find any Python 3.12
    for py in python3.12 python3.12.0 python3.12.1 python3.12.2 python3.12.3 python3.12.4 python3.12.5; do
        if command -v $py &> /dev/null; then
            PYTHON312=$py
            echo -e "${GREEN}   ✅ Found: $py${NC}"
            break
        fi
    done
    
    if [ -z "$PYTHON312" ]; then
        echo -e "${RED}   ❌ Python 3.12 not found. Please install it first.${NC}"
        exit 1
    fi
fi
echo ""

# Step 2: Verify Python version
echo "Step 2: Verifying Python version..."
PYTHON_VER=$($PYTHON312 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
MINOR=$(echo $PYTHON_VER | cut -d. -f2)

if [ "$MAJOR" -ne 3 ] || [ "$MINOR" -lt 11 ]; then
    echo -e "${RED}   ❌ Python 3.11+ required, found Python $PYTHON_VER${NC}"
    exit 1
elif [ "$MINOR" -eq 14 ]; then
    echo -e "${YELLOW}   ⚠️  Python 3.14 detected. Some packages may not work.${NC}"
    echo -e "${YELLOW}   Consider using Python 3.12 for best compatibility.${NC}"
elif [ "$MINOR" -ge 12 ]; then
    echo -e "${GREEN}   ✅ Python $PYTHON_VER is compatible${NC}"
else
    echo -e "${YELLOW}   ⚠️  Python $PYTHON_VER should work, but 3.12+ is recommended${NC}"
fi
echo ""

# Step 3: Remove old virtual environment if exists
echo "Step 3: Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "   Removing old virtual environment..."
    rm -rf venv
fi

# Create new virtual environment with specified Python
echo "   Creating virtual environment with $PYTHON312..."
$PYTHON312 -m venv venv

if [ ! -d "venv" ]; then
    echo -e "${RED}   ❌ Failed to create virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}   ✅ Virtual environment created${NC}"
echo ""

# Step 4: Activate virtual environment
echo "Step 4: Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}   ❌ Failed to activate virtual environment${NC}"
    exit 1
fi

ACTUAL_PYTHON=$(python --version 2>&1)
echo -e "${GREEN}   ✅ Virtual environment activated${NC}"
echo "   Using: $ACTUAL_PYTHON"
echo ""

# Step 5: Upgrade pip
echo "Step 5: Upgrading pip..."
pip install --upgrade pip --quiet
PIP_VERSION=$(pip --version)
echo -e "${GREEN}   ✅ pip upgraded: $PIP_VERSION${NC}"
echo ""

# Step 6: Install dependencies
echo "Step 6: Installing dependencies..."
echo "   This may take a few minutes..."

# First, try to install without chromadb (which may fail on Python 3.14)
if [ "$MINOR" -ge 14 ]; then
    echo "   Installing core dependencies (skipping chromadb for Python 3.14+)..."
    pip install fastapi==0.104.1 'uvicorn[standard]==0.24.0' pydantic==2.5.0 pydantic-settings==2.1.0 sqlalchemy==2.0.23 alembic==1.13.0 'python-jose[cryptography]==3.3.0' 'passlib[bcrypt]==1.7.4' python-multipart==0.0.6 pytest==7.4.3 pytest-asyncio==0.21.1 pytest-cov==4.1.0 httpx==0.25.2 python-dotenv==1.0.0
    
    echo ""
    echo -e "${YELLOW}   ⚠️  ChromaDB skipped (not compatible with Python 3.14)${NC}"
    echo -e "${YELLOW}   Vector store features will not be available${NC}"
else
    # For Python 3.11-3.13, install all dependencies including chromadb
    if pip install -r requirements.txt; then
        echo -e "${GREEN}   ✅ All dependencies installed successfully${NC}"
    else
        echo -e "${YELLOW}   ⚠️  Some dependencies failed to install${NC}"
        echo "   Installing core dependencies..."
        pip install fastapi==0.104.1 'uvicorn[standard]==0.24.0' pydantic==2.5.0 pydantic-settings==2.1.0 sqlalchemy==2.0.23 alembic==1.13.0 'python-jose[cryptography]==3.3.0' 'passlib[bcrypt]==1.7.4' python-multipart==0.0.6 pytest==7.4.3 pytest-asyncio==0.21.1 pytest-cov==4.1.0 httpx==0.25.2 python-dotenv==1.0.0
        
        # Try to install chromadb separately
        echo "   Attempting to install chromadb..."
        if pip install 'chromadb>=0.4.22' 2>/dev/null; then
            echo -e "${GREEN}   ✅ ChromaDB installed${NC}"
        else
            echo -e "${YELLOW}   ⚠️  ChromaDB installation failed (optional)${NC}"
        fi
    fi
fi
echo ""

# Step 7: Verify installation
echo "Step 7: Verifying installation..."
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')" 2>/dev/null || echo -e "${RED}   ❌ FastAPI not installed${NC}"
python -c "import sqlalchemy; print(f'✅ SQLAlchemy {sqlalchemy.__version__}')" 2>/dev/null || echo -e "${RED}   ❌ SQLAlchemy not installed${NC}"
python -c "import pydantic; print(f'✅ Pydantic {pydantic.__version__}')" 2>/dev/null || echo -e "${RED}   ❌ Pydantic not installed${NC}"

# Check chromadb if Python < 3.14
if [ "$MINOR" -lt 14 ]; then
    python -c "import chromadb; print('✅ ChromaDB installed')" 2>/dev/null || echo -e "${YELLOW}   ⚠️  ChromaDB not installed (optional)${NC}"
fi
echo ""

# Step 8: Setup database
echo "Step 8: Setting up database..."
if [ -f "research_agent.db" ]; then
    echo "   Database file already exists (skipping setup)"
else
    if python scripts/setup_db.py; then
        echo -e "${GREEN}   ✅ Database initialized${NC}"
        echo "   Default admin credentials:"
        echo "     Email: admin@example.com"
        echo "     Password: admin123"
        echo -e "${YELLOW}     ⚠️  Please change password after first login!${NC}"
    else
        echo -e "${YELLOW}   ⚠️  Database setup failed (you can run it manually later)${NC}"
        echo "   Run: python scripts/setup_db.py"
    fi
fi
echo ""

# Step 9: Create .env file if it doesn't exist
echo "Step 9: Checking environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}   ✅ Created .env file from .env.example${NC}"
        echo -e "${YELLOW}   ⚠️  Please review and update .env file if needed${NC}"
    else
        echo "   No .env.example found, using default settings"
    fi
else
    echo "   .env file already exists"
fi
echo ""

# Final summary
echo "=========================================="
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "To start the application:"
echo ""
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start the server:"
echo "     uvicorn src.api.main:app --reload"
echo ""
echo "  3. Access the API:"
echo "     - API Docs: http://localhost:8000/docs"
echo "     - Health: http://localhost:8000/api/v1/health"
echo ""
echo "Or use the quick start script:"
echo "    ./quick_start.sh"
echo ""


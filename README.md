# AI Research Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered research assistant featuring voice interaction, domain-scoped retrieval, and citation-grounded answers.

## 🎯 Project Vision

AI Research Agent is a comprehensive end-to-end application that helps researchers efficiently explore and understand academic literature through:

- **Voice Interaction**: Natural conversation interface for hands-free research
- **Domain-Scoped Retrieval**: Optimized search within specific academic domains
- **Citation-Grounded Answers**: All responses backed by verifiable sources
- **Multi-Modal Input**: Support for PDFs, web content, and voice queries

## 🏗️ Architecture

The system follows clean architecture principles with ports/adapters pattern for maximum modularity and reusability:

```
ai-research-agent/
├── src/
│   ├── core/           # Domain logic and entities
│   ├── adapters/       # External integrations
│   ├── api/            # API endpoints
│   └── infrastructure/ # Cross-cutting concerns
├── tests/              # Comprehensive test suite
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## 📦 Batch 1 - Foundation Layer (Current)

**Status**: ✅ Complete

### Modules Implemented:

- **A0 - Authentication & Authorization**: JWT-based auth with role-based access control
- **B1 - API Gateway**: FastAPI-based RESTful API with request validation
- **S1 - Vector Store Setup**: Chroma database for semantic search
- **T0 - Testing Framework**: pytest with fixtures and integration tests
- **O0 - Basic Observability**: Structured JSON logging with correlation IDs

### Tech Stack:

- **Backend**: FastAPI 0.104+
- **Database**: PostgreSQL (users/metadata) + Chroma (vector store)
- **Authentication**: JWT with passlib bcrypt
- **Testing**: pytest with pytest-asyncio
- **Logging**: Structured JSON logs

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 14+ (for production)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/EmmaW215/ai-research-agent.git
cd ai-research-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Run database setup
python scripts/setup_db.py

# Verify installation
python scripts/verify_setup.py
```

### Running the Application

```bash
# Development server with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API will be available at: http://localhost:8000
Interactive docs at: http://localhost:8000/docs

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_auth.py -v
```

## 📖 API Documentation

### Authentication Endpoints

```bash
# Register new user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'

# Get current user
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 👤 Author

**Emma Wang**
- GitHub: [@EmmaW215](https://github.com/EmmaW215)
- Focus: RF Engineering → AI/ML Transition

---

**Current Status**: Batch 1 Complete ✅ | Next: Batch 2 - Content Processing

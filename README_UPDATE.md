# Update for README.md

Add this section after the Batch 1 section:

## 📦 Batch 2 - Content Processing Pipeline (Current)

**Status**: ✅ Complete

This release implements the document ingestion and processing pipeline:

### Modules Implemented:

- **D1 - Document Ingestion**: Upload and manage research papers
- **E1 - Text Extraction**: Extract clean text from PDFs
- **C1 - Chunking Engine**: Recursive and semantic chunking
- **V1 - Embedding Generation**: Vector embeddings for semantic search
- **P1 - Pipeline Orchestration**: End-to-end automated processing

### New Features:

- PDF upload API with progress tracking
- Intelligent text chunking (2 strategies)
- Local embedding generation
- Vector storage in ChromaDB
- Background processing pipeline
- Comprehensive status tracking

### API Endpoints:

```bash
# Upload PDF
POST /api/v1/documents/upload

# List documents
GET /api/v1/documents

# Get document details
GET /api/v1/documents/{id}

# Check processing status
GET /api/v1/documents/{id}/status

# Get document chunks
GET /api/v1/documents/{id}/chunks

# Delete document
DELETE /api/v1/documents/{id}
```

### Quick Start for Batch 2:

```bash
# Install new dependencies
pip install -r requirements.txt

# Run database migrations
python scripts/run_migrations.py

# Start the application
uvicorn src.api.main:app --reload

# Upload a document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@paper.pdf"
```

---

**Current Status**: Batch 2 Complete ✅ | Next: Batch 3 - Search & Retrieval

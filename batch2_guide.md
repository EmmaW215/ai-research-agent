# Batch 2 - Complete Implementation Guide

## 📦 What You're Getting

I've created **~25 new files** for Batch 2 that implement the complete Content Processing Pipeline:

### File Breakdown:

**Configuration (2 files)**:
- ✅ Updated `requirements.txt` - 10 new dependencies
- ✅ Updated `.env.example` - Batch 2 configuration

**Core Models (2 files)**:
- ✅ `src/core/document_models.py` - Document & Chunk entities
- ✅ `src/core/document_schemas.py` - Pydantic schemas

**Adapters (4 files)**:
- ✅ `src/adapters/storage.py` - File storage
- ✅ `src/adapters/pdf_extractor.py` - PDF text extraction
- ✅ `src/adapters/embeddings.py` - Embedding generation
- ✅ `src/adapters/chunking_strategies.py` - Chunking algorithms

**Services (4 files)**:
- ✅ `src/services/__init__.py` - Service layer init
- ✅ `src/services/document_service.py` - Document management
- ✅ `src/services/chunking_service.py` - Chunking & embedding
- ✅ `src/services/ingestion_service.py` - Full pipeline

**API Routes (2 files)**:
- ✅ `src/api/routes/documents.py` - Document endpoints
- ✅ `src/api/main_update.py` - Instructions for main.py

**Tests (5 files)**:
- ✅ `tests/test_pdf_extraction.py` - PDF tests
- ✅ `tests/test_chunking.py` - Chunking tests
- ✅ `tests/test_document_api.py` - API tests
- ✅ `tests/test_embeddings.py` - Embedding tests
- ✅ `tests/test_ingestion_pipeline.py` - Pipeline tests

**Scripts (2 files)**:
- ✅ `scripts/run_migrations.py` - Database setup
- ✅ `scripts/process_sample_documents.py` - Sample processor

**Documentation (2 files)**:
- ✅ `docs/BATCH2_SUMMARY.md` - Complete summary
- ✅ `README_UPDATE.md` - README additions

---

## 🚀 How to Use the Script

### Step 1: Generate Files

```bash
# In your ai-research-agent directory
python create_batch2_files.py
```

This creates all 25 files in the correct structure.

### Step 2: Manual Updates Required

**Update `src/api/main.py`** (add these 2 lines):

```python
# After existing imports
from src.api.routes import documents

# After existing router includes (around line 85)
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
```

**Update `src/infrastructure/config.py`** (add these settings to Settings class):

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # File Upload Settings
    max_upload_size: int = 52428800  # 50MB
    allowed_file_types: str = "application/pdf"
    upload_directory: str = "./uploads"
    
    # PDF Processing
    pdf_extract_images: bool = False
    pdf_extract_tables: bool = True
    
    # Chunking Settings
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    chunking_strategy: str = "recursive"
    
    # Embedding Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
```

### Step 3: Install Dependencies

```bash
# Install all new packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Setup Database

```bash
# Run migrations to create new tables
python scripts/run_migrations.py
```

This creates:
- `documents` table
- `document_chunks` table

### Step 5: Update .env

```bash
# Copy new settings from .env.example
# Or manually add the Batch 2 settings to your existing .env
```

### Step 6: Test Everything

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=html

# Test specific modules
pytest tests/test_chunking.py -v
pytest tests/test_embeddings.py -v
```

### Step 7: Start the Application

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📝 What Each Module Does

### Document Ingestion (D1)
**Purpose**: Handle document uploads and storage

**Key Features**:
- Async file upload
- Status tracking (7 stages)
- File validation
- User isolation

**Example Usage**:
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@paper.pdf"
```

### PDF Extraction (E1)
**Purpose**: Extract text from PDF files

**Key Features**:
- PyMuPDF for fast extraction
- Metadata extraction
- Text cleaning
- Table support (pdfplumber)

**Handles**:
- Multi-column layouts
- Academic papers
- Complex formatting

### Chunking Engine (C1)
**Purpose**: Split text into meaningful segments

**Strategies**:
1. **Recursive** (default) - Smart boundary detection
2. **Semantic** - Embedding-based similarity

**Configuration**:
- Chunk size: 512 tokens
- Overlap: 50 tokens
- Customizable separators

### Embedding Generation (V1)
**Purpose**: Convert text to vectors

**Model**: sentence-transformers/all-MiniLM-L6-v2
- Fast and efficient
- 384 dimensions
- Runs on CPU (CUDA optional)

**Features**:
- Batch processing
- Similarity computation
- Easy model swapping

### Pipeline Orchestration (P1)
**Purpose**: Coordinate end-to-end processing

**Stages**:
1. Validation → Check file
2. Extraction → Get text
3. Chunking → Segment text
4. Embedding → Generate vectors
5. Storage → Save to ChromaDB
6. Finalization → Update status

**Features**:
- Background processing
- Error recovery
- Progress tracking

---

## 🎯 Testing Your Implementation

### Test 1: Upload a Document

```bash
# 1. Login to get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "admin123"}'

# 2. Upload PDF (use token from step 1)
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@sample.pdf"

# Response will include document_id
```

### Test 2: Check Processing Status

```bash
curl -X GET http://localhost:8000/api/v1/documents/{document_id}/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

You'll see status progress through:
- uploaded → validating → extracting → chunking → embedding → ready

### Test 3: List Your Documents

```bash
curl -X GET http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Test 4: Get Document Chunks

```bash
curl -X GET http://localhost:8000/api/v1/documents/{document_id}/chunks \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 📊 Performance Expectations

Based on a typical research paper (10-20 pages, 5MB):

- **Upload**: < 1 second
- **Validation**: < 0.5 seconds
- **Text Extraction**: 3-5 seconds
- **Chunking**: 1-2 seconds
- **Embedding**: 5-10 seconds (CPU)
- **Vector Storage**: < 1 second
- **Total**: 20-30 seconds

For GPU (if CUDA available):
- Embedding drops to 2-3 seconds
- Total: 10-15 seconds

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Issue: Database errors

**Solution**:
```bash
# Re-run migrations
python scripts/run_migrations.py

# Or drop and recreate (dev only!)
rm research_agent.db
python scripts/setup_db.py
python scripts/run_migrations.py
```

### Issue: Upload fails with 413

**Solution**: File too large. Increase `MAX_UPLOAD_SIZE` in .env

### Issue: Slow embedding generation

**Solution**: 
- Use smaller batch size: `EMBEDDING_BATCH_SIZE=16`
- Or enable GPU: `EMBEDDING_DEVICE=cuda`

### Issue: Out of memory

**Solution**:
- Reduce chunk size: `DEFAULT_CHUNK_SIZE=256`
- Use smaller embedding model
- Process fewer documents simultaneously

---

## 🔍 Code Quality

All code includes:
- ✅ Type hints
- ✅ Docstrings
- ✅ Error handling
- ✅ Logging
- ✅ Async support where beneficial
- ✅ Dependency injection
- ✅ Clean architecture patterns

---

## 📈 Next Steps After Batch 2

Once you've tested Batch 2 and everything works:

### Batch 3 Preview: Search & Retrieval
- **Semantic search** with ChromaDB
- **Query understanding** and expansion
- **Hybrid search** (keyword + semantic)
- **Re-ranking** algorithms
- **Citation extraction** from chunks
- **Search API** endpoints

**Estimated Timeline**: 2-3 weeks

---

## ✅ Checklist Before Committing

- [ ] All files created successfully
- [ ] `src/api/main.py` updated with documents router
- [ ] `src/infrastructure/config.py` updated with Batch 2 settings
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] spaCy model downloaded
- [ ] Database migrations run successfully
- [ ] `.env` file updated
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Application starts without errors
- [ ] Can upload a test PDF
- [ ] Document processes successfully
- [ ] Can retrieve chunks
- [ ] README.md updated with Batch 2 info

---

## 🎓 Architecture Highlights

### Clean Architecture
```
Presentation → API Routes (FastAPI)
Application → Services (Business Logic)
Domain → Models & Schemas
Infrastructure → Adapters (DB, Files, ML)
```

### Design Patterns
- **Dependency Injection**: Services receive dependencies
- **Factory Pattern**: `get_chunker()` creates strategies
- **Singleton**: Storage, embeddings, vector store
- **Strategy Pattern**: Multiple chunking algorithms
- **Repository Pattern**: Document service

### Best Practices
- Async file operations
- Background task processing
- Comprehensive error handling
- Status tracking throughout pipeline
- Metadata-rich storage

---

## 💡 Tips for Success

1. **Start Small**: Test with 1-2 PDFs first
2. **Monitor Logs**: Watch the structured JSON logs
3. **Use the Status Endpoint**: Track processing progress
4. **Test Both Strategies**: Try recursive and semantic chunking
5. **Check Vector Store**: Verify embeddings are stored
6. **Profile Performance**: Use timing logs to identify bottlenecks

---

**Ready to build? Run the script and let's get Batch 2 deployed! 🚀**

Questions? I'm here to help with any issues or clarifications.

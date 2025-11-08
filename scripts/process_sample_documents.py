#!/usr/bin/env python
"""Process sample documents for testing"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.database import SessionLocal
from src.services.document_service import DocumentService
from src.services.chunking_service import ChunkingService
from src.services.ingestion_service import IngestionService
from src.adapters.storage import get_storage
from src.adapters.pdf_extractor import get_pdf_extractor
from src.adapters.embeddings import get_embedding_adapter
from src.adapters.vector_store import get_vector_store


async def process_sample():
    """Process sample documents"""
    print("=" * 60)
    print("Processing Sample Documents")
    print("=" * 60)
    print()
    
    # Initialize services
    db = SessionLocal()
    storage = get_storage()
    pdf_extractor = get_pdf_extractor()
    embedding_adapter = get_embedding_adapter()
    vector_store = get_vector_store()
    
    doc_service = DocumentService(db, storage)
    chunking_service = ChunkingService(db, embedding_adapter, vector_store)
    ingestion_service = IngestionService(db, doc_service, pdf_extractor, chunking_service)
    
    # TODO: Add sample document processing logic
    print("Add your sample PDF files to process here")
    print()
    print("=" * 60)
    
    db.close()

if __name__ == "__main__":
    asyncio.run(process_sample())

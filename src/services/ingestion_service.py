"""Document ingestion pipeline service"""

from typing import Optional
from sqlalchemy.orm import Session
from src.core.document_models import Document, DocumentStatus
from src.adapters.pdf_extractor import PDFExtractor
from src.adapters.embeddings import EmbeddingAdapter
from src.adapters.vector_store import VectorStore
from src.services.chunking_service import ChunkingService
from src.services.document_service import DocumentService
from src.infrastructure.config import get_settings
from src.infrastructure.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class IngestionService:
    """End-to-end document ingestion pipeline"""
    
    def __init__(
        self,
        db: Session,
        document_service: DocumentService,
        pdf_extractor: PDFExtractor,
        chunking_service: ChunkingService
    ):
        """Initialize ingestion service"""
        self.db = db
        self.document_service = document_service
        self.pdf_extractor = pdf_extractor
        self.chunking_service = chunking_service
    
    async def process_document(self, document_id: str) -> Document:
        """Process a document through the full pipeline"""
        try:
            # Get document
            document = self.db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document not found: {document_id}")
            
            logger.info(f"Starting pipeline for document {document_id}")
            
            # Stage 1: Validation
            await self._validate_document(document)
            
            # Stage 2: Text Extraction
            text, metadata = await self._extract_text(document)
            
            # Stage 3: Chunking & Embedding
            await self._chunk_and_embed(document, text)
            
            # Stage 4: Finalization
            await self._finalize_document(document, metadata)
            
            logger.info(f"Pipeline completed for document {document_id}")
            return document
            
        except Exception as e:
            logger.error(f"Pipeline failed for document {document_id}: {str(e)}")
            self.document_service.update_document_status(
                document_id,
                DocumentStatus.FAILED,
                error_message=str(e)
            )
            raise
    
    async def _validate_document(self, document: Document):
        """Stage 1: Validate document"""
        logger.info(f"Validating document {document.id}")
        
        self.document_service.update_document_status(
            document.id,
            DocumentStatus.VALIDATING
        )
        
        # Check file exists
        from src.adapters.storage import get_storage
        storage = get_storage()
        
        if not storage.file_exists(document.file_path):
            raise ValueError(f"File not found: {document.file_path}")
        
        # Check file size
        if document.file_size > settings.max_upload_size:
            raise ValueError(f"File too large: {document.file_size} bytes")
        
        # Check mime type
        if document.mime_type not in settings.allowed_file_types.split(','):
            raise ValueError(f"Unsupported file type: {document.mime_type}")
        
        logger.info(f"Document {document.id} validated")
    
    async def _extract_text(self, document: Document) -> tuple:
        """Stage 2: Extract text and metadata"""
        logger.info(f"Extracting text from document {document.id}")
        
        self.document_service.update_document_status(
            document.id,
            DocumentStatus.EXTRACTING
        )
        
        # Extract text
        text = self.pdf_extractor.extract_text(document.file_path)
        
        # Clean text
        text = self.pdf_extractor.clean_text(text)
        
        # Extract metadata
        metadata = self.pdf_extractor.extract_metadata(document.file_path)
        
        logger.info(f"Extracted {len(text)} characters, {metadata.get('num_pages', 0)} pages")
        
        return text, metadata
    
    async def _chunk_and_embed(self, document: Document, text: str):
        """Stage 3: Chunk and embed text"""
        logger.info(f"Chunking and embedding document {document.id}")
        
        self.document_service.update_document_status(
            document.id,
            DocumentStatus.CHUNKING
        )
        
        # Chunk and embed
        strategy = settings.chunking_strategy
        chunks = self.chunking_service.chunk_and_embed(document, text, strategy)
        
        # Update document with chunk count
        document.num_chunks = len(chunks)
        document.chunking_strategy = strategy
        self.db.commit()
        
        logger.info(f"Created {len(chunks)} chunks for document {document.id}")
    
    async def _finalize_document(self, document: Document, metadata: dict):
        """Stage 4: Finalize document processing"""
        logger.info(f"Finalizing document {document.id}")
        
        # Update metadata
        self.document_service.update_document_metadata(document.id, {
            "title": metadata.get("title"),
            "num_pages": metadata.get("num_pages"),
            "extraction_method": "pymupdf"
        })
        
        # Update status to ready
        self.document_service.update_document_status(
            document.id,
            DocumentStatus.READY
        )
        
        logger.info(f"Document {document.id} is ready")

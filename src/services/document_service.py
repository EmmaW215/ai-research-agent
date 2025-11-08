"""Document management service"""

import uuid
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.core.document_models import Document, DocumentStatus
from src.core.document_schemas import DocumentResponse, DocumentListResponse
from src.adapters.storage import StorageAdapter
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Service for document management operations"""
    
    def __init__(self, db: Session, storage: StorageAdapter):
        """Initialize document service"""
        self.db = db
        self.storage = storage
    
    async def create_document(
        self,
        user_id: int,
        filename: str,
        file_content: bytes,
        mime_type: str
    ) -> Document:
        """Create a new document record"""
        try:
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Save file to storage
            file_path = await self.storage.save_file(file_content, filename, user_id)
            file_size = self.storage.get_file_size(file_path)
            
            # Create database record
            document = Document(
                id=document_id,
                user_id=user_id,
                filename=f"{document_id}_{filename}",
                original_filename=filename,
                file_path=file_path,
                file_size=file_size,
                mime_type=mime_type,
                status=DocumentStatus.UPLOADED
            )
            
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            
            logger.info(f"Document created: {document_id}")
            return document
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating document: {str(e)}")
            raise
    
    def get_document(self, document_id: str, user_id: int) -> Optional[Document]:
        """Get document by ID"""
        return self.db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
    
    def list_documents(
        self,
        user_id: int,
        page: int = 1,
        page_size: int = 20,
        status: Optional[DocumentStatus] = None
    ) -> DocumentListResponse:
        """List user's documents with pagination"""
        query = self.db.query(Document).filter(Document.user_id == user_id)
        
        if status:
            query = query.filter(Document.status == status)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        documents = query.order_by(desc(Document.created_at))\
            .offset((page - 1) * page_size)\
            .limit(page_size)\
            .all()
        
        return DocumentListResponse(
            documents=[DocumentResponse.model_validate(doc) for doc in documents],
            total=total,
            page=page,
            page_size=page_size
        )
    
    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> Document:
        """Update document processing status"""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        document.status = status
        
        if error_message:
            document.error_message = error_message
        
        if status == DocumentStatus.READY:
            document.processed_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(document)
        
        logger.info(f"Document {document_id} status updated to {status}")
        return document
    
    def update_document_metadata(
        self,
        document_id: str,
        metadata: dict
    ) -> Document:
        """Update document metadata"""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        # Update metadata fields
        for key, value in metadata.items():
            if hasattr(document, key):
                setattr(document, key, value)
        
        self.db.commit()
        self.db.refresh(document)
        
        return document
    
    def delete_document(self, document_id: str, user_id: int) -> bool:
        """Delete document and associated files"""
        document = self.get_document(document_id, user_id)
        
        if not document:
            return False
        
        try:
            # Delete file from storage
            if self.storage.file_exists(document.file_path):
                self.storage.delete_file(document.file_path)
            
            # Delete from database (cascades to chunks)
            self.db.delete(document)
            self.db.commit()
            
            logger.info(f"Document deleted: {document_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting document: {str(e)}")
            raise

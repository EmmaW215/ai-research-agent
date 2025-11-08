"""Document management endpoints"""

import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from src.core.document_models import DocumentStatus
from src.core.document_schemas import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    ChunkResponse,
    ProcessingStatusResponse
)
from src.services.document_service import DocumentService
from src.services.chunking_service import ChunkingService
from src.services.ingestion_service import IngestionService
from src.adapters.storage import get_storage
from src.adapters.pdf_extractor import get_pdf_extractor
from src.adapters.embeddings import get_embedding_adapter
from src.adapters.vector_store import get_vector_store
from src.infrastructure.database import get_db
from src.infrastructure.config import get_settings
from src.infrastructure.logging import get_logger
from src.api.routes.auth import get_current_user
from src.core.models import User

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a document for processing"""
    
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_upload_size} bytes"
        )
    
    try:
        # Create document record
        storage = get_storage()
        doc_service = DocumentService(db, storage)
        
        document = await doc_service.create_document(
            user_id=current_user.id,
            filename=file.filename,
            file_content=content,
            mime_type=file.content_type
        )
        
        # Start background processing
        background_tasks.add_task(process_document_pipeline, document.id, db)
        
        logger.info(f"Document uploaded: {document.id} by user {current_user.id}")
        
        return DocumentUploadResponse.model_validate(document)
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


async def process_document_pipeline(document_id: str, db: Session):
    """Background task to process document"""
    try:
        # Initialize services
        storage = get_storage()
        doc_service = DocumentService(db, storage)
        pdf_extractor = get_pdf_extractor()
        embedding_adapter = get_embedding_adapter()
        vector_store = get_vector_store()
        
        chunking_service = ChunkingService(db, embedding_adapter, vector_store)
        ingestion_service = IngestionService(db, doc_service, pdf_extractor, chunking_service)
        
        # Process document
        await ingestion_service.process_document(document_id)
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")


@router.get("", response_model=DocumentListResponse)
def list_documents(
    page: int = 1,
    page_size: int = 20,
    status: Optional[DocumentStatus] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's documents"""
    storage = get_storage()
    doc_service = DocumentService(db, storage)
    
    return doc_service.list_documents(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status=status
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document details"""
    storage = get_storage()
    doc_service = DocumentService(db, storage)
    
    document = doc_service.get_document(document_id, current_user.id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    storage = get_storage()
    doc_service = DocumentService(db, storage)
    
    success = doc_service.delete_document(document_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )


@router.get("/{document_id}/status", response_model=ProcessingStatusResponse)
def get_processing_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document processing status"""
    storage = get_storage()
    doc_service = DocumentService(db, storage)
    
    document = doc_service.get_document(document_id, current_user.id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Calculate progress
    status_progress = {
        DocumentStatus.UPLOADED: 0.0,
        DocumentStatus.VALIDATING: 0.1,
        DocumentStatus.EXTRACTING: 0.3,
        DocumentStatus.CHUNKING: 0.5,
        DocumentStatus.EMBEDDING: 0.7,
        DocumentStatus.READY: 1.0,
        DocumentStatus.FAILED: 0.0
    }
    
    return ProcessingStatusResponse(
        document_id=document.id,
        status=document.status,
        progress=status_progress.get(document.status, 0.0),
        current_stage=document.status.value,
        error_message=document.error_message
    )


@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
def get_document_chunks(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all chunks for a document"""
    storage = get_storage()
    doc_service = DocumentService(db, storage)
    
    # Verify document belongs to user
    document = doc_service.get_document(document_id, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get chunks
    embedding_adapter = get_embedding_adapter()
    vector_store = get_vector_store()
    chunking_service = ChunkingService(db, embedding_adapter, vector_store)
    
    chunks = chunking_service.get_document_chunks(document_id)
    
    return [ChunkResponse.model_validate(chunk) for chunk in chunks]

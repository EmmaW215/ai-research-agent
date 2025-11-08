"""Pydantic schemas for document operations"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from .document_models import DocumentStatus


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    id: str
    filename: str
    file_size: int
    status: DocumentStatus
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class DocumentMetadata(BaseModel):
    """Document metadata"""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    num_pages: Optional[int] = None


class DocumentResponse(BaseModel):
    """Detailed document response"""
    id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    status: DocumentStatus
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None
    extraction_method: Optional[str] = None
    chunking_strategy: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Response for document list"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class ChunkResponse(BaseModel):
    """Document chunk response"""
    id: str
    document_id: str
    chunk_index: int
    chunk_text: str
    chunk_size: int
    chunking_strategy: str
    chunk_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ProcessingStatusResponse(BaseModel):
    """Document processing status"""
    document_id: str
    status: DocumentStatus
    progress: float  # 0.0 to 1.0
    current_stage: str
    error_message: Optional[str] = None
    estimated_time_remaining: Optional[int] = None  # seconds

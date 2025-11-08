"""Document domain models"""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .models import Base


class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    """Document entity"""
    
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    status = Column(String(50), default=DocumentStatus.UPLOADED, nullable=False)
    
    # Extracted metadata
    title = Column(String(512), nullable=True)
    authors = Column(JSON, nullable=True)
    abstract = Column(Text, nullable=True)
    doi = Column(String(255), nullable=True)
    
    # Processing metadata
    num_pages = Column(Integer, nullable=True)
    num_chunks = Column(Integer, nullable=True)
    extraction_method = Column(String(50), nullable=True)
    chunking_strategy = Column(String(50), nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status={self.status})>"


class DocumentChunk(Base):
    """Document chunk entity"""
    
    __tablename__ = "document_chunks"
    
    id = Column(String(36), primary_key=True)  # UUID
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    
    # Chunking metadata
    chunking_strategy = Column(String(50), nullable=False)
    chunk_metadata = Column(JSON, nullable=True)
    
    # Vector store reference
    vector_id = Column(String(255), nullable=True)  # ChromaDB ID
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"

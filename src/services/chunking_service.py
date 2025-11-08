"""Chunking service"""

import uuid
from typing import List
from sqlalchemy.orm import Session
from src.core.document_models import Document, DocumentChunk
from src.adapters.chunking_strategies import get_chunker, Chunk
from src.adapters.embeddings import EmbeddingAdapter
from src.adapters.vector_store import VectorStore
from src.infrastructure.config import get_settings
from src.infrastructure.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ChunkingService:
    """Service for text chunking operations"""
    
    def __init__(
        self,
        db: Session,
        embedding_adapter: EmbeddingAdapter,
        vector_store: VectorStore
    ):
        """Initialize chunking service"""
        self.db = db
        self.embedding_adapter = embedding_adapter
        self.vector_store = vector_store
    
    def chunk_and_embed(
        self,
        document: Document,
        text: str,
        strategy: str = "recursive"
    ) -> List[DocumentChunk]:
        """Chunk text, generate embeddings, and store"""
        try:
            logger.info(f"Chunking document {document.id} using {strategy} strategy")
            
            # Get chunker
            chunker = get_chunker(strategy)
            
            # Chunk text
            chunks = chunker.chunk(text)
            
            if not chunks:
                logger.warning(f"No chunks created for document {document.id}")
                return []
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_adapter.embed_batch(chunk_texts)
            
            # Create database records and vector store entries
            db_chunks = []
            vector_ids = []
            vector_texts = []
            vector_embeddings = []
            vector_metadatas = []
            
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create database record
                chunk_id = str(uuid.uuid4())
                vector_id = f"{document.id}_chunk_{idx}"
                
                db_chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    chunk_index=idx,
                    chunk_text=chunk.text,
                    chunk_size=len(chunk.text),
                    chunking_strategy=strategy,
                    chunk_metadata=chunk.metadata,
                    vector_id=vector_id
                )
                
                self.db.add(db_chunk)
                db_chunks.append(db_chunk)
                
                # Prepare vector store data
                vector_ids.append(vector_id)
                vector_texts.append(chunk.text)
                vector_embeddings.append(embedding)
                vector_metadatas.append({
                    "document_id": document.id,
                    "chunk_index": idx,
                    "filename": document.filename,
                    "chunking_strategy": strategy
                })
            
            # Commit database changes
            self.db.commit()
            
            # Store in vector database
            self.vector_store.collection.add(
                ids=vector_ids,
                documents=vector_texts,
                embeddings=vector_embeddings,
                metadatas=vector_metadatas
            )
            
            logger.info(f"Created {len(db_chunks)} chunks for document {document.id}")
            return db_chunks
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error chunking document: {str(e)}")
            raise
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        return self.db.query(DocumentChunk)\
            .filter(DocumentChunk.document_id == document_id)\
            .order_by(DocumentChunk.chunk_index)\
            .all()

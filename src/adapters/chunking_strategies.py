"""Text chunking strategies"""

from typing import Any, Dict, Callable, Iterable, List
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.infrastructure.config import get_settings
from src.infrastructure.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


@dataclass
class Chunk:
    """Chunk data class"""
    text: str
    index: int
    metadata: Dict[str, Any]


class RecursiveChunker:
    """Recursive character text splitting"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """Initialize recursive chunker"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        logger.info(f"Recursive chunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into chunks"""
        try:
            # Split text
            chunks_text = self.splitter.split_text(text)
            
            # Create chunk objects
            chunks = []
            for idx, chunk_text in enumerate(chunks_text):
                chunks.append(Chunk(
                    text=chunk_text,
                    index=idx,
                    metadata={
                        "chunk_size": len(chunk_text),
                        "strategy": "recursive",
                        "overlap": self.chunk_overlap
                    }
                ))
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in recursive chunking: {str(e)}")
            raise


class SemanticChunker:
    """Semantic chunking based on sentence embeddings"""
    
    def __init__(
        self,
        embedding_adapter,
        similarity_threshold: float = 0.7,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """Initialize semantic chunker"""
        self.embedding_adapter = embedding_adapter
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        logger.info(f"Semantic chunker initialized: threshold={similarity_threshold}")
    
    def chunk(self, text: str) -> List[Chunk]:
        """Split text into semantic chunks"""
        try:
            # Split into sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if not sentences:
                return []
            
            # Generate embeddings for all sentences
            embeddings = self.embedding_adapter.embed_batch(sentences)
            
            # Group sentences by semantic similarity
            chunks = []
            current_chunk = [sentences[0]]
            current_embedding = embeddings[0]
            
            for i in range(1, len(sentences)):
                similarity = self.embedding_adapter.compute_similarity(
                    current_embedding,
                    embeddings[i]
                )
                
                current_size = sum(len(s) for s in current_chunk)
                
                # Check if we should add to current chunk or start new one
                if (similarity >= self.similarity_threshold and 
                    current_size + len(sentences[i]) <= self.max_chunk_size):
                    current_chunk.append(sentences[i])
                else:
                    # Save current chunk if it meets minimum size
                    if current_size >= self.min_chunk_size:
                        chunk_text = " ".join(current_chunk)
                        chunks.append(Chunk(
                            text=chunk_text,
                            index=len(chunks),
                            metadata={
                                "chunk_size": len(chunk_text),
                                "strategy": "semantic",
                                "num_sentences": len(current_chunk),
                                "avg_similarity": similarity
                            }
                        ))
                    
                    # Start new chunk
                    current_chunk = [sentences[i]]
                    current_embedding = embeddings[i]
            
            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=len(chunks),
                        metadata={
                            "chunk_size": len(chunk_text),
                            "strategy": "semantic",
                            "num_sentences": len(current_chunk)
                        }
                    ))
            
            logger.info(f"Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            raise


def get_chunker(strategy: str = "recursive", **kwargs):
    """Factory function to get chunker by strategy"""
    if strategy == "recursive":
        return RecursiveChunker(
            chunk_size=kwargs.get("chunk_size", settings.default_chunk_size),
            chunk_overlap=kwargs.get("chunk_overlap", settings.default_chunk_overlap)
        )
    elif strategy == "semantic":
        from src.adapters.embeddings import get_embedding_adapter
        return SemanticChunker(
            embedding_adapter=get_embedding_adapter(),
            similarity_threshold=kwargs.get("similarity_threshold", 0.7)
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

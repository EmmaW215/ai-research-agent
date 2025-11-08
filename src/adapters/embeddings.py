"""Embedding generation adapter"""

from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from src.infrastructure.config import get_settings
from src.infrastructure.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EmbeddingAdapter:
    """Embedding generation using sentence-transformers"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize embedding adapter"""
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}, Device: {self.device}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=True)
            return embedding.cpu().tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            batch_size = batch_size or settings.embedding_batch_size
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True
            )
            
            return embeddings.cpu().tolist()
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(cosine_sim)


# Singleton instance
_embedding_adapter: Optional[EmbeddingAdapter] = None


def get_embedding_adapter() -> EmbeddingAdapter:
    """Get or create embedding adapter instance"""
    global _embedding_adapter
    if _embedding_adapter is None:
        _embedding_adapter = EmbeddingAdapter()
    return _embedding_adapter

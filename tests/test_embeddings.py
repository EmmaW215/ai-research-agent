"""Tests for embedding generation"""

import pytest
from src.adapters.embeddings import EmbeddingAdapter


@pytest.fixture
def embedding_adapter():
    """Create embedding adapter instance"""
    return EmbeddingAdapter(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_embedding_adapter_initialization(embedding_adapter):
    """Test embedding adapter initialization"""
    assert embedding_adapter.model is not None
    assert embedding_adapter.dimension > 0


def test_single_embedding_generation(embedding_adapter):
    """Test generating single embedding"""
    text = "This is a test sentence."
    embedding = embedding_adapter.embed_text(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == embedding_adapter.dimension
    assert all(isinstance(x, float) for x in embedding)


def test_batch_embedding_generation(embedding_adapter):
    """Test generating batch embeddings"""
    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence."
    ]
    
    embeddings = embedding_adapter.embed_batch(texts)
    
    assert len(embeddings) == len(texts)
    assert all(len(emb) == embedding_adapter.dimension for emb in embeddings)

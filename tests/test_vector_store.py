"""Tests for vector store adapter"""

import pytest
from src.adapters.vector_store import VectorStore


@pytest.fixture
def vector_store():
    """Create a test vector store instance"""
    return VectorStore()


def test_vector_store_initialization(vector_store):
    """Test vector store initialization"""
    assert vector_store.client is not None
    assert vector_store.collection is not None
    
    stats = vector_store.get_collection_stats()
    assert "name" in stats
    assert "count" in stats


def test_add_and_search_documents(vector_store):
    """Test adding and searching documents"""
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text."
    ]
    
    metadatas = [
        {"topic": "ML", "year": 2023},
        {"topic": "DL", "year": 2023},
        {"topic": "NLP", "year": 2023}
    ]
    
    ids = ["doc1", "doc2", "doc3"]
    
    vector_store.add_documents(documents, metadatas, ids)
    
    stats = vector_store.get_collection_stats()
    assert stats["count"] >= 3
    
    results = vector_store.search("What is machine learning?", n_results=2)
    
    assert "ids" in results
    assert len(results["ids"][0]) <= 2

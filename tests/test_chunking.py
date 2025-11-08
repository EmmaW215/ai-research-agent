"""Tests for chunking strategies"""

import pytest
from src.adapters.chunking_strategies import RecursiveChunker, get_chunker


@pytest.fixture
def sample_text():
    """Sample text for chunking"""
    return """
    Machine learning is a subset of artificial intelligence.
    It focuses on building systems that can learn from data.
    Deep learning uses neural networks with multiple layers.
    Natural language processing helps computers understand text.
    """


def test_recursive_chunker_initialization():
    """Test recursive chunker initialization"""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 10


def test_recursive_chunking(sample_text):
    """Test recursive text chunking"""
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(sample_text)
    
    assert len(chunks) > 0
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.index >= 0 for chunk in chunks)


def test_get_chunker_factory():
    """Test chunker factory function"""
    chunker = get_chunker("recursive", chunk_size=200)
    assert chunker is not None
    assert chunker.chunk_size == 200

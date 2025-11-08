"""Tests for document ingestion pipeline"""

import pytest
from src.core.document_models import DocumentStatus


# Integration tests for full pipeline
# These would require test PDF files and database setup

def test_pipeline_stages():
    """Test that all pipeline stages are defined"""
    stages = [
        DocumentStatus.UPLOADED,
        DocumentStatus.VALIDATING,
        DocumentStatus.EXTRACTING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.READY,
        DocumentStatus.FAILED
    ]
    
    assert all(isinstance(stage, DocumentStatus) for stage in stages)

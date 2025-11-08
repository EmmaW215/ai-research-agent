"""Tests for document API endpoints"""

import pytest
from fastapi import status
from io import BytesIO


def test_upload_document_unauthorized(client):
    """Test document upload without authentication"""
    response = client.post("/api/v1/documents/upload")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_documents(authenticated_client):
    """Test listing documents"""
    response = authenticated_client.get("/api/v1/documents")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "documents" in data
    assert "total" in data


# Add more tests with sample PDF uploads

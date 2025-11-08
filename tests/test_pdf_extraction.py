"""Tests for PDF extraction"""

import pytest
from src.adapters.pdf_extractor import PDFExtractor


@pytest.fixture
def pdf_extractor():
    """Create PDF extractor instance"""
    return PDFExtractor()


def test_pdf_extractor_initialization(pdf_extractor):
    """Test PDF extractor initialization"""
    assert pdf_extractor is not None


# Add more tests with sample PDF files

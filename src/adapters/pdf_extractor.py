"""PDF text extraction adapter"""

from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
import pdfplumber
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """PDF text extraction using PyMuPDF and pdfplumber"""
    
    def __init__(self):
        """Initialize PDF extractor"""
        logger.info("PDF extractor initialized")
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_content.append(text)
            
            doc.close()
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from {len(doc)} pages")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Extract additional info
            result = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "num_pages": len(doc),
                "format": metadata.get("format", ""),
            }
            
            doc.close()
            
            logger.info(f"Extracted metadata: {result.get('title', 'Untitled')}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}
    
    def extract_with_layout(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text with layout preservation"""
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text with layout
                blocks = page.get_text("dict")["blocks"]
                page_text = []
                
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            page_text.append(line_text)
                
                pages_content.append({
                    "page_num": page_num + 1,
                    "text": "\n".join(page_text)
                })
            
            doc.close()
            
            return {
                "num_pages": len(doc),
                "pages": pages_content
            }
            
        except Exception as e:
            logger.error(f"Error extracting layout: {str(e)}")
            raise
    
    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber"""
        try:
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(page_tables):
                        tables.append({
                            "page": page_num + 1,
                            "table_index": table_idx,
                            "data": table
                        })
            
            logger.info(f"Extracted {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove page numbers (common patterns)
        import re
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        
        return text


# Singleton instance
_pdf_extractor: Optional[PDFExtractor] = None


def get_pdf_extractor() -> PDFExtractor:
    """Get or create PDF extractor instance"""
    global _pdf_extractor
    if _pdf_extractor is None:
        _pdf_extractor = PDFExtractor()
    return _pdf_extractor

#!/usr/bin/env python
"""Run database migrations for Batch 2"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.database import engine
from src.core.models import Base
from src.core.document_models import Document, DocumentChunk

def run_migrations():
    """Create new tables for Batch 2"""
    print("=" * 60)
    print("Running Batch 2 Database Migrations")
    print("=" * 60)
    print()
    
    try:
        print("Creating new tables...")
        Base.metadata.create_all(bind=engine)
        
        print("✓ documents table created")
        print("✓ document_chunks table created")
        print()
        print("=" * 60)
        print("✓ Migrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Migration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_migrations()

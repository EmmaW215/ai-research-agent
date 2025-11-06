"""Health check endpoints"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.infrastructure.database import get_db
from src.infrastructure.config import get_settings
from src.adapters.vector_store import get_vector_store, VectorStore

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check(
    db: Session = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Basic health check endpoint"""
    # Check database connection
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Check vector store
    try:
        stats = vector_store.get_collection_stats()
        vector_store_status = "healthy"
        vector_store_stats = stats
    except Exception as e:
        vector_store_status = f"unhealthy: {str(e)}"
        vector_store_stats = None
    
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "database": db_status,
        "vector_store": {
            "status": vector_store_status,
            "stats": vector_store_stats
        }
    }

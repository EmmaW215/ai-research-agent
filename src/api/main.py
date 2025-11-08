"""FastAPI application entry point"""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.infrastructure.config import get_settings
from src.infrastructure.database import init_db
from src.infrastructure.logging import setup_logging, get_logger
from src.api.routes import auth, health
from src.api.routes import documents

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered research assistant with voice interaction and RAG",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests with correlation ID and timing"""
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    
    start_time = time.time()
    
    logger.info(
        "Request started",
        extra={
            "correlation_id": correlation_id,
            "request_method": request.method,
            "request_path": request.url.path,
        }
    )
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    
    logger.info(
        "Request completed",
        extra={
            "correlation_id": correlation_id,
            "request_method": request.method,
            "request_path": request.url.path,
            "response_status": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }
    )
    
    response.headers["X-Correlation-ID"] = correlation_id
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "correlation_id": correlation_id,
            "request_method": request.method,
            "request_path": request.url.path,
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "correlation_id": correlation_id
        }
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )

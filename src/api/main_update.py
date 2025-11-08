# Add this to src/api/main.py after the existing routers

# Import at the top
from src.api.routes import documents

# Add this after the existing router includes
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])

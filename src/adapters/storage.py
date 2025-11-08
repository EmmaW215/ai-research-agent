"""File storage adapter"""

import os
import shutil
from pathlib import Path
from typing import BinaryIO, Optional
import aiofiles
from src.infrastructure.config import get_settings
from src.infrastructure.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class StorageAdapter:
    """File storage adapter (local filesystem)"""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize storage adapter"""
        self.base_path = Path(base_path or settings.upload_directory)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage adapter initialized: {self.base_path}")
    
    async def save_file(self, file_content: bytes, filename: str, user_id: int) -> str:
        """Save file and return file path"""
        try:
            # Create user directory
            user_dir = self.base_path / str(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            file_path = user_dir / filename
            
            # Handle duplicate filenames
            counter = 1
            while file_path.exists():
                name, ext = os.path.splitext(filename)
                file_path = user_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            # Save file asynchronously
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
    
    async def read_file(self, file_path: str) -> bytes:
        """Read file content"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            raise
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        return os.path.exists(file_path)


# Singleton instance
_storage: Optional[StorageAdapter] = None


def get_storage() -> StorageAdapter:
    """Get or create storage adapter instance"""
    global _storage
    if _storage is None:
        _storage = StorageAdapter()
    return _storage

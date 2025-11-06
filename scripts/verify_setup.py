#!/usr/bin/env python
"""Verify installation and setup"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    if sys.version_info < (3, 11):
        print("  ✗ Python 3.11+ required")
        return False
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "pydantic",
        "jose",
        "passlib",
        "chromadb"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not found")
            missing.append(package)
    
    return len(missing) == 0


def check_env_file():
    """Check if .env file exists"""
    print("\nChecking environment configuration...")
    env_file = Path(".env")
    if env_file.exists():
        print("  ✓ .env file exists")
        return True
    else:
        print("  ⚠ .env file not found")
        print("    Copy .env.example to .env and configure it")
        return False


def check_database():
    """Check database connection"""
    print("\nChecking database connection...")
    try:
        from src.infrastructure.database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("  ✓ Database connection successful")
        return True
    except Exception as e:
        print(f"  ✗ Database connection failed: {str(e)}")
        return False


def check_vector_store():
    """Check vector store setup"""
    print("\nChecking vector store...")
    try:
        from src.adapters.vector_store import get_vector_store
        vs = get_vector_store()
        stats = vs.get_collection_stats()
        print(f"  ✓ Vector store initialized")
        print(f"    Collection: {stats['name']}")
        print(f"    Documents: {stats['count']}")
        return True
    except Exception as e:
        print(f"  ✗ Vector store check failed: {str(e)}")
        return False


def main():
    """Main verification function"""
    print("=" * 50)
    print("AI Research Agent - Installation Verification")
    print("=" * 50)
    print()
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_env_file(),
        check_database(),
        check_vector_store()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("✓ All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. Run: uvicorn src.api.main:app --reload")
        print("  2. Visit: http://localhost:8000/docs")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    
    print("=" * 50)


if __name__ == "__main__":
    main()

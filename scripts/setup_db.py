#!/usr/bin/env python
"""Database setup script"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.database import init_db, engine
from src.core.models import Base, User, UserRole
from src.infrastructure.security import get_password_hash
from sqlalchemy.orm import Session


def create_tables():
    """Create all database tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created successfully")


def create_admin_user():
    """Create default admin user"""
    print("\nCreating admin user...")
    
    session = Session(bind=engine)
    
    try:
        admin = session.query(User).filter(
            User.email == "admin@example.com"
        ).first()
        
        if admin:
            print("⚠ Admin user already exists")
            return
        
        admin = User(
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            full_name="System Administrator",
            role=UserRole.ADMIN,
            is_active=True
        )
        
        session.add(admin)
        session.commit()
        
        print("✓ Admin user created successfully")
        print("  Email: admin@example.com")
        print("  Password: admin123")
        print("  ⚠ Please change the password after first login!")
        
    except Exception as e:
        print(f"✗ Error creating admin user: {str(e)}")
        session.rollback()
    finally:
        session.close()


def main():
    """Main setup function"""
    print("=" * 50)
    print("AI Research Agent - Database Setup")
    print("=" * 50)
    
    try:
        create_tables()
        create_admin_user()
        
        print("\n" + "=" * 50)
        print("✓ Database setup completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

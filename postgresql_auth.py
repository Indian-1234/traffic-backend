from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional

# Database Configuration
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:1234@localhost:5432/traffic_db"

# Create SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# User database model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

# Pydantic models for request/response
class UserCreate(BaseModel):
    name: str
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    username: str
    email: str
    
    class Config:
        orm_mode = True

# Create the tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password verification function
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Password hashing function
def get_password_hash(password):
    return pwd_context.hash(password)

# Authentication function
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user

# Add these routes to your FastAPI app
def setup_auth_routes(app: FastAPI):
    @app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    async def register_user(user: UserCreate, db: Session = Depends(get_db)):
        """Register a new user"""
        # Check if username already exists
        db_user = db.query(User).filter(User.username == user.username).first()
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash the password
        hashed_password = get_password_hash(user.password)
        
        # Create new user
        new_user = User(
            name=user.name,
            username=user.username,
            email=user.email,
            password=hashed_password
        )
        
        # Add to database
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return new_user

    @app.post("/login")
    async def login_user(form_data: UserLogin, db: Session = Depends(get_db)):
        """Login a user"""
        user = authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # For a basic system, just return user info without tokens
        return {
            "id": user.id,
            "name": user.name,
            "username": user.username,
            "email": user.email,
            "status": "success",
            "message": "Login successful"
        }
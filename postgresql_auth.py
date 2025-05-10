# postgresql_auth.py

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import Column, String, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from pydantic import BaseModel

# Database Configuration
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:1234@localhost:5432/traffic_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# SQLAlchemy User model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

# Pydantic schemas
class UserCreate(BaseModel):
    name: str
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    username: str
    email: str
    class Config:
        orm_mode = True

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# Authenticate user
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password):
        return None
    return user

# Auth router setup
def setup_auth_routes(app):
    @app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    async def register_user(user: UserCreate, db: Session = Depends(get_db)):
        if db.query(User).filter(User.username == user.username).first():
            raise HTTPException(status_code=400, detail="Username already registered")
        if db.query(User).filter(User.email == user.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = get_password_hash(user.password)
        new_user = User(
            name=user.name,
            username=user.username,
            email=user.email,
            password=hashed_password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    @app.post("/login")
    async def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
        user = authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {
            "id": user.id,
            "name": user.name,
            "username": user.username,
            "email": user.email,
            "status": "success",
            "message": "Login successful"
        }

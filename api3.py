from fastapi import FastAPI, HTTPException, Query, Depends, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional, Dict, Union, Any
import numpy as np
import requests
from datetime import datetime, timedelta
import networkx as nx
import os
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, DateTime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import declarative_base  # No longer deprecated in this context
# from quantum.traffic_opt import SynchronousTrafficNetworkOptimizer
from routes import traffic_router
# from routes.optimize_route import router as optimize_router
from contextlib import asynccontextmanager

import pandas as pd
import uuid
import os
from datetime import datetime
from fastapi import HTTPException, Depends, File, UploadFile
import xml.etree.ElementTree as ET

# Import your models and other required components
from models import TrafficRequest, TrafficPrediction, OptimizationResult, LiveTrafficData
from quantum_config import QuantumTrafficPredictor, generate_training_data
from traffic_graph import TrafficGraph


# Configuration Settings
TOMTOM_API_KEY = "IV7dQDp5vey54vgGvRlIDmn7qazKzAaN"  # Replace with your actual API key
TOMTOM_TRAFFIC_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
MODEL_SAVE_PATH = "quantum_traffic_model.json"

# JWT Authentication Settings
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"  # Replace with your secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database Configuration
DATABASE_URL = "postgresql://postgres:1234@localhost:5432/traffic_db"  # Replace with your DB credentials
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize the FastAPI app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialization code (previously in startup_event)
    # Create database tables
    Base.metadata.create_all(bind=engine)
    global quantum_optimizer
    # quantum_optimizer = SynchronousTrafficNetworkOptimizer(n_qubits=8)
    print("Quantum traffic optimizer initialized")
    
    yield
    # Create admin user if not exists
    db = SessionLocal()
    try:
        admin_user = get_user(db, "admin")
        if not admin_user:
            admin = UserCreate(
                email="admin@trafficapi.com",
                username="admin",
                password="adminpassword"  # Change this in production
            )
            admin_user = create_user(db, admin)
            admin_user.is_admin = True
            db.commit()
            print("Created admin user")
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        db.close()
    
    # Check if model file exists
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_model(MODEL_SAVE_PATH)
        print(f"Loaded model from {MODEL_SAVE_PATH}")
    else:
        # Train model with synthetic data
        print("Training model with synthetic data...")
        features, targets = generate_training_data(n_samples=100)
        loss = model.train(features, targets, n_iterations=50)
        model.save_model(MODEL_SAVE_PATH)
        print(f"Trained model (loss: {loss:.4f}) and saved to {MODEL_SAVE_PATH}")
    
    # Initialize sample traffic network
    print("Initializing traffic network...")
    # Add some nodes (intersections)
    for i in range(10):
        traffic_network.add_node(i, 
                                type="intersection", 
                                latitude=40.7 + i*0.01, 
                                longitude=-74.0 + i*0.01)
    
    # Add edges (road segments)
    for i in range(9):
        # Two-way roads between adjacent intersections
        traffic_network.add_edge(i, i+1, weight=1.0, distance=0.5, speed_limit=50)
        traffic_network.add_edge(i+1, i, weight=1.0, distance=0.5, speed_limit=50)
    
    # Add some diagonal connections
    for i in range(7):
        traffic_network.add_edge(i, i+3, weight=1.5, distance=1.2, speed_limit=60)
        traffic_network.add_edge(i+3, i, weight=1.5, distance=1.2, speed_limit=60)
    
    print("Initialization complete!")
    
    yield  # This is where the app runs
    
    # Cleanup code (if any) would go here

# Update the FastAPI app initialization
app = FastAPI(
    title="Quantum Traffic Optimization API",
    description="API for traffic prediction and optimization using quantum computing",
    version="1.0.0",
    lifespan=lifespan  # Add this line
)

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# Add CORS middleware with multiple origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize the model
model = QuantumTrafficPredictor(n_qubits=4)

# Initialize the traffic graph
traffic_network = TrafficGraph()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

# Authentication Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True  # Changed from orm_mode = True

# Database helper functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=get_password_hash(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    # Update last login time
    user.last_login = datetime.utcnow()
    db.commit()
    return user
async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_admin_user(current_user: UserResponse = Depends(get_current_active_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

def get_tomtom_traffic(lat: float, lon: float) -> Optional[dict]:
    """Fetch traffic data from TomTom API"""
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{lat},{lon}"
    }
    try:
        response = requests.get(TOMTOM_TRAFFIC_URL, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"TomTom API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching TomTom data: {e}")
        return None


# @app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    # Create database tables
    Base.metadata.create_all(bind=engine)
    global quantum_optimizer
    # quantum_optimizer = SynchronousTrafficNetworkOptimizer(n_qubits=8)
    
    print("Quantum traffic optimizer initialized")

    # Create admin user if not exists
    db = SessionLocal()
    try:
        admin_user = get_user(db, "admin")
        if not admin_user:
            admin = UserCreate(
                email="admin@trafficapi.com",
                username="admin",
                password="adminpassword"  # Change this in production
            )
            admin_user = create_user(db, admin)
            admin_user.is_admin = True
            db.commit()
            print("Created admin user")
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        db.close()
    
    # Check if model file exists
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_model(MODEL_SAVE_PATH)
        print(f"Loaded model from {MODEL_SAVE_PATH}")
    else:
        # Train model with synthetic data
        print("Training model with synthetic data...")
        features, targets = generate_training_data(n_samples=100)
        loss = model.train(features, targets, n_iterations=50)
        model.save_model(MODEL_SAVE_PATH)
        print(f"Trained model (loss: {loss:.4f}) and saved to {MODEL_SAVE_PATH}")
    
    # Initialize sample traffic network
    print("Initializing traffic network...")
    # Add some nodes (intersections)
    for i in range(10):
        traffic_network.add_node(i, 
                                type="intersection", 
                                latitude=40.7 + i*0.01, 
                                longitude=-74.0 + i*0.01)
    
    # Add edges (road segments)
    for i in range(9):
        # Two-way roads between adjacent intersections
        traffic_network.add_edge(i, i+1, weight=1.0, distance=0.5, speed_limit=50)
        traffic_network.add_edge(i+1, i, weight=1.0, distance=0.5, speed_limit=50)
    
    # Add some diagonal connections
    for i in range(7):
        traffic_network.add_edge(i, i+3, weight=1.5, distance=1.2, speed_limit=60)
        traffic_network.add_edge(i+3, i, weight=1.5, distance=1.2, speed_limit=60)
    
    print("Initialization complete!")
# app.include_router(optimize_router)
app.include_router(traffic_router.router)
# Authentication endpoints
@app.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    username_exists = get_user(db, username=user.username)
    if username_exists:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    return create_user(db, user)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.post("/predict", response_model=TrafficPrediction)
async def predict_traffic(
    data: TrafficRequest, 
    current_user: User = Depends(get_current_active_user)
):
    """Predict traffic conditions based on input parameters"""
    try:
        # Get external traffic data
        tomtom_data = get_tomtom_traffic(data.latitude, data.longitude)
        
        # Extract current traffic flow rate
        if tomtom_data and "flowSegmentData" in tomtom_data:
            flow_rate = tomtom_data["flowSegmentData"].get("currentSpeed", 30) / \
                        tomtom_data["flowSegmentData"].get("freeFlowSpeed", 50)
        else:
            # Default values if API fails
            flow_rate = 0.7
        
        # Create feature vector for model
        features = [
            data.vehicle_count,
            data.weather_condition,
            data.time_of_day / 24.0,
            data.day_of_week / 6.0
        ]
        
        # Get prediction from quantum model
        congestion_prediction = model.predict(features)
        
        # Calculate additional metrics
        signalization_density = 0.5  # Placeholder - would come from real data
        
        # Calculate recommended speed
        max_speed = 50.0  # km/h
        recommended_speed = max_speed * (1 - congestion_prediction * 0.7)
        
        # Calculate estimated travel time
        base_time = 15.0  # minutes
        estimated_travel_time = base_time / (1 - congestion_prediction * 0.6)
        
        # Calculate confidence score
        confidence = 0.85 - 0.2 * abs(flow_rate - congestion_prediction)
        
        return TrafficPrediction(
            prediction=float(congestion_prediction),
            flow_rate=float(flow_rate),
            signalization_density=float(signalization_density),
            recommended_speed=float(recommended_speed),
            estimated_travel_time=float(estimated_travel_time),
            confidence=float(confidence)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/optimize-route", response_model=OptimizationResult)
async def optimize_route(
    start_node: int = Query(..., description="Starting intersection ID"),
    end_node: int = Query(..., description="Destination intersection ID"),
    departure_time: int = Query(..., description="Departure time (hour of day, 0-23)"),
    current_user: User = Depends(get_current_active_user),
    token: str = Depends(oauth2_scheme)  # 👈 ADD THIS


):
    """Optimize route between two points using quantum optimization"""
    try:
        print(token)
        # Extract subgraph for optimization
        try:
            subgraph = traffic_network.extract_subgraph(start_node, radius=5)
        except nx.NodeNotFound:
            raise HTTPException(status_code=404, detail=f"Node {start_node} not found")
        
        # Update edge weights based on predicted traffic
        for u, v, data in subgraph.edges(data=True):
            # Get node coordinates
            u_lat = subgraph.nodes[u].get("latitude", 40.7)
            u_lon = subgraph.nodes[u].get("longitude", -74.0)
            
            # Create a request for this segment
            request_data = TrafficRequest(
                vehicle_count=100,  # Placeholder
                weather_condition=0.2,  # Placeholder
                latitude=u_lat,
                longitude=u_lon,
                time_of_day=departure_time,
                day_of_week=datetime.now().weekday()
            )
            
            # Get prediction
            prediction = await predict_traffic(request_data, current_user)
            
            # Update edge weight based on congestion
            congestion_factor = 1 + prediction.prediction * 2  # Scale congestion impact
            traffic_network.update_edge_weight(u, v, data["weight"] * congestion_factor)
        
        # Find optimal path
        path, distance = traffic_network.compute_shortest_path(start_node, end_node)
        
        if not path:
            raise HTTPException(status_code=404, detail="No path found")
        
        # Prepare route information
        route_segments = []
        total_distance = 0
        total_time = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = traffic_network.graph[u][v]
            
            segment_distance = edge_data.get("distance", 0.5)
            speed_limit = edge_data.get("speed_limit", 50)
            congestion = edge_data.get("weight", 1.0)
            
            # Calculate time for this segment
            segment_time = segment_distance / (speed_limit / 60) * congestion
            
            total_distance += segment_distance
            total_time += segment_time
            
            route_segments.append({
                "from_node": str(u),
                "to_node": str(v),
                "distance_km": segment_distance,
                "estimated_time_min": segment_time,
                "congestion_factor": congestion
            })
        
        # Calculate optimization metrics
        standard_time = total_distance / (50 / 60)  # Time without optimization
        time_saved = standard_time - total_time
        traffic_reduction = 0.35  # Placeholder - would be calculated from real data
        
        return OptimizationResult(
            optimal_routes=route_segments,
            traffic_reduction=traffic_reduction,
            average_time_saved=time_saved
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/live-data", response_model=List[LiveTrafficData])
async def get_live_traffic_data(
    area: str = Query(None, description="Area name or identifier"),
    radius: float = Query(2.0, description="Radius in km from center point"),
    lat: Optional[float] = Query(None, description="Latitude for center point"),
    lon: Optional[float] = Query(None, description="Longitude for center point"),
    current_user: User = Depends(get_current_active_user)
):
    """Get live traffic data for the specified area or coordinates"""
    try:
        # Get current timestamp
        current_time = datetime.now().isoformat()
        
        # Determine coordinates based on input
        if lat is None or lon is None:
            # Use predefined coordinates for named areas
            area_coords = {
                "downtown": (40.7128, -74.0060),  # Example: NYC
                "uptown": (40.8075, -73.9626),
                "midtown": (40.7549, -73.9840),
                "airport": (40.6413, -73.7781)
            }
            coordinates = area_coords.get(area.lower(), (40.7128, -74.0060))
        else:
            coordinates = (lat, lon)
            
        # Get external traffic data from TomTom API
        tomtom_data = get_tomtom_traffic(coordinates[0], coordinates[1])
        
        # Generate live traffic data for multiple segments
        live_data = []
        
        # Get all nodes within radius of center point
        center_node = None
        min_distance = float('inf')
        
        # Find closest node to requested coordinates
        for node_id, node_data in traffic_network.graph.nodes(data=True):
            if 'latitude' in node_data and 'longitude' in node_data:
                node_lat = node_data['latitude']
                node_lon = node_data['longitude']
                
                # Calculate rough distance
                dist = ((node_lat - coordinates[0])**2 + (node_lon - coordinates[1])**2)**0.5
                
                if dist < min_distance:
                    min_distance = dist
                    center_node = node_id
        
        if center_node is not None:
            # Extract subgraph within radius
            subgraph = traffic_network.extract_subgraph(center_node, radius=int(radius * 2))
            
            # Create a live data entry for each node
            for node_id in subgraph.nodes():
                node_data = traffic_network.graph.nodes[node_id]
                
                # Skip nodes without geographic coordinates
                if 'latitude' not in node_data or 'longitude' not in node_data:
                    continue
                
                # Create feature vector for this location
                current_hour = datetime.now().hour
                current_day = datetime.now().weekday()
                
                # Generate a traffic prediction for this node
                request_data = TrafficRequest(
                    vehicle_count=np.random.randint(50, 150),  # Simulated vehicle count
                    weather_condition=0.3,  # Simulated weather condition
                    latitude=node_data['latitude'],
                    longitude=node_data['longitude'],
                    time_of_day=current_hour,
                    day_of_week=current_day
                )
                
                prediction = await predict_traffic(request_data, current_user)
                
                # Get adjacent road segments
                road_segments = []
                for _, neighbor, edge_data in traffic_network.graph.edges(node_id, data=True):
                    # Generate random variation for realistic data
                    speed_variation = np.random.uniform(0.8, 1.2)
                    neighbor_data = traffic_network.graph.nodes[neighbor]
                    
                    road_segments.append({
                        "segment_id": f"{node_id}-{neighbor}",
                        "from": {
                            "id": str(node_id),
                            "lat": node_data.get('latitude'),
                            "lon": node_data.get('longitude')
                        },
                        "to": {
                            "id": str(neighbor),
                            "lat": neighbor_data.get('latitude', node_data.get('latitude') + 0.01),
                            "lon": neighbor_data.get('longitude', node_data.get('longitude') + 0.01)
                        },
                        "speed_limit": edge_data.get("speed_limit", 50),
                        "current_speed": edge_data.get("speed_limit", 50) * (1 - prediction.prediction * 0.7) * speed_variation,
                        "congestion": prediction.prediction * np.random.uniform(0.9, 1.1),
                        "distance": edge_data.get("distance", 0.5)
                    })
                
                # Determine traffic trend (randomly for simulation)
                trends = ["increasing", "decreasing", "stable"]
                trend_weights = [0.3, 0.3, 0.4]
                traffic_trend = np.random.choice(trends, p=trend_weights)
                
                # Create live data response
                live_data.append(LiveTrafficData(
                    timestamp=current_time,
                    location={
                        "latitude": node_data['latitude'],
                        "longitude": node_data['longitude']
                    },
                    current_congestion=float(prediction.prediction),
                    vehicle_count=np.random.randint(50, 200),  # Simulated vehicle count
                    average_speed=float(50 * (1 - prediction.prediction * 0.7)),  # Derived from congestion
                    traffic_trend=traffic_trend,
                    road_segments=road_segments
                ))
        
        # If no data from the graph, create a simulated data point
        if not live_data:
            # Create simulated data for the requested coordinates
            congestion = np.random.uniform(0.2, 0.8)
            
            live_data.append(LiveTrafficData(
                timestamp=current_time,
                location={
                    "latitude": coordinates[0],
                    "longitude": coordinates[1]
                },
                current_congestion=congestion,
                vehicle_count=np.random.randint(50, 200),
                average_speed=float(50 * (1 - congestion * 0.7)),
                traffic_trend=np.random.choice(["increasing", "decreasing", "stable"]),
                road_segments=[{
                    "segment_id": "simulated-segment",
                    "from": {
                        "id": "sim-1",
                        "lat": coordinates[0],
                        "lon": coordinates[1]
                    },
                    "to": {
                        "id": "sim-2",
                        "lat": coordinates[0] + 0.01,
                        "lon": coordinates[1] + 0.01
                    },
                    "speed_limit": 50,
                    "current_speed": 50 * (1 - congestion * 0.7),
                    "congestion": congestion,
                    "distance": 0.5
                }]
            ))
        
        return live_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving live data: {str(e)}")


# Integration with Real Traffic Data Sources


class RoadSignDataIntegrator:
    """Integrates with road sign detection data from XML annotation files"""
    
    def __init__(self, annotations_dir="annotations", images_dir="images"):
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        
    def get_road_sign_data(self, latitude, longitude, radius=2.0):
        """Process XML annotation files and return road sign data for a specific location"""
        try:
            # Get list of all XML files in the annotations directory
            annotation_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
            
            # Process each annotation file
            all_signs = []
            for xml_file in annotation_files:
                file_path = os.path.join(self.annotations_dir, xml_file)
                signs = self._parse_xml_annotation(file_path)
                all_signs.extend(signs)
            
            # Filter signs by distance to the provided coordinates
            filtered_signs = []
            for sign in all_signs:
                sign_lat = sign.get("location", {}).get("lat")
                sign_lon = sign.get("location", {}).get("lon")
                
                if sign_lat is not None and sign_lon is not None:
                    distance = self._calculate_distance(latitude, longitude, sign_lat, sign_lon)
                    if distance <= radius:
                        sign["distance"] = distance
                        filtered_signs.append(sign)
            
            return {"signs": filtered_signs}
        except Exception as e:
            print(f"Error processing annotation files: {e}")
            return None
    
    def _parse_xml_annotation(self, xml_file_path):
        """Parse an XML annotation file and extract road sign data"""
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            signs = []
            
            # Get filename without extension to link with image file
            filename = os.path.basename(xml_file_path)
            image_name = os.path.splitext(filename)[0]
            
            # Get image path
            image_path = os.path.join(self.images_dir, f"{image_name}.jpg")
            if not os.path.exists(image_path):
                # Try other common image extensions if jpg doesn't exist
                for ext in ['.png', '.jpeg', '.bmp']:
                    alt_path = os.path.join(self.images_dir, f"{image_name}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            
            # Extract coordinates from filename or XML (implementation depends on your XML structure)
            # This is a placeholder - adjust based on your actual XML structure
            location_data = self._extract_location_from_xml(root)
            
            # Extract objects/signs from XML
            for obj in root.findall('.//object'):
                sign_type = obj.find('name').text if obj.find('name') is not None else "unknown"
                
                # Extract bounding box
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = float(bndbox.find('xmin').text) if bndbox.find('xmin') is not None else 0
                    ymin = float(bndbox.find('ymin').text) if bndbox.find('ymin') is not None else 0
                    xmax = float(bndbox.find('xmax').text) if bndbox.find('xmax') is not None else 0
                    ymax = float(bndbox.find('ymax').text) if bndbox.find('ymax') is not None else 0
                    
                    # Create sign object
                    sign = {
                        "id": f"{image_name}_{sign_type}_{int(xmin)}_{int(ymin)}",
                        "type": sign_type,
                        "location": location_data,
                        "image_path": image_path,
                        "bounding_box": {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax
                        },
                        "confidence": 0.95,  # Default confidence value
                        "timestamp": datetime.now().isoformat(),
                        "road_segment": image_name,  # Using image name as road segment identifier
                        "nearby_signs": [],  # Will be populated later if needed
                        "segment_length": 0.5  # Default segment length
                    }
                    
                    # Extract additional data if available
                    for attr in obj.findall('./attribute'):
                        if attr.find('name') is not None and attr.find('value') is not None:
                            attr_name = attr.find('name').text
                            attr_value = attr.find('value').text
                            sign[attr_name] = attr_value
                    
                    signs.append(sign)
            
            return signs
        except Exception as e:
            print(f"Error parsing XML file {xml_file_path}: {e}")
            return []

    def _extract_location_from_xml(self, root):
        """Extract GPS coordinates from XML annotation
        Adjust this method based on your XML structure
        """
        try:
            # This is a placeholder implementation - replace with your actual XML structure
            # Check if location info is in the XML
            lat = root.find('.//latitude')
            lon = root.find('.//longitude')
            
            if lat is not None and lon is not None:
                return {
                    "lat": float(lat.text),
                    "lon": float(lon.text)
                }
            
            # If not in XML, try to extract from filename or use default
            return {
                "lat": 40.7128,  # New York City default coordinates
                "lon": -74.0060  # Modify as needed
            }
        except Exception as e:
            print(f"Error extracting location data: {e}")
            return {"lat": 40.7128, "lon": -74.0060}
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the distance between two coordinate points in kilometers"""
        import math
        
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        radius = 6371  # Radius of the Earth in kilometers
        
        return radius * c
    
    def process_road_sign_data(self, raw_data):
        """Process raw road sign data into a format suitable for our model"""
        if not raw_data or "signs" not in raw_data:
            return None
        
        processed_data = []
        for sign in raw_data["signs"]:
            sign_data = {
                "sign_id": sign.get("id"),
                "sign_type": sign.get("type"),
                "latitude": sign.get("location", {}).get("lat"),
                "longitude": sign.get("location", {}).get("lon"),
                "road_segment": sign.get("road_segment"),
                "image_path": sign.get("image_path"),
                "detected_at": sign.get("timestamp"),
                "confidence": sign.get("confidence", 0.0),
                "signalization_density": self._calculate_signalization_density(sign)
            }
            processed_data.append(sign_data)
        
        return pd.DataFrame(processed_data)
    
    def _calculate_signalization_density(self, sign_data):
        """Calculate signalization density based on sign data"""
        if "nearby_signs" in sign_data and "segment_length" in sign_data:
            return len(sign_data["nearby_signs"]) / sign_data["segment_length"]
        return 0.3  # Default value
    
    def integrate_with_traffic_network(self, traffic_graph, sign_data_df):
        """Update traffic network with road sign data"""
        if sign_data_df is None or sign_data_df.empty:
            return
        
        for _, row in sign_data_df.iterrows():
            # Find the nearest node in the traffic graph
            nearest_node = traffic_graph.find_nearest_node(row["latitude"], row["longitude"])
            
            if nearest_node:
                # Update node attributes with sign data
                traffic_graph.update_node_attribute(
                    nearest_node, 
                    "signalization_density", 
                    row["signalization_density"]
                )
                
                # Update edges connected to this node
                for neighbor in traffic_graph.graph.neighbors(nearest_node):
                    if row["sign_type"] == "speed_limit":
                        traffic_graph.update_edge_attribute(
                            nearest_node, neighbor, "speed_limit", row.get("value", 50)
                        )
                    
                    # Add signalization factor to edge weight calculation
                    traffic_graph.update_edge_weight(
                        nearest_node, neighbor, 
                        weight_modifier=1 + (row["signalization_density"] * 0.2)
                    )
    
    def get_all_sign_data(self):
        """Get all road sign data from the XML annotations"""
        raw_data = self.get_road_sign_data(0, 0, radius=999999)  # Large radius to get all signs
        return self.process_road_sign_data(raw_data)


road_sign_integrator = RoadSignDataIntegrator(
    annotations_dir="E:/traffic_csv/annotations",
    images_dir="E:/traffic_csv/images"
)

# Modified endpoint to update traffic data from XML annotations
@app.get("/update-traffic-data")
async def update_traffic_data(
    current_user: User = Depends(get_admin_user)
):
    """Update traffic data from road sign XML annotations"""
    updated_nodes = 0
    
    try:
        # Get data for different regions of the traffic network
        for region in ["downtown", "uptown", "midtown"]:
            coords = {
                "downtown": (40.7128, -74.0060),
                "uptown": (40.8075, -73.9626),
                "midtown": (40.7549, -73.9840)
            }.get(region)
            
            if coords:
                # Get road sign data from XML annotations
                raw_data = road_sign_integrator.get_road_sign_data(coords[0], coords[1], radius=3.0)
                processed_data = road_sign_integrator.process_road_sign_data(raw_data)
                
                # Update traffic network
                if processed_data is not None:
                    road_sign_integrator.integrate_with_traffic_network(traffic_network, processed_data)
                    updated_nodes += len(processed_data)
        
        return {
            "status": "success",
            "updated_nodes": updated_nodes,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating traffic data: {str(e)}")

# New endpoint to list all road signs
@app.get("/road-signs")
async def list_road_signs():
    """Get a list of all road signs from the annotations"""
    try:
        # Get all sign data
        signs_df = road_sign_integrator.get_all_sign_data()
        
        if signs_df is None or signs_df.empty:
            return {"signs": []}
        
        # Convert to dictionary format
        signs_list = signs_df.to_dict(orient="records")
        
        return {"signs": signs_list}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving road sign data: {str(e)}")

# Endpoint to get a specific road sign by ID
@app.get("/road-sign/{sign_id}")
async def get_road_sign(sign_id: str):
    """Get details of a specific road sign"""
    try:
        # Get all sign data
        signs_df = road_sign_integrator.get_all_sign_data()
        
        if signs_df is None or signs_df.empty:
            raise HTTPException(status_code=404, detail=f"No road signs found")
        
        # Filter by sign ID
        sign_data = signs_df[signs_df["sign_id"] == sign_id]
        
        if sign_data.empty:
            raise HTTPException(status_code=404, detail=f"Road sign with ID {sign_id} not found")
        
        # Return the first matching sign
        return sign_data.iloc[0].to_dict()
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error retrieving road sign: {str(e)}")

@app.get("/traffic-summary")
async def get_traffic_summary(current_user: User = Depends(get_current_active_user)):
    """Get a summary of current traffic conditions across the network"""
    try:
        # Generate network-wide statistics
        congestion_values = []
        speed_values = []
        total_vehicles = 0
        
        # Sample a few nodes for data
        sampled_nodes = np.random.choice(list(traffic_network.graph.nodes()), 
                                         min(5, len(traffic_network.graph.nodes())),
                                         replace=False)
        
        # Collect predictions for sampled nodes
        for node_id in sampled_nodes:
            node_data = traffic_network.graph.nodes[node_id]
            
            # Skip nodes without coords
            if 'latitude' not in node_data or 'longitude' not in node_data:
                continue
                
            # Generate a traffic prediction for this node
            request_data = TrafficRequest(
                vehicle_count=np.random.randint(50, 150),
                weather_condition=0.3,
                latitude=node_data.get('latitude', 40.7),
                longitude=node_data.get('longitude', -74.0),
                time_of_day=datetime.now().hour,
                day_of_week=datetime.now().weekday()
            )
            
            prediction = await predict_traffic(request_data, current_user)
            
            # Store values
            congestion_values.append(prediction.prediction)
            speed_values.append(50 * (1 - prediction.prediction * 0.7))
            total_vehicles += np.random.randint(50, 150)
        
        # Calculate averages
        avg_congestion = np.mean(congestion_values) if congestion_values else 0.5
        avg_speed = np.mean(speed_values) if speed_values else 30
        
        # Get eigenvalues of the network
        eigenvalues = traffic_network.compute_eigenvalues()
        
        # Return summary
        return {
            "timestamp": datetime.now().isoformat(),
            "average_congestion": float(avg_congestion),
            "average_speed": float(avg_speed),
            "total_vehicles_estimated": int(total_vehicles),
            "network_density": float(nx.density(traffic_network.graph)),
            "spectral_radius": float(max(abs(eigenvalues))),
            "congestion_hotspots": int(sum(1 for c in congestion_values if c > 0.7)),
            "network_status": "congested" if avg_congestion > 0.6 else "normal" if avg_congestion > 0.3 else "clear"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating traffic summary: {str(e)}")

# Admin endpoints
@app.get("/admin/users", response_model=List[UserResponse])
async def get_all_users(
    skip: int = 0, 
    limit: int = 100, 
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.delete("/admin/users/{user_id}", response_model=UserResponse)
async def delete_user(
    user_id: int, 
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    return user

@app.put("/admin/users/{user_id}/admin", response_model=UserResponse)
async def set_admin_status(
    user_id: int,
    is_admin: bool,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_admin = is_admin
    db.commit()
    db.refresh(user)
    return user
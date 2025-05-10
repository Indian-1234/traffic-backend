# from pydantic import BaseModel, Field
# from typing import Any, Dict, List, Union

# class TrafficRequest(BaseModel):
#     vehicle_count: int = Field(..., description="Number of vehicles in the area")
#     weather_condition: float = Field(..., description="Weather impact factor (0-1)")
#     latitude: float = Field(..., description="Latitude coordinate")
#     longitude: float = Field(..., description="Longitude coordinate")
#     time_of_day: int = Field(..., description="Hour of day (0-23)")
#     day_of_week: int = Field(..., description="Day of week (0-6, 0=Monday)")

# class TrafficPrediction(BaseModel):
#     prediction: float = Field(..., description="Traffic congestion prediction (0-1)")
#     flow_rate: float = Field(..., description="Current traffic flow rate")
#     signalization_density: float = Field(..., description="Density of traffic signals")
#     recommended_speed: float = Field(..., description="Recommended speed (km/h)")
#     estimated_travel_time: float = Field(..., description="Estimated travel time (minutes)")
#     confidence: float = Field(..., description="Prediction confidence score (0-1)")

# class OptimizationResult(BaseModel):
#     optimal_routes: List[Dict[str, Union[str, float]]]
#     traffic_reduction: float
#     average_time_saved: float

# class LiveTrafficData(BaseModel):
#     timestamp: str = Field(..., description="Current timestamp")
#     location: Dict[str, float] = Field(..., description="Coordinates of the measurement")
#     current_congestion: float = Field(..., description="Current congestion level (0-1)")
#     vehicle_count: int = Field(..., description="Number of vehicles detected")
#     average_speed: float = Field(..., description="Average vehicle speed (km/h)")
#     traffic_trend: str = Field(..., description="Traffic trend (increasing/decreasing/stable)")
#     road_segments: List[Dict[str, Any]] = Field(..., description="Data for individual road segments")

from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any

# Traffic prediction models
class TrafficRequest(BaseModel):
    vehicle_count: int
    weather_condition: float  # 0-1 where 0 is clear and 1 is severe
    latitude: float
    longitude: float
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # Day of week (0-6, where 0 is Monday)

class TrafficPrediction(BaseModel):
    prediction: float  # Congestion prediction (0-1)
    flow_rate: float  # Current traffic flow rate
    signalization_density: float  # Density of traffic signals
    recommended_speed: float  # Recommended speed (km/h)
    estimated_travel_time: float  # Estimated travel time (minutes)
    confidence: float  # Confidence in prediction (0-1)

class OptimizationResult(BaseModel):
    optimal_routes: List[Dict[str, Any]]
    traffic_reduction: float
    average_time_saved: float

class LiveTrafficData(BaseModel):
    timestamp: str
    location: Dict[str, float]
    current_congestion: float
    vehicle_count: int
    average_speed: float
    traffic_trend: str
    road_segments: List[Dict[str, Any]]

# Auth models
class UserBase(BaseModel):
    username: str
    email: str
    name: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    
    class Config:
        orm_mode = True
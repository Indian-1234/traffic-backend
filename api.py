from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import numpy as np
import requests
from datetime import datetime
import networkx as nx
import os

from models import TrafficRequest, TrafficPrediction, OptimizationResult, LiveTrafficData
from quantum_config import QuantumTrafficPredictor, generate_training_data
from traffic_graph import TrafficGraph

# Import PostgreSQL auth module
from postgresql_auth import setup_auth_routes, get_db, User, Base, engine

# Configuration Settings
TOMTOM_API_KEY = "IV7dQDp5vey54vgGvRlIDmn7qazKzAaN"  # Replace with your actual API key
TOMTOM_TRAFFIC_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
MODEL_SAVE_PATH = "quantum_traffic_model.json"

# Initialize the FastAPI app    
app = FastAPI(
    title="Quantum Traffic Optimization API",
    description="API for traffic prediction and optimization using quantum computing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model = QuantumTrafficPredictor(n_qubits=4)

# Initialize the traffic graph
traffic_network = TrafficGraph()

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

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    # Ensure PostgreSQL tables are created
    Base.metadata.create_all(bind=engine)
    
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

@app.post("/predict", response_model=TrafficPrediction)
async def predict_traffic(data: TrafficRequest):
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
    latitude: float = Query(..., description="Latitude of user/device"),
    longitude: float = Query(..., description="Longitude of user/device")
):
    """Optimize route between two points using quantum optimization"""
    try:
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
                vehicle_count=100,
                weather_condition=0.2,
                latitude=latitude,
                longitude=longitude,
                time_of_day=departure_time,
                day_of_week=datetime.now().weekday()
            )
            
            # Get prediction
            prediction = await predict_traffic(request_data)
            
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
    lon: Optional[float] = Query(None, description="Longitude for center point")
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
                
                prediction = await predict_traffic(request_data)
                
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

@app.get("/traffic-summary")
async def get_traffic_summary():
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
            
            prediction = await predict_traffic(request_data)
            
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

# Set up authentication routes
setup_auth_routes(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
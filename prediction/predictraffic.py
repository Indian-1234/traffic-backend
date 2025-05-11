import numpy as np
import boto3
import requests
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

# FastAPI app
app = FastAPI(title="Quantum Traffic Optimization API")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# TomTom API configuration
TOMTOM_API_KEY = "IV7dQDp5vey54vgGvRlIDmn7qazKzAaN"
TOMTOM_TRAFFIC_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# Rest of your code remains unchanged

# AWS Braket configuration
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
S3_FOLDER = ("amazon-braket-quantiumhitter", "quantum-results")

# Data models for the API
class TrafficRequest(BaseModel):
    vehicle_count: int
    weather_condition: float  # 0 to 1, where 0 is clear and 1 is severe
    latitude: float
    longitude: float
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # 0-6 (Monday to Sunday)

class TrafficPrediction(BaseModel):
    prediction: float
    confidence: float
    status: str

class RouteSegment(BaseModel):
    from_node: str
    to_node: str
    distance_km: float
    estimated_time_min: float
    congestion_factor: float

class OptimizationResult(BaseModel):
    optimal_routes: List[Dict[str, Any]]
    traffic_reduction: float
    average_time_saved: float

def get_traffic_data(latitude, longitude, zoom=10):
    """
    Get traffic data from TomTom API for a specific location.
    Returns speed, free flow speed, and confidence values.
    """
    params = {
        "point": f"{latitude},{longitude}",
        "unit": "KMPH",
        "openLr": "false",
        "zoom": zoom,
        "key": TOMTOM_API_KEY
    }
    
    try:
        response = requests.get(TOMTOM_TRAFFIC_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant traffic data
        flow_data = data.get('flowSegmentData', {})
        current_speed = flow_data.get('currentSpeed', 0)
        free_flow_speed = flow_data.get('freeFlowSpeed', 0)
        confidence = flow_data.get('confidence', 0)
        
        print(f"Retrieved traffic data: Current speed = {current_speed}, Free flow = {free_flow_speed}")
        return current_speed, free_flow_speed, confidence
    
    except Exception as e:
        print(f"Error fetching traffic data: {str(e)}")
        # Return default values in case of error
        return 30, 50, 0.5


def normalize_traffic_data(current_speed, free_flow_speed, confidence):
    """
    Normalize traffic data to values suitable for quantum circuit parameters.
    Returns values between 0 and 2π for angles.
    """
    # Calculate congestion ratio (0 to 1, where 1 means no congestion)
    if free_flow_speed == 0:  # Avoid division by zero
        congestion_ratio = 0.5
    else:
        congestion_ratio = min(current_speed / free_flow_speed, 1.0)
    
    # Convert to angles for quantum gates
    congestion_angle = congestion_ratio * np.pi
    confidence_angle = confidence * np.pi / 2
    
    return congestion_angle, confidence_angle


def create_traffic_prediction_circuit(congestion_angle, confidence_angle, n_qubits=3):
    """
    Create a quantum circuit for traffic prediction based on traffic parameters.
    Uses rotation gates parameterized by traffic data.
    """
    circuit = Circuit()
    
    # Initialize circuit with superposition
    for i in range(n_qubits):
        circuit.h(i)
    
    # Apply rotations based on traffic data
    # First qubit: congestion level
    circuit.rx(0, congestion_angle)
    
    # Second qubit: confidence level
    circuit.ry(1, confidence_angle)
    
    # Create entanglement between qubits
    for i in range(n_qubits-1):
        circuit.cnot(i, i+1)
    
    # Apply final phase based on combination of parameters
    combined_angle = (congestion_angle + confidence_angle) / 2
    circuit.rz(n_qubits-1, combined_angle)
    
    # Apply QFT-inspired operations for the prediction
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            angle = 2 * np.pi / (2 ** (j - i + 1))
            circuit.cphaseshift(control=i, target=j, angle=angle * congestion_angle)
    
    # Measure all qubits
    for i in range(n_qubits):
        circuit.measure(i)
    
    return circuit


def run_circuit_locally(circuit, shots=1000):
    """
    Run the quantum circuit on a local simulator.
    """
    device = LocalSimulator()
    
    print(f"Running traffic prediction circuit with {len(circuit.qubits)} qubits locally...")
    start_time = time.time()
    result = device.run(circuit, shots=shots).result()
    end_time = time.time()
    
    print(f"Circuit execution time: {end_time - start_time:.2f} seconds")
    
    # Get measurement counts
    counts = result.measurement_counts
    
    return counts


def run_circuit_on_braket(circuit, shots=1000, use_simulator=True):
    """
    Run the quantum circuit on AWS Braket.
    By default uses the SV1 simulator, but can be configured to use actual quantum hardware.
    """
    try:
        if use_simulator:
            # Use SV1 simulator
            device = AwsDevice(SV1_ARN)
            device_name = "SV1 simulator"
        else:
            # For demonstration - in reality, you would use a real quantum device ARN
            device = AwsDevice(SV1_ARN)  # Replace with real hardware ARN if available
            device_name = "quantum device"
        
        print(f"Running traffic prediction circuit with {len(circuit.qubits)} qubits on {device_name}...")
        start_time = time.time()
        task = device.run(circuit, S3_FOLDER, shots=shots)
        print(f"Task ARN: {task.id}")
        
        print("Waiting for results...")
        result = task.result()
        end_time = time.time()
        
        print(f"Circuit execution time: {end_time - start_time:.2f} seconds")
        
        # Get measurement counts
        counts = result.measurement_counts
        
        return counts
    
    except Exception as e:
        print(f"Error running circuit on AWS Braket: {str(e)}")
        print("Falling back to local simulator...")
        return run_circuit_locally(circuit, shots)


def interpret_traffic_prediction(counts, n_qubits=3):
    """
    Interpret the quantum measurement results as traffic predictions.
    Returns predicted congestion level and confidence.
    """
    if not counts:
        return {"congestion": 0.5, "confidence": 0.5, "status": "MODERATE"}
    
    # Find the most probable state
    most_probable = max(counts, key=counts.get)
    probability = counts[most_probable] / sum(counts.values())
    
    # Convert binary string to integer
    state_int = int(most_probable, 2)
    
    # Calculate normalized congestion value (0 to 1)
    max_state = 2**n_qubits - 1
    congestion = state_int / max_state
    
    # Calculate prediction confidence based on probability distribution
    # Higher probability of a single outcome indicates higher confidence
    confidence = probability
    
    # Generate status
    if congestion < 0.3:
        status = "LIGHT"
    elif congestion < 0.7:
        status = "MODERATE"
    else:
        status = "HEAVY"
    
    return {
        "prediction": congestion,
        "confidence": confidence,
        "status": status
    }

async def predict_traffic(request: TrafficRequest) -> TrafficPrediction:
    """
    Predict traffic using quantum computing.
    Takes traffic data and uses a quantum circuit to predict congestion.
    """
    # Extract parameters from request
    time_factor = request.time_of_day / 24.0  # Normalize time to 0-1
    day_factor = request.day_of_week / 6.0    # Normalize day to 0-1
    
    # Calculate base congestion from factors
    base_congestion = (
        0.3 +  # Base level
        0.2 * request.vehicle_count / 200 +  # Vehicle density impact
        0.1 * request.weather_condition +    # Weather impact
        0.2 * time_factor                    # Time of day impact
    )
    
    # Clamp congestion to 0-1 range
    base_congestion = min(max(base_congestion, 0.0), 1.0)
    
    # Set angles for quantum circuit
    congestion_angle = base_congestion * np.pi
    confidence_angle = 0.8 * np.pi / 2  # Fixed confidence for now
    
    # Create quantum circuit for prediction
    circuit = create_traffic_prediction_circuit(
        congestion_angle, confidence_angle, n_qubits=3
    )
    
    # Run circuit locally
    counts = run_circuit_locally(circuit, shots=1000)
    
    # Interpret results
    result = interpret_traffic_prediction(counts)
    
    # Convert to response model
    return TrafficPrediction(
        prediction=result["prediction"],
        confidence=result["confidence"],
        status=result["status"]
    )



# Create a simple graph representation of the traffic network
class TrafficNetwork:
    def __init__(self):
        # Initialize a simple graph with some nodes and edges
        self.graph = {}
        self.nodes = {}
        
        # Add some sample nodes (would be loaded from a real map in production)
        for i in range(1, 11):
            # Create nodes in a grid pattern
            lat = 40.7 + (i % 4) * 0.01
            lon = -74.0 + (i // 4) * 0.01
            self.nodes[i] = {"id": i, "latitude": lat, "longitude": lon}
        
        # Add edges between nodes
        self._create_edges()
    
    def _create_edges(self):
        # Create some simple edges between nodes
        edges = [
            (1, 2, 0.5, 50), (2, 1, 0.5, 50),
            (2, 3, 0.7, 60), (3, 2, 0.7, 60),
            (3, 4, 0.6, 40), (4, 3, 0.6, 40),
            (4, 5, 0.8, 70), (5, 4, 0.8, 70),
            (5, 6, 0.5, 30), (6, 5, 0.5, 30),
            (6, 7, 0.9, 60), (7, 6, 0.9, 60),
            (7, 8, 0.4, 50), (8, 7, 0.4, 50),
            (8, 9, 0.6, 40), (9, 8, 0.6, 40),
            (9, 10, 0.7, 60), (10, 9, 0.7, 60),
            (1, 5, 1.2, 70), (5, 1, 1.2, 70),
            (2, 6, 1.1, 60), (6, 2, 1.1, 60),
            (3, 7, 1.3, 50), (7, 3, 1.3, 50),
            (4, 8, 1.0, 40), (8, 4, 1.0, 40)
        ]
        
        # Add edges to the graph
        for u, v, dist, speed in edges:
            if u not in self.graph:
                self.graph[u] = {}
            
            self.graph[u][v] = {
                "distance": dist,  # in km
                "speed_limit": speed,  # in km/h
                "weight": 1.0  # default weight (will be updated with traffic)
            }
    
    def extract_subgraph(self, start_node, radius=3):
        """Extract a subgraph centered at start_node with given radius"""
        if start_node not in self.nodes:
            raise Exception(f"Node {start_node} not found")
        
        # Use a simple BFS to extract nodes within radius
        subgraph = {}
        visited = set()
        queue = [(start_node, 0)]  # (node, distance)
        
        while queue:
            node, dist = queue.pop(0)
            
            if node in visited:
                continue
                
            visited.add(node)
            
            if node not in subgraph:
                subgraph[node] = {}
            
            # Add all neighbors within radius
            if dist < radius and node in self.graph:
                for neighbor, edge_data in self.graph[node].items():
                    if neighbor not in subgraph:
                        subgraph[neighbor] = {}
                    
                    # Copy edge data
                    subgraph[node][neighbor] = edge_data.copy()
                    
                    # Add neighbor to queue
                    queue.append((neighbor, dist + 1))
        
        return subgraph
    
    def update_edge_weight(self, u, v, weight):
        """Update the weight of an edge in the graph"""
        if u in self.graph and v in self.graph[u]:
            self.graph[u][v]["weight"] = weight
    
    def compute_shortest_path(self, start, end):
        """Compute shortest path using Dijkstra's algorithm"""
        if start not in self.graph or end not in self.graph:
            return [], 0
        
        # Initialize
        distances = {node: float('infinity') for node in self.graph}
        distances[start] = 0
        previous = {node: None for node in self.graph}
        unvisited = list(self.graph.keys())
        
        while unvisited:
            # Find node with minimum distance
            current = min(unvisited, key=lambda node: distances[node])
            
            # If we reached the end or no path exists
            if current == end or distances[current] == float('infinity'):
                break
                
            unvisited.remove(current)
            
            # Update distances to neighbors
            for neighbor, edge_data in self.graph[current].items():
                weight = edge_data["weight"]
                distance = distances[current] + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
        
        # Reconstruct the path
        path = []
        current = end
        
        while current:
            path.append(current)
            current = previous[current]
            
        # Reverse the path
        path = path[::-1]
        
        # Return empty path if no path exists
        if not path or path[0] != start:
            return [], 0
            
        return path, distances[end]

# Initialize the traffic network
traffic_network = TrafficNetwork()


# API endpoints
@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"status": "online", "message": "Quantum Traffic Optimization API"}

@app.post("/predict-traffic", response_model=TrafficPrediction)
async def predict_traffic_endpoint(request: TrafficRequest):
    """
    Predict traffic using quantum computing
    """
    try:
        prediction = await predict_traffic(request)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Modified optimize_route endpoint to handle non-existent nodes better
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
        # Check if nodes exist in the network
        if start_node not in traffic_network.nodes:
            # If start node doesn't exist, find closest available node
            closest_node = find_closest_node(start_node, traffic_network.nodes)
            start_node = closest_node
            print(f"Start node {start_node} not found, using closest node {closest_node} instead")
        
        if end_node not in traffic_network.nodes:
            # If end node doesn't exist, find closest available node
            closest_node = find_closest_node(end_node, traffic_network.nodes)
            end_node = closest_node
            print(f"End node {end_node} not found, using closest node {closest_node} instead")
            
        # Extract subgraph for optimization
        try:
            subgraph = traffic_network.extract_subgraph(start_node, radius=5)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to extract subgraph: {str(e)}")
            
        # Update edge weights based on predicted traffic
        for u, v in [(u, v) for u in subgraph for v in subgraph[u]]:
            # Get node coordinates
            u_lat = traffic_network.nodes[u].get("latitude", 40.7)
            u_lon = traffic_network.nodes[u].get("longitude", -74.0)
            
            # Create a request for this segment
            request_data = TrafficRequest(
                vehicle_count=100,
                weather_condition=0.2,
                latitude=latitude,
                longitude=longitude,
                time_of_day=departure_time,
                day_of_week=datetime.now().weekday()
            )
            
            # Get prediction using quantum circuit
            prediction = await predict_traffic(request_data)
            
            # Update edge weight based on congestion
            congestion_factor = 1 + prediction.prediction * 2  # Scale congestion impact
            traffic_network.update_edge_weight(u, v, congestion_factor)
        
        # Find optimal path
        path, distance = traffic_network.compute_shortest_path(start_node, end_node)
        
        if not path:
            raise HTTPException(status_code=404, detail="No path found between the specified nodes")
            
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

def find_closest_node(requested_id, available_nodes):
    """Find the closest node to the requested ID from available nodes"""
    available_ids = list(available_nodes.keys())
    if not available_ids:
        return 1  # Default to node 1 if no nodes available
    
    # Simple approach: find the node with the closest ID number
    return min(available_ids, key=lambda x: abs(x - requested_id))

# Additionally, let's add more nodes to our network
def expand_traffic_network():
    """Expand the traffic network with more nodes"""
    # Add more nodes to the network (up to node 20)
    for i in range(11, 21):
        lat = 40.7 + (i % 5) * 0.01
        lon = -74.0 + (i // 5) * 0.01
        traffic_network.nodes[i] = {"id": i, "latitude": lat, "longitude": lon}
    
    # Add edges between new nodes
    new_edges = [
        (10, 11, 0.6, 55), (11, 10, 0.6, 55),
        (11, 12, 0.7, 50), (12, 11, 0.7, 50),
        (12, 13, 0.5, 60), (13, 12, 0.5, 60),
        (13, 14, 0.8, 45), (14, 13, 0.8, 45),
        (14, 15, 0.6, 50), (15, 14, 0.6, 50),
        (15, 16, 0.9, 40), (16, 15, 0.9, 40),
        (16, 17, 0.7, 55), (17, 16, 0.7, 55),
        (17, 18, 0.5, 60), (18, 17, 0.5, 60),
        (18, 19, 0.8, 45), (19, 18, 0.8, 45),
        (19, 20, 0.6, 50), (20, 19, 0.6, 50),
        # Connect new nodes to existing network
        (5, 11, 1.0, 55), (11, 5, 1.0, 55),
        (7, 13, 1.1, 50), (13, 7, 1.1, 50),
        (9, 15, 1.2, 45), (15, 9, 1.2, 45),
        (10, 17, 1.0, 60), (17, 10, 1.0, 60)
    ]
    
    # Add edges to the graph
    for u, v, dist, speed in new_edges:
        if u not in traffic_network.graph:
            traffic_network.graph[u] = {}
        
        traffic_network.graph[u][v] = {
            "distance": dist,  # in km
            "speed_limit": speed,  # in km/h
            "weight": 1.0  # default weight (will be updated with traffic)
        }

# Call this function after traffic_network is initialized
expand_traffic_network()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
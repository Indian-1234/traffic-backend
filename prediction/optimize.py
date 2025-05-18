# Add boto3 to imports at the top of your file
import numpy as np
import time
import requests
import heapq
import folium
import boto3  # Add this import
import json   # Add this import
import io     # Add this import
from datetime import datetime
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice  # Added for AWS Braket
import os
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
# AWS Braket configuration
# AWS Braket configuration
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
S3_FOLDER = ("amazon-braket-quantiumhitter", "quantum-results")
AWS_REGION = "us-east-1"  # Add this line

# Import functions from predict.py
from predict import (
    get_traffic_data,
    normalize_traffic_data,
    create_traffic_prediction_circuit,
    run_circuit_locally,
    interpret_traffic_prediction
)

# TomTom API configuration
TOMTOM_API_KEY = "IV7dQDp5vey54vgGvRlIDmn7qazKzAaN"
TOMTOM_ROUTING_URL = "https://api.tomtom.com/routing/1/calculateRoute"


class Node:
    """Represents a geographical node in the route network"""
    def __init__(self, name, lat, lon):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.traffic_prediction = None  # Will store predicted traffic data

    def __str__(self):
        return f"{self.name} ({self.lat}, {self.lon})"


class Edge:
    """Represents a road segment between two nodes"""
    def __init__(self, start_node, end_node, distance=None):
        self.start_node = start_node
        self.end_node = end_node
        self.distance = distance  # in km
        self.congestion_factor = None  # Will be updated based on quantum prediction
        
    def calculate_distance(self):
        """Calculate distance between two nodes using Haversine formula"""
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1 = np.radians(self.start_node.lat), np.radians(self.start_node.lon)
        lat2, lon2 = np.radians(self.end_node.lat), np.radians(self.end_node.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        self.distance = distance
        return distance
    
    def get_weighted_distance(self):
        """Get distance weighted by congestion factor"""
        if self.congestion_factor is None:
            return self.distance * 1.5  # Default penalty if no prediction available
        
        # More congestion = higher weight = less favorable route
        return self.distance * (1 + self.congestion_factor)


def run_circuit_on_sv1(circuit):
    """
    Run a quantum circuit on AWS SV1 simulator
    """
    print("Running circuit on AWS SV1 quantum simulator...")
    
    try:
        # Method 1: Try with explicit region in environment
        import os
        os.environ['AWS_DEFAULT_REGION'] = AWS_REGION
        
        # Simple device initialization
        device = AwsDevice(SV1_ARN)
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
        try:
            # Method 2: Try with AwsSession
            from braket.aws import AwsSession
            aws_session = AwsSession()
            device = AwsDevice(SV1_ARN, aws_session=aws_session)
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            # Method 3: Force region in boto session
            from braket.aws import AwsSession
            boto_session = boto3.Session()
            boto_session.region_name = AWS_REGION  # Force set region
            aws_session = AwsSession(boto_session=boto_session)
            device = AwsDevice(SV1_ARN, aws_session=aws_session)
    
    # Run the circuit
    task = device.run(
        circuit,
        s3_destination_folder=S3_FOLDER,
        shots=1000
    )
    
    print(f"Task ARN: {task.id}")
    print("Waiting for quantum task to complete...")
    
    result = task.result()
    counts = result.measurement_counts
    
    print(f"Circuit execution completed. Result: {counts}")
    return counts

def predict_traffic_for_node(node, use_aws=True):
    """
    Use the quantum traffic prediction model to predict traffic at a node
    
    Args:
        node: The node to predict traffic for
        use_aws: If True, use AWS SV1 simulator; otherwise use local simulator
    """
    print(f"Predicting traffic for {node.name}...")
    
    # Get traffic data from TomTom API
    current_speed, free_flow_speed, confidence = get_traffic_data(
        node.lat, node.lon
    )
    
    # Normalize traffic data for quantum circuit
    congestion_angle, confidence_angle = normalize_traffic_data(
        current_speed, free_flow_speed, confidence
    )
    
    # Create quantum circuit for traffic prediction
    circuit = create_traffic_prediction_circuit(
        congestion_angle, confidence_angle, n_qubits=3
    )
    
    # Run circuit on AWS SV1 or locally
    if use_aws:
        counts = run_circuit_on_sv1(circuit)
    else:
        counts = run_circuit_locally(circuit)
    
    # Interpret the results
    node.traffic_prediction = interpret_traffic_prediction(counts, n_qubits=3)
    
    print(f"Traffic at {node.name}: {node.traffic_prediction['prediction']} " + 
          f"(Congestion level: {node.traffic_prediction['congestion']:.2f})")
    
    return node.traffic_prediction


def get_route_from_tomtom(start_lat, start_lon, end_lat, end_lon):
    """
    Get route information from TomTom Routing API
    """
    url = f"{TOMTOM_ROUTING_URL}/{start_lat},{start_lon}:{end_lat},{end_lon}/json"
    
    params = {
        "key": TOMTOM_API_KEY,
        "traffic": "true",
        "travelMode": "car"
    }
    
    try:
        print("Fetching route from TomTom API...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        route_data = response.json()
        
        # Extract route information
        route = route_data.get('routes', [{}])[0]
        summary = route.get('summary', {})
        
        distance = summary.get('lengthInMeters', 0) / 1000  # Convert to km
        travel_time = summary.get('travelTimeInSeconds', 0) / 60  # Convert to minutes
        
        return {
            "distance": distance,
            "time": travel_time,
            "route": route
        }
    except Exception as e:
        print(f"Error fetching route from TomTom: {str(e)}")
        return None


def create_network(nodes, max_distance=100):
    """
    Create a network of nodes and edges with increased max_distance for Tamil Nadu
    """
    graph = {}
    edges = []
    
    # Create edges between nodes based on proximity
    for i, node1 in enumerate(nodes):
        graph[node1.name] = []
        
        for j, node2 in enumerate(nodes):
            if i != j:
                edge = Edge(node1, node2)
                distance = edge.calculate_distance()
                
                # Creating connections with increased distance threshold for Tamil Nadu locations
                is_tamil_nadu_node1 = node1.lat > 8.0 and node1.lat < 14.0 and node1.lon > 76.0 and node1.lon < 81.0
                is_tamil_nadu_node2 = node2.lat > 8.0 and node2.lat < 14.0 and node2.lon > 76.0 and node2.lon < 81.0
                
                # Use larger distance threshold for Tamil Nadu locations
                distance_threshold = max_distance if (is_tamil_nadu_node1 and is_tamil_nadu_node2) else 30
                
                if distance < distance_threshold:
                    graph[node1.name].append((node2.name, edge))
                    edges.append(edge)
    
    return graph, edges


def predict_traffic_for_network(graph, edges, use_aws=True):
    """
    Predict traffic for all nodes in the network and update edge congestion factors
    
    Args:
        graph: The network graph
        edges: List of edges in the network
        use_aws: If True, use AWS SV1 simulator; otherwise use local simulator
    """
    print("Predicting traffic conditions for the network...")
    
    # Get all unique nodes from the graph
    nodes = set()
    for source in graph:
        for destination, edge in graph[source]:
            nodes.add(edge.start_node)
            nodes.add(edge.end_node)
    
    # Predict traffic for each node
    for node in nodes:
        if node.traffic_prediction is None:
            predict_traffic_for_node(node, use_aws=use_aws)
    
    # Update congestion factors for edges
    for edge in edges:
        # Average congestion between start and end nodes
        start_congestion = edge.start_node.traffic_prediction['congestion']
        end_congestion = edge.end_node.traffic_prediction['congestion']
        edge.congestion_factor = (start_congestion + end_congestion) / 2
        
        print(f"Edge {edge.start_node.name} -> {edge.end_node.name}: " +
              f"Distance: {edge.distance:.2f}km, Congestion: {edge.congestion_factor:.2f}")


def dijkstra(graph, start_node_name, end_node_name):
    """
    Dijkstra's algorithm to find the shortest path considering traffic
    """
    if start_node_name not in graph or end_node_name not in graph:
        print(f"Start node '{start_node_name}' or end node '{end_node_name}' not found in graph")
        return None, None
    
    # Distance from source to all nodes
    distances = {node: float('inf') for node in graph}
    distances[start_node_name] = 0
    
    # Track previous node in optimal path
    previous = {node: None for node in graph}
    
    # Priority queue for nodes to visit
    priority_queue = [(0, start_node_name)]
    
    # Track visited nodes
    visited = set()
    
    while priority_queue and len(visited) < len(graph):
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue
        
        if current_node == end_node_name:
            break
            
        visited.add(current_node)
        
        for neighbor, edge in graph[current_node]:
            if neighbor in visited:
                continue
                
            weight = edge.get_weighted_distance()
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = (current_node, edge)
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # Check if end node was reached
    if distances[end_node_name] == float('inf'):
        print(f"No path exists from {start_node_name} to {end_node_name}")
        return None, None
    
    # Reconstruct the path
    path = []
    current = end_node_name
    total_distance = 0
    total_time = 0  # Estimated time in minutes
    
    while current != start_node_name:
        if previous[current] is None:
            return None, None  # No path exists
            
        prev_node, edge = previous[current]
        path.append((prev_node, current, edge))
        
        # Calculate time based on distance and congestion
        # Assume average speed of 60km/h (1km/min) adjusted for congestion
        speed_factor = 1.0 / (1 + edge.congestion_factor)  # Reduce speed with congestion
        segment_time = edge.distance / speed_factor
        
        total_distance += edge.distance
        total_time += segment_time
        
        current = prev_node
    
    path.reverse()
    
    return path, {
        "distance": total_distance,
        "time": total_time
    }


def visualize_route(nodes, optimal_path, map_file="optimal_route.html"):
    """
    Create an interactive map visualization of the optimal route
    Returns the map object and also saves it to a file
    """
    # Find center point for the map
    lats = [node.lat for node in nodes]
    lons = [node.lon for node in nodes]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create map
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add all nodes to the map
    for node in nodes:
        # Determine color based on congestion
        if node.traffic_prediction:
            congestion = node.traffic_prediction['congestion']
            if congestion < 0.3:
                color = 'green'
            elif congestion < 0.7:
                color = 'orange'
            else:
                color = 'red'
        else:
            color = 'blue'
        
        # Fix the format specifier issue by properly handling None values
        if node.traffic_prediction:
            congestion_text = f"{node.traffic_prediction['congestion']:.2f}"
        else:
            congestion_text = "Unknown"
            
        folium.CircleMarker(
            location=[node.lat, node.lon],
            radius=6,
            popup=f"{node.name}<br>Congestion: {congestion_text}",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(route_map)
    
    # Add route path
    route_points = []
    
    for start, end, edge in optimal_path:
        # Get start and end node objects from their names
        start_node = None
        end_node = None
        for node in nodes:
            if node.name == start:
                start_node = node
            if node.name == end:
                end_node = node
        
        if start_node and end_node:
            route_points.extend([[start_node.lat, start_node.lon], [end_node.lat, end_node.lon]])
            
            # Determine color based on congestion
            congestion = edge.congestion_factor
            if congestion < 0.3:
                color = 'green'
            elif congestion < 0.7:
                color = 'orange'
            else:
                color = 'red'
            
            # Add line segment with popup showing details
            folium.PolyLine(
                [[start_node.lat, start_node.lon], [end_node.lat, end_node.lon]],
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"Distance: {edge.distance:.2f}km<br>Congestion: {edge.congestion_factor:.2f}"
            ).add_to(route_map)
    
    # Save the map
    route_map.save(map_file)
    print(f"Route map saved as {map_file}")
    
    # Return the map object
    return route_map

def get_specific_tamil_nadu_nodes():
    """
    Create a focused set of Tamil Nadu nodes with coordinates
    """
    return [
        Node("Chennai", 13.0827, 80.2707),
        Node("Coimbatore", 11.0168, 76.9558),
        Node("Madurai", 9.9252, 78.1198),
        Node("Salem", 11.6643, 78.1460),
        Node("Tiruppur", 11.1085, 77.3411),
        Node("Tiruchirappalli", 10.7905, 78.7047),
        Node("Erode", 11.3410, 77.7172),
        Node("Vellore", 12.9165, 79.1325),
        Node("Thanjavur", 10.7867, 79.1378),
        Node("Dindigul", 10.3624, 77.9695),
        Node("Karur", 10.9601, 78.0766),
        Node("Namakkal", 11.2196, 78.1671),
        Node("Tirunelveli", 8.7139, 77.7567),
        Node("Pudukkottai", 10.3813, 78.8214),
        Node("Krishnagiri", 12.5307, 78.2138),
        Node("Tiruvannamalai", 12.2253, 79.0747),
        Node("Dharmapuri", 12.1357, 78.1617),
        Node("Virudhunagar", 9.5851, 77.9580),
        Node("Cuddalore", 11.7447, 79.7683),
        Node("Sivaganga", 9.8474, 78.4836)
    ]

def save_results_to_s3(optimal_path, route_stats, nodes, bucket_name="amazon-braket-quantiumhitter", 
                      prefix="quantum-route-results"):
    """
    Save route optimization results to Amazon S3
    
    Args:
        optimal_path: The calculated optimal path
        route_stats: Statistics about the route (distance, time)
        nodes: List of network nodes
        bucket_name: S3 bucket name
        prefix: Prefix/folder for saving files in S3
    """
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results JSON
        results = {
            "timestamp": datetime.now().isoformat(),
            "route_stats": {
                "distance_km": round(route_stats['distance'], 2),
                "time_minutes": round(route_stats['time'], 2)
            },
            "path_segments": []
        }
        
        # Add path details to results
        for start, end, edge in optimal_path:
            # Find start and end node objects
            start_node = next((node for node in nodes if node.name == start), None)
            end_node = next((node for node in nodes if node.name == end), None)
            
            if start_node and end_node:
                segment = {
                    "start": {
                        "name": start,
                        "coordinates": [start_node.lat, start_node.lon]
                    },
                    "end": {
                        "name": end,
                        "coordinates": [end_node.lat, end_node.lon]
                    },
                    "distance_km": round(edge.distance, 2),
                    "congestion_factor": round(edge.congestion_factor, 2)
                }
                results["path_segments"].append(segment)
        
        # Save JSON results
        json_key = f"{prefix}/{timestamp}_route_results.json"
        s3_client.put_object(
            Body=json.dumps(results, indent=2),
            Bucket=bucket_name,
            Key=json_key,
            ContentType='application/json'
        )
        
        # Create and save HTML map if route visualization exists
        if optimal_path:
            # This assumes the visualize_route function has been modified to return the map object
            route_map = create_route_map(nodes, optimal_path)
            
            # Save map to a bytes buffer
            map_data = io.BytesIO()
            route_map.save(map_data, close_file=False)
            map_data.seek(0)
            
            # Upload HTML map to S3
            html_key = f"{prefix}/{timestamp}_route_map.html"
            s3_client.put_object(
                Body=map_data.getvalue(),
                Bucket=bucket_name,
                Key=html_key,
                ContentType='text/html'
            )
            
            print(f"Results saved to S3 bucket '{bucket_name}':")
            print(f"- JSON: s3://{bucket_name}/{json_key}")
            print(f"- Map HTML: s3://{bucket_name}/{html_key}")
            
            # Generate a presigned URL for the HTML map (valid for 1 hour)
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': html_key
                },
                ExpiresIn=3600
            )
            print(f"Map URL (valid for 1 hour): {url}")
            
            return {
                "json_path": f"s3://{bucket_name}/{json_key}",
                "html_path": f"s3://{bucket_name}/{html_key}",
                "presigned_url": url
            }
            
    except Exception as e:
        print(f"Error saving results to S3: {str(e)}")
        return None


def create_route_map(nodes, optimal_path):
    """
    Create a map visualization of the optimal route
    Returns the map object instead of saving it to a file
    """
    # Find center point for the map
    lats = [node.lat for node in nodes]
    lons = [node.lon for node in nodes]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create map
    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add all nodes to the map
    for node in nodes:
        # Determine color based on congestion
        if node.traffic_prediction:
            congestion = node.traffic_prediction['congestion']
            if congestion < 0.3:
                color = 'green'
            elif congestion < 0.7:
                color = 'orange'
            else:
                color = 'red'
        else:
            color = 'blue'
        
        # Handle None values properly
        if node.traffic_prediction:
            congestion_text = f"{node.traffic_prediction['congestion']:.2f}"
        else:
            congestion_text = "Unknown"
            
        folium.CircleMarker(
            location=[node.lat, node.lon],
            radius=6,
            popup=f"{node.name}<br>Congestion: {congestion_text}",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(route_map)
    
    # Add route path
    for start, end, edge in optimal_path:
        # Get start and end node objects from their names
        start_node = next((node for node in nodes if node.name == start), None)
        end_node = next((node for node in nodes if node.name == end), None)
        
        if start_node and end_node:
            # Determine color based on congestion
            congestion = edge.congestion_factor
            if congestion < 0.3:
                color = 'green'
            elif congestion < 0.7:
                color = 'orange'
            else:
                color = 'red'
            
            # Add line segment with popup showing details
            folium.PolyLine(
                [[start_node.lat, start_node.lon], [end_node.lat, end_node.lon]],
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"Distance: {edge.distance:.2f}km<br>Congestion: {edge.congestion_factor:.2f}"
            ).add_to(route_map)
    
    return route_map

def main():
    """
    Main function to run the route optimization
    """
    print("Quantum Route Optimization System")
    print("================================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ask if user wants to use AWS SV1 quantum simulator
    use_aws = input("Use AWS SV1 quantum simulator? (y/n, default: y): ").strip().lower() != 'n'
    if use_aws:
        print("Using AWS SV1 quantum simulator for traffic predictions")
    else:
        print("Using local quantum simulator for traffic predictions")
    
    # Use only Tamil Nadu locations for focused network
    nodes = get_specific_tamil_nadu_nodes()
    
    # Create network graph with increased max_distance for Tamil Nadu nodes
    graph, edges = create_network(nodes, max_distance=100)
    print(f"Created network with {len(nodes)} nodes and {len(edges)} edges")
    
    # Get start and destination from user
    start_node_name = input("Enter starting location (e.g., 'Karur'): ").strip()
    if not start_node_name:
        start_node_name = "Karur"  # Default
        print(f"Using default start location: {start_node_name}")
        
    end_node_name = input("Enter destination (e.g., 'Erode'): ").strip()
    if not end_node_name:
        end_node_name = "Erode"  # Default
        print(f"Using default destination: {end_node_name}")
    
    # Validate input
    valid_start = False
    valid_end = False
    
    for node in nodes:
        if node.name.lower() == start_node_name.lower():
            start_node_name = node.name  # Use exact case from node list
            valid_start = True
        if node.name.lower() == end_node_name.lower():
            end_node_name = node.name  # Use exact case from node list
            valid_end = True
    
    if not valid_start or not valid_end:
        print("Invalid location name(s). Available locations:")
        for node in nodes:
            print(f"- {node.name}")
        return
    
    # Predict traffic for the network using AWS or local simulator
    predict_traffic_for_network(graph, edges, use_aws=use_aws)
    
    # Find optimal route
    print(f"Finding optimal route from {start_node_name} to {end_node_name}...")
    start_time = time.time()
    optimal_path, route_stats = dijkstra(graph, start_node_name, end_node_name)
    end_time = time.time()
    
    if optimal_path is None:
        print(f"No route found from {start_node_name} to {end_node_name}")
        return
    
    print(f"Route optimization completed in {end_time - start_time:.2f} seconds")
    
    # Display route information
    print("\nOptimal Route:")
    print(f"Total distance: {route_stats['distance']:.2f} km")
    print(f"Estimated travel time: {route_stats['time']:.2f} minutes")
    print("\nRoute details:")
    
    for start, end, edge in optimal_path:
        congestion_level = "Low" if edge.congestion_factor < 0.3 else "Medium" if edge.congestion_factor < 0.7 else "High"
        print(f"  {start} → {end}: {edge.distance:.2f} km (Traffic: {congestion_level})")
    
    # Compare with TomTom direct route if possible
    start_node = next((node for node in nodes if node.name == start_node_name), None)
    end_node = next((node for node in nodes if node.name == end_node_name), None)
    
    if start_node and end_node:
        tomtom_route = get_route_from_tomtom(
            start_node.lat, start_node.lon, 
            end_node.lat, end_node.lon
        )
        
        if tomtom_route:
            print("\nComparison with TomTom route:")
            print(f"TomTom distance: {tomtom_route['distance']:.2f} km")
            print(f"TomTom estimated time: {tomtom_route['time']:.2f} minutes")
            
            distance_diff = ((route_stats['distance'] - tomtom_route['distance']) / 
                            tomtom_route['distance'] * 100)
            time_diff = ((route_stats['time'] - tomtom_route['time']) / 
                        tomtom_route['time'] * 100)
            
            print(f"Distance difference: {distance_diff:.1f}%")
            print(f"Time difference: {time_diff:.1f}%")
            
            if route_stats['time'] < tomtom_route['time']:
                print("Our quantum-assisted route is faster than the standard route!")
            else:
                print("The standard route is faster, but our route may avoid unexpected congestion")
    
    # Visualize the route
    visualize_route(nodes, optimal_path)
    
    # Save results to S3 bucket
    print("\nSaving results to S3...")
    s3_results = save_results_to_s3(optimal_path, route_stats, nodes)
    
    if s3_results:
        print("\nResults available at:")
        print(f"JSON: {s3_results['json_path']}")
        print(f"Map: {s3_results['html_path']}")
        print(f"View map: {s3_results['presigned_url']}")
    
    print("\nRoute optimization complete!")
    print("\nRoute optimization complete!")



if __name__ == "__main__":
    main()
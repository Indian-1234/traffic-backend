import requests
import json
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random  # For fallback simulation

class QuantumEnhancedRouter:
    def __init__(self,
                api_key="IV7dQDp5vey54vgGvRlIDmn7qazKzAaN",
                shots=200,
                device_arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
                region="us-west-1",
                s3_bucket="amazon-braket-quantiumhitter"):
        self.api_key = api_key
        self.shots = shots
        self.device_arn = device_arn
        self.region = region
        self.s3_bucket = s3_bucket
        try:
            self.device = AwsDevice(self.device_arn)
        except Exception as e:
            print(f"Warning: Could not initialize quantum device: {e}")
            self.device = None
        self.routes = {}
        
        # TomTom API key - check if it's properly set
        self.tomtom_api_key = None  # Will be set when needed
    
    def set_tomtom_api_key(self, api_key):
        """Set a valid TomTom API key"""
        self.tomtom_api_key = api_key
        return self
    
    def get_optimized_route(self, start_location, end_location, use_fallback=False):
        """
        Get the optimized route between two locations using TomTom API
        and enhance it with quantum computation for traffic prediction
        
        Args:
            start_location (tuple): (latitude, longitude) of start location
            end_location (tuple): (latitude, longitude) of end location
            use_fallback (bool): Whether to use simulated data if API fails
            
        Returns:
            dict: Optimized route information
        """
        if not use_fallback and self.tomtom_api_key:
            try:
                return self._get_route_from_api(start_location, end_location)
            except Exception as e:
                print(f"API Error: {e}")
                print("Falling back to simulated data...")
        
        # Fallback to simulated data for Chennai route
        return self._get_simulated_chennai_route(start_location, end_location)
    
    def _get_route_from_api(self, start_location, end_location):
        """Get route using TomTom API"""
        # TomTom API endpoint for routing
        base_url = "https://api.tomtom.com/routing/1/calculateRoute"
        
        # Construct the request URL
        url = f"{base_url}/{start_location[0]},{start_location[1]}:{end_location[0]},{end_location[1]}/json"
        
        # Request parameters
        params = {
            "key": self.tomtom_api_key,
            "traffic": "true",
            "routeRepresentation": "summaryOnly",
            "computeTravelTimeFor": "all",
            "routeType": "fastest"
        }
        
        # Make the request
        response = requests.get(url, params=params)
        response.raise_for_status()
        route_data = response.json()
        
        # Extract basic route information
        route_summary = route_data["routes"][0]["summary"]
        length_meters = route_summary["lengthInMeters"]
        travel_time_seconds = route_summary["travelTimeInSeconds"]
        traffic_delay_seconds = route_summary.get("trafficDelayInSeconds", 0)
        
        # Apply quantum enhancement for traffic prediction
        if self.device:
            predicted_traffic = self._quantum_traffic_prediction(
                travel_time_seconds, 
                traffic_delay_seconds
            )
            enhanced_time = travel_time_seconds + predicted_traffic
        else:
            enhanced_time = travel_time_seconds + traffic_delay_seconds
            
        # Format the results
        route_info = {
            "distance_km": length_meters / 1000,
            "baseline_time_mins": travel_time_seconds / 60,
            "enhanced_time_mins": enhanced_time / 60,
            "traffic_factor": 1 + (traffic_delay_seconds / travel_time_seconds if travel_time_seconds > 0 else 0),
            "estimated_speed_kmh": (length_meters / 1000) / (enhanced_time / 3600) if enhanced_time > 0 else 0,
            "source": "TomTom API",
            "raw_data": route_data
        }
        
        # Cache the route
        route_key = f"{start_location}_{end_location}"
        self.routes[route_key] = route_info
        
        return route_info
    
    def _get_simulated_chennai_route(self, start_location, end_location):
        """
        Generate simulated route data specifically for Chennai T Nagar to Poonamallee
        based on typical traffic patterns and distance
        """
        # Approximate straight-line distance between T Nagar and Poonamallee is about 15 km
        # But actual road distance is longer, roughly 19-21 km
        
        # Get current hour (0-23) to simulate time-based traffic
        current_hour = datetime.now().hour
        
        # Simulate Chennai traffic patterns based on time of day
        # Morning rush hour: 8-10 AM, Evening rush hour: 5-8 PM
        if current_hour in [8, 9, 10, 17, 18, 19, 20]:
            traffic_multiplier = 1.6 + random.uniform(0, 0.4)  # Heavy traffic
        elif current_hour in [7, 11, 12, 15, 16, 21]:
            traffic_multiplier = 1.3 + random.uniform(0, 0.3)  # Moderate traffic
        else:
            traffic_multiplier = 1.1 + random.uniform(0, 0.2)  # Light traffic
        
        # Basic route information for T Nagar to Poonamallee
        base_distance_km = 20.5 + random.uniform(-1.5, 1.5)  # ~19-22 km
        
        # Average speed without traffic would be around 40-45 km/h in Chennai
        base_speed_kmh = 42.0 + random.uniform(-3, 3)
        
        # Calculate times
        base_time_mins = (base_distance_km / base_speed_kmh) * 60
        actual_time_mins = base_time_mins * traffic_multiplier
        
        # Calculate actual speed with traffic
        actual_speed_kmh = base_distance_km / (actual_time_mins / 60)
        
        # Enhanced time using quantum prediction (simulated)
        if self.device:
            enhanced_time_mins = self._apply_quantum_enhancement(actual_time_mins)
        else:
            # Classical enhancement (slightly more pessimistic)
            enhanced_time_mins = actual_time_mins * (1 + random.uniform(0, 0.15))
        
        # Prepare route information
        route_info = {
            "distance_km": base_distance_km,
            "baseline_time_mins": base_time_mins,
            "enhanced_time_mins": enhanced_time_mins,
            "traffic_factor": traffic_multiplier,
            "estimated_speed_kmh": actual_speed_kmh,
            "source": "Simulated data (T Nagar to Poonamallee)",
            "note": "This is generated data based on typical Chennai traffic patterns"
        }
        
        # Add some route points (simplified)
        route_info["waypoints"] = [
            {"name": "T Nagar", "coordinates": start_location},
            {"name": "Vadapalani", "coordinates": (13.0521, 80.2121)},
            {"name": "Koyambedu", "coordinates": (13.0694, 80.1948)},
            {"name": "Maduravoyal", "coordinates": (13.0662, 80.1662)},
            {"name": "Poonamallee", "coordinates": end_location}
        ]
        
        # Cache the route
        route_key = f"{start_location}_{end_location}"
        self.routes[route_key] = route_info
        
        return route_info
    
    def _apply_quantum_enhancement(self, base_time_mins):
        """Simulate quantum enhancement for predicted travel time"""
        # Create a simple quantum circuit to model traffic variation
        circuit = Circuit()
        
        # Add randomness to simulate different traffic conditions
        theta = np.pi * random.uniform(0.2, 0.7)
        
        # Prepare circuit
        circuit.ry(0, theta)
        circuit.cnot(0, 1)
        circuit.ry(1, theta/2)
        circuit.measure([0, 1])
        
        try:
            # Run the quantum computation
            task = self.device.run(circuit, shots=self.shots)
            result = task.result()
            counts = result.measurement_counts
            
            # Extract the prediction from measurement results
            binary_outcomes = list(counts.keys())
            weights = np.array([int(outcome, 2) for outcome in binary_outcomes])
            probabilities = np.array([counts[outcome]/self.shots for outcome in binary_outcomes])
            
            # Calculate predicted adjustment factor
            variation_factor = np.sum(weights * probabilities) / 3
            
            # Apply to base time
            return base_time_mins * (1 + variation_factor * 0.2)
            
        except Exception as e:
            print(f"Quantum computation failed: {e}")
            # Return with a small random variation
            return base_time_mins * (1 + random.uniform(0.05, 0.15))
    
    def _quantum_traffic_prediction(self, base_time, current_delay):
        """
        Use quantum computing to predict potential additional traffic delays
        based on historical patterns and current conditions
        
        Args:
            base_time (float): Base travel time in seconds
            current_delay (float): Current traffic delay in seconds
            
        Returns:
            float: Predicted traffic delay
        """
        # Create a simple quantum circuit for traffic variance prediction
        circuit = Circuit()
        
        # Use current traffic conditions to initialize qubit states
        traffic_factor = min(current_delay / max(base_time, 1), 1.0)
        theta = traffic_factor * np.pi / 2
        
        # Add gates
        circuit.ry(0, theta)
        circuit.cnot(0, 1)
        circuit.ry(1, theta/2)
        circuit.measure([0, 1])
        
        try:
            # Run the quantum computation
            task = self.device.run(circuit, shots=self.shots)
            result = task.result()
            counts = result.measurement_counts
            
            # Extract the prediction from measurement results
            binary_outcomes = list(counts.keys())
            weights = np.array([int(outcome, 2) for outcome in binary_outcomes])
            probabilities = np.array([counts[outcome]/self.shots for outcome in binary_outcomes])
            
            # Calculate predicted additional delay
            variance_factor = np.sum(weights * probabilities) / 3
            additional_delay = base_time * 0.2 * variance_factor
            
            return current_delay + additional_delay
            
        except Exception as e:
            print(f"Quantum computation failed: {e}")
            # Fallback to classical prediction
            return current_delay * 1.2
    
    def visualize_route(self, route_info):
        """
        Visualize the route and traffic conditions
        """
        if not route_info or "waypoints" not in route_info:
            print("No route waypoints available for visualization")
            return
        
        # Extract waypoints for visualization
        waypoints = route_info["waypoints"]
        
        # Extract coordinates
        lats = [wp["coordinates"][0] for wp in waypoints]
        lons = [wp["coordinates"][1] for wp in waypoints]
        names = [wp["name"] for wp in waypoints]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Plot the route line
        plt.plot(lons, lats, 'b-', linewidth=2.5)
        
        # Plot the waypoints
        plt.scatter(lons, lats, c='red', s=100, zorder=5)
        
        # Add labels
        for i, name in enumerate(names):
            plt.annotate(name, (lons[i], lats[i]), xytext=(5, 5), 
                         textcoords='offset points', fontsize=12)
        
        # Set title with route information
        plt.title(f"Route: {names[0]} to {names[-1]}\n" + 
                 f"Distance: {route_info['distance_km']:.1f} km, " + 
                 f"Time: {route_info['enhanced_time_mins']:.1f} min, " + 
                 f"Speed: {route_info['estimated_speed_kmh']:.1f} km/h", 
                 fontsize=14)
        
        # Set axis labels
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Add traffic indicator
        traffic_factor = route_info.get("traffic_factor", 1.0)
        if traffic_factor > 1.5:
            traffic_status = "Heavy Traffic"
            color = 'red'
        elif traffic_factor > 1.2:
            traffic_status = "Moderate Traffic"
            color = 'orange'
        else:
            traffic_status = "Light Traffic"
            color = 'green'
        
        plt.figtext(0.5, 0.01, f"Traffic Status: {traffic_status}", 
                   fontsize=12, ha='center', color=color)
        
        # Show the plot
        plt.tight_layout()
        plt.show()


# Example usage for Chennai T Nagar to Poonamallee route
if __name__ == "__main__":
    # Initialize the router
    router = QuantumEnhancedRouter()
    
    # Location coordinates (latitude, longitude)
    # T Nagar coordinates: approximately 13.0416° N, 80.2339° E
    # Poonamallee coordinates: approximately 13.0465° N, 80.1160° E
    t_nagar = (13.0416, 80.2339)
    poonamallee = (13.0465, 80.1160)
    
    # Get optimized route with fallback data
    # Set use_fallback=True to skip API and use simulated data directly
    route = router.get_optimized_route(t_nagar, poonamallee, use_fallback=True)
    
    if route:
        print("\n--- Optimized Route from T Nagar to Poonamallee ---")
        print(f"Data source: {route['source']}")
        print(f"Distance: {route['distance_km']:.2f} km")
        print(f"Estimated travel time: {route['enhanced_time_mins']:.1f} minutes")
        print(f"Average speed: {route['estimated_speed_kmh']:.1f} km/h")
        print(f"Traffic factor: {route['traffic_factor']:.2f}")
        
        # Show traffic condition
        traffic_factor = route["traffic_factor"]
        if traffic_factor > 1.5:
            print("Traffic condition: Heavy traffic")
        elif traffic_factor > 1.2:
            print("Traffic condition: Moderate traffic")
        else:
            print("Traffic condition: Light traffic")
        
        print("----------------------------------------------------")
        
        # Visualize the route (uncomment if matplotlib is available)
        # router.visualize_route(route)
    else:
        print("Failed to retrieve route information.")
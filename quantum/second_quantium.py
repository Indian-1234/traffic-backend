import requests
import json
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random
import math
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

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
        
        # TomTom API key
        self.tomtom_api_key = os.environ.get("TOMTOM_API_KEY", None)
        
        # Initialize geocoder for address lookup
        try:
            self.geocoder = Nominatim(user_agent="quantum_router")
        except:
            print("Warning: Geocoder could not be initialized")
            self.geocoder = None
    
    def set_tomtom_api_key(self, api_key):
        """Set a valid TomTom API key"""
        self.tomtom_api_key = api_key
        return self
    
    def get_coordinates(self, location_name):
        """
        Convert a location name to coordinates
        
        Args:
            location_name (str): Name of location (e.g., "T Nagar, Chennai")
            
        Returns:
            tuple: (latitude, longitude) or None if not found
        """
        if not self.geocoder:
            print("Error: Geocoder not available")
            return None
            
        try:
            location = self.geocoder.geocode(location_name)
            if location:
                return (location.latitude, location.longitude)
            else:
                print(f"Could not find coordinates for '{location_name}'")
                return None
        except Exception as e:
            print(f"Error in geocoding: {e}")
            return None
    
    def get_optimized_route(self, start_location, end_location, use_fallback=False):
        """
        Get the optimized (fastest) route between two locations
        
        Args:
            start_location: Either (latitude, longitude) tuple or location name string
            end_location: Either (latitude, longitude) tuple or location name string
            use_fallback (bool): Whether to use simulated data if API fails
            
        Returns:
            dict: Optimized route information
        """
        # Convert location names to coordinates if needed
        start_coords = self._ensure_coordinates(start_location)
        end_coords = self._ensure_coordinates(end_location)
        
        if not start_coords or not end_coords:
            return None
        
        if not use_fallback and self.tomtom_api_key:
            try:
                return self._get_route_from_api(start_coords, end_coords, "fastest")
            except Exception as e:
                print(f"API Error: {e}")
                print("Falling back to simulated data...")
        
        # Fallback to simulated data for route
        return self._get_simulated_route(start_coords, end_coords, "fastest")
    
    def get_shortest_path_route(self, start_location, end_location, use_fallback=False):
        """
        Get the shortest path route between two locations (by distance rather than time)
        
        Args:
            start_location: Either (latitude, longitude) tuple or location name string
            end_location: Either (latitude, longitude) tuple or location name string
            use_fallback (bool): Whether to use simulated data if API fails
            
        Returns:
            dict: Shortest path route information
        """
        # Convert location names to coordinates if needed
        start_coords = self._ensure_coordinates(start_location)
        end_coords = self._ensure_coordinates(end_location)
        
        if not start_coords or not end_coords:
            return None
            
        if not use_fallback and self.tomtom_api_key:
            try:
                return self._get_route_from_api(start_coords, end_coords, "shortest")
            except Exception as e:
                print(f"API Error with shortest path: {e}")
                print("Falling back to simulated shortest path data...")
        
        # Fallback to simulated shortest path data
        return self._get_simulated_route(start_coords, end_coords, "shortest")
    
    def _ensure_coordinates(self, location):
        """
        Ensure the location is in coordinate form
        
        Args:
            location: Either (latitude, longitude) tuple or location name string
            
        Returns:
            tuple: (latitude, longitude) coordinates or None if conversion fails
        """
        if isinstance(location, tuple) and len(location) == 2:
            # Already coordinates
            return location
        elif isinstance(location, str):
            # Try to geocode the location name
            return self.get_coordinates(location)
        else:
            print(f"Invalid location format: {location}")
            return None
    
    def _get_route_from_api(self, start_coords, end_coords, route_type="fastest"):
        """
        Get route using TomTom API
        
        Args:
            start_coords (tuple): (latitude, longitude) of start location
            end_coords (tuple): (latitude, longitude) of end location
            route_type (str): Type of route - 'fastest' or 'shortest'
            
        Returns:
            dict: Route information
        """
        # TomTom API endpoint for routing
        base_url = "https://api.tomtom.com/routing/1/calculateRoute"
        
        # Construct the request URL
        url = f"{base_url}/{start_coords[0]},{start_coords[1]}:{end_coords[0]},{end_coords[1]}/json"
        
        # Request parameters
        params = {
            "key": self.tomtom_api_key,
            "traffic": "true",
            "routeRepresentation": "summaryOnly",
            "computeTravelTimeFor": "all",
            "routeType": route_type  # 'fastest' or 'shortest'
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
            "routing_type": route_type,
            "source": "TomTom API",
            "start": start_coords,
            "end": end_coords,
            "raw_data": route_data
        }
        
        # Attempt to extract waypoints if available
        try:
            route_points = route_data["routes"][0]["legs"][0]["points"]
            waypoints = []
            
            # Extract a reasonable number of waypoints (not all points)
            num_points = len(route_points)
            step = max(1, num_points // 5)  # At most 5 waypoints
            
            for i in range(0, num_points, step):
                if len(waypoints) < 6:  # Limit to 6 total points including start/end
                    point = route_points[i]
                    waypoints.append({
                        "name": f"Waypoint {len(waypoints)+1}",
                        "coordinates": (point["latitude"], point["longitude"])
                    })
            
            # Ensure start and end are included
            if waypoints[0]["coordinates"] != start_coords:
                waypoints.insert(0, {"name": "Start", "coordinates": start_coords})
            if waypoints[-1]["coordinates"] != end_coords:
                waypoints.append({"name": "End", "coordinates": end_coords})
                
            route_info["waypoints"] = waypoints
        except (KeyError, IndexError):
            # If we can't extract waypoints, just use start and end
            route_info["waypoints"] = [
                {"name": "Start", "coordinates": start_coords},
                {"name": "End", "coordinates": end_coords}
            ]
        
        # Cache the route
        route_key = f"{start_coords}_{end_coords}_{route_type}"
        self.routes[route_key] = route_info
        
        return route_info
    
    def _get_simulated_route(self, start_coords, end_coords, route_type="fastest"):
        """
        Generate simulated route data for any location pair
        
        Args:
            start_coords (tuple): (latitude, longitude) of start location
            end_coords (tuple): (latitude, longitude) of end location
            route_type (str): 'fastest' or 'shortest'
            
        Returns:
            dict: Route information
        """
        # Calculate straight-line distance in km
        straight_distance = self._haversine_distance(start_coords, end_coords)
        
        # Route distances are typically longer than straight-line distances
        # Shortest path is closer to straight line but still longer
        if route_type == "shortest":
            # Shortest path is about 1.2-1.3x the straight-line distance
            base_distance_km = straight_distance * (1.25 + random.uniform(-0.05, 0.05))
        else:
            # Fastest route is about 1.3-1.5x the straight-line distance
            base_distance_km = straight_distance * (1.4 + random.uniform(-0.1, 0.1))
        
        # Get current hour (0-23) to simulate time-based traffic
        current_hour = datetime.now().hour
        
        # Traffic multiplier depends on time of day
        if current_hour in [7, 8, 9, 16, 17, 18, 19]:  # Rush hours
            if route_type == "shortest":
                # Shortest routes often have more congestion
                traffic_multiplier = 1.7 + random.uniform(-0.1, 0.3)
            else:
                # Fastest routes might have less congestion (highways, etc.)
                traffic_multiplier = 1.4 + random.uniform(-0.1, 0.2)
        else:  # Non-rush hours
            if route_type == "shortest":
                traffic_multiplier = 1.3 + random.uniform(-0.1, 0.2)
            else:
                traffic_multiplier = 1.1 + random.uniform(-0.05, 0.15)
        
        # Base speeds differ by route type
        if route_type == "shortest":
            # Shorter routes often use smaller roads
            base_speed_kmh = 35.0 + random.uniform(-5, 5)
        else:
            # Faster routes often use highways
            base_speed_kmh = 50.0 + random.uniform(-5, 5)
        
        # Calculate times
        base_time_mins = (base_distance_km / base_speed_kmh) * 60
        actual_time_mins = base_time_mins * traffic_multiplier
        
        # Apply quantum enhancement if available
        if self.device:
            enhanced_time_mins = self._apply_quantum_enhancement(actual_time_mins)
        else:
            enhanced_time_mins = actual_time_mins * (1 + random.uniform(0, 0.15))
        
        # Calculate actual speed with traffic
        actual_speed_kmh = base_distance_km / (enhanced_time_mins / 60)
        
        # Generate waypoints along the path
        waypoints = self._generate_waypoints(start_coords, end_coords)
        
        # Prepare route info
        route_info = {
            "distance_km": base_distance_km,
            "baseline_time_mins": base_time_mins,
            "enhanced_time_mins": enhanced_time_mins,
            "traffic_factor": traffic_multiplier,
            "estimated_speed_kmh": actual_speed_kmh,
            "routing_type": route_type,
            "source": f"Simulated {route_type} route data",
            "start": start_coords,
            "end": end_coords,
            "waypoints": waypoints
        }
        
        # Cache the route
        route_key = f"{start_coords}_{end_coords}_{route_type}"
        self.routes[route_key] = route_info
        
        return route_info
    
    def _haversine_distance(self, point1, point2):
        """
        Calculate the great-circle distance between two points in km
        using the Haversine formula
        """
        try:
            return geodesic(point1, point2).kilometers
        except:
            # Fallback if geopy is not available
            lat1, lon1 = point1
            lat2, lon2 = point2
            
            # Convert latitude and longitude from degrees to radians
            lat1 = math.radians(lat1)
            lon1 = math.radians(lon1)
            lat2 = math.radians(lat2)
            lon2 = math.radians(lon2)
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            # Earth radius in kilometers
            earth_radius = 6371.0
            
            # Calculate distance
            distance = earth_radius * c
            return distance
    
    def _generate_waypoints(self, start_coords, end_coords, num_points=3):
        """
        Generate waypoints between start and end coordinates
        
        Args:
            start_coords (tuple): (latitude, longitude) of start
            end_coords (tuple): (latitude, longitude) of end
            num_points (int): Number of intermediate points to generate
            
        Returns:
            list: List of waypoint dictionaries
        """
        waypoints = [{"name": "Start", "coordinates": start_coords}]
        
        # Generate intermediate points with slight variation
        for i in range(num_points):
            # How far along the path (0.0 to 1.0)
            t = (i + 1) / (num_points + 1)
            
            # Basic linear interpolation between start and end
            lat = start_coords[0] + t * (end_coords[0] - start_coords[0])
            lon = start_coords[1] + t * (end_coords[1] - start_coords[1])
            
            # Add some random variation to make it look like a real route
            # The variation is proportional to the distance
            distance = self._haversine_distance(start_coords, end_coords)
            variation_factor = min(0.01, distance / 1000)  # Limit variation for very long distances
            
            lat_variation = random.uniform(-0.01, 0.01) * variation_factor
            lon_variation = random.uniform(-0.01, 0.01) * variation_factor
            
            waypoints.append({
                "name": f"Waypoint {i+1}",
                "coordinates": (lat + lat_variation, lon + lon_variation)
            })
        
        waypoints.append({"name": "End", "coordinates": end_coords})
        return waypoints
    
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
        routing_type = route_info.get("routing_type", "optimized")
        start_name = waypoints[0]["name"]
        end_name = waypoints[-1]["name"]
        plt.title(f"{routing_type.capitalize()} Route: {start_name} to {end_name}\n" + 
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
        
    def compare_routes(self, start_location, end_location, use_fallback=True):
        """
        Compare fastest and shortest routes between two locations
        
        Args:
            start_location: Either (latitude, longitude) tuple or location name string
            end_location: Either (latitude, longitude) tuple or location name string
            use_fallback (bool): Whether to use simulated data
            
        Returns:
            tuple: (fastest_route, shortest_route) - dictionaries with route information
        """
        # Get the fastest route
        fastest_route = self.get_optimized_route(start_location, end_location, use_fallback=use_fallback)
        if not fastest_route:
            print("Failed to get fastest route")
            return None, None
            
        # Get the shortest route
        shortest_route = self.get_shortest_path_route(start_location, end_location, use_fallback=use_fallback)
        if not shortest_route:
            print("Failed to get shortest route")
            return fastest_route, None
        
        # Get location names
        if "waypoints" in fastest_route and len(fastest_route["waypoints"]) > 0:
            start_name = fastest_route["waypoints"][0]["name"]
            end_name = fastest_route["waypoints"][-1]["name"]
        else:
            start_name = "Start"
            end_name = "End"
        
        # Display comparison
        print(f"\n--- Route Comparison: {start_name} to {end_name} ---")
        
        print("\nFASTEST ROUTE:")
        print(f"Source: {fastest_route['source']}")
        print(f"Distance: {fastest_route['distance_km']:.2f} km")
        print(f"Estimated travel time: {fastest_route['enhanced_time_mins']:.1f} minutes")
        print(f"Average speed: {fastest_route['estimated_speed_kmh']:.1f} km/h")
        print(f"Traffic factor: {fastest_route['traffic_factor']:.2f}")
        
        print("\nSHORTEST ROUTE:")
        print(f"Source: {shortest_route['source']}")
        print(f"Distance: {shortest_route['distance_km']:.2f} km")
        print(f"Estimated travel time: {shortest_route['enhanced_time_mins']:.1f} minutes")
        print(f"Average speed: {shortest_route['estimated_speed_kmh']:.1f} km/h")
        print(f"Traffic factor: {shortest_route['traffic_factor']:.2f}")
        
        # Calculate difference
        distance_diff = fastest_route['distance_km'] - shortest_route['distance_km']
        time_diff = fastest_route['enhanced_time_mins'] - shortest_route['enhanced_time_mins']
        
        print("\nCOMPARISON:")
        print(f"Distance difference: {abs(distance_diff):.2f} km ({'shorter' if distance_diff > 0 else 'longer'} on shortest path)")
        print(f"Time difference: {abs(time_diff):.1f} minutes ({'faster' if time_diff > 0 else 'slower'} on shortest path)")
        
        # Recommendation
        if time_diff < -5:  # Fastest route is more than 5 minutes quicker
            print("\nRECOMMENDATION: Take the FASTEST route - significantly quicker")
        elif time_diff > 5:  # Shortest route is more than 5 minutes quicker
            print("\nRECOMMENDATION: Take the SHORTEST route - significantly quicker")
        elif abs(distance_diff) > 2 and abs(time_diff) < 5:  # Similar time, but shortest saves distance
            print("\nRECOMMENDATION: Take the SHORTEST route - similar time but saves fuel")
        else:
            print("\nRECOMMENDATION: Routes are comparable - choose based on preference")
            
        return fastest_route, shortest_route


# Example usage for any location
if __name__ == "__main__":
    # Initialize the router
    router = QuantumEnhancedRouter()
    
    # Test with different location formats
    # Option 1: Using coordinates
    print("\n=== TEST WITH COORDINATES ===")
    # New York City to Boston coordinates
    nyc = (40.7128, -74.0060)
    boston = (42.3601, -71.0589)
    fastest1, shortest1 = router.compare_routes(nyc, boston, use_fallback=True)
    
    # Option 2: Using location names (requires geocoder to be working)
    print("\n=== TEST WITH LOCATION NAMES ===")
    fastest2, shortest2 = router.compare_routes("Mumbai, India", "Pune, India", use_fallback=True)
    
    # Option 3: Chennai T Nagar to Poonamallee (original example)
    print("\n=== TEST WITH CHENNAI LOCATIONS ===")
    t_nagar = (13.0416, 80.2339)
    poonamallee = (13.0465, 80.1160)
    fastest3, shortest3 = router.compare_routes(t_nagar, poonamallee, use_fallback=True)
    
    # Visualize one of the routes
    print("\nVisualizing route from NYC to Boston:")
    if fastest1:
        router.visualize_route(fastest1)
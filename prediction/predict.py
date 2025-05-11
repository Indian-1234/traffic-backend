import numpy as np
import boto3
import requests
import time
import matplotlib.pyplot as plt
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

# TomTom API configuration
TOMTOM_API_KEY = "IV7dQDp5vey54vgGvRlIDmn7qazKzAaN"
TOMTOM_TRAFFIC_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

# AWS Braket configuration
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
S3_FOLDER = ("amazon-braket-quantiumhitter", "quantum-results")


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
        return {"congestion": 0.5, "confidence": 0.5, "prediction": "Moderate traffic expected"}
    
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
    
    # Generate prediction message
    if congestion < 0.3:
        prediction = "Light traffic expected"
    elif congestion < 0.7:
        prediction = "Moderate traffic expected"
    else:
        prediction = "Heavy traffic expected"
    
    return {
        "congestion": congestion,
        "confidence": confidence,
        "prediction": prediction
    }


def visualize_prediction(counts, prediction, location):
    """
    Visualize the quantum measurement results and traffic prediction.
    """
    if not counts:
        print("No results to visualize.")
        return
    
    # Sort counts by bitstring
    sorted_counts = dict(sorted(counts.items()))
    
    # Create a bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Quantum states distribution
    plt.subplot(2, 1, 1)
    plt.bar(sorted_counts.keys(), sorted_counts.values(), color='blue')
    plt.xlabel('Quantum State')
    plt.ylabel('Measurement Count')
    plt.title(f'Quantum Traffic Prediction for {location}')
    plt.xticks(rotation=45)
    
    # Plot 2: Traffic prediction visualization
    plt.subplot(2, 1, 2)
    congestion = prediction['congestion']
    
    # Create a simple traffic visualization
    plt.barh(['Traffic Level'], [congestion], color='red', alpha=0.7)
    plt.barh(['Traffic Level'], [1], color='lightgray', alpha=0.3)
    
    # Add markers for traffic levels
    plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.5)
    
    # Add text labels
    plt.text(0.15, 0.8, 'Light', transform=plt.gca().transAxes, color='green')
    plt.text(0.45, 0.8, 'Moderate', transform=plt.gca().transAxes, color='orange')
    plt.text(0.8, 0.8, 'Heavy', transform=plt.gca().transAxes, color='red')
    
    # Add prediction
    plt.text(0.1, -0.2, f"Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.2f})",
             transform=plt.gca().transAxes, fontsize=12)
    
    plt.xlim(0, 1)
    plt.title('Traffic Congestion Prediction')
    
    plt.tight_layout()
    plt.savefig(f"traffic_prediction_{location.replace(' ', '_')}.png")
    print(f"Visualization saved as traffic_prediction_{location.replace(' ', '_')}.png")
    plt.close()


def main():
    """
    Main function to run the quantum traffic prediction.
    """
    print("Quantum Traffic Prediction System")
    print("================================")
    
    # Define locations for traffic prediction
    locations = [
        {"name": "Downtown", "lat": 40.7128, "lon": -74.0060},  # New York
        {"name": "Suburbs", "lat": 40.7431, "lon": -74.0335}    # Jersey City
    ]
    
    # Number of qubits to use for prediction
    n_qubits = 3
    
    # Use local simulation by default
    use_aws = False  # Set to True to use AWS Braket (requires AWS credentials)
    
    for location in locations:
        print(f"\nPredicting traffic for {location['name']}...")
        
        # Get traffic data from TomTom API
        current_speed, free_flow_speed, confidence = get_traffic_data(
            location['lat'], location['lon']
        )
        
        # Normalize traffic data for quantum circuit
        congestion_angle, confidence_angle = normalize_traffic_data(
            current_speed, free_flow_speed, confidence
        )
        
        # Create quantum circuit for traffic prediction
        print("Creating quantum circuit...")
        circuit = create_traffic_prediction_circuit(
            congestion_angle, confidence_angle, n_qubits
        )
        print(f"Circuit created: {circuit}")
        
        # Run the circuit
        if use_aws:
            try:
                print("Running on AWS Braket...")
                counts = run_circuit_on_braket(circuit)
            except Exception as e:
                print(f"Error with AWS: {str(e)}")
                print("Falling back to local simulation...")
                counts = run_circuit_locally(circuit)
        else:
            counts = run_circuit_locally(circuit)
        
        print(f"Measurement results: {counts}")
        
        # Interpret the results
        prediction = interpret_traffic_prediction(counts, n_qubits)
        print(f"Traffic prediction for {location['name']}: {prediction['prediction']}")
        print(f"Congestion level: {prediction['congestion']:.2f}")
        print(f"Prediction confidence: {prediction['confidence']:.2f}")
        
        # Visualize the results
        visualize_prediction(counts, prediction, location['name'])
    
    print("\nTraffic prediction complete!")


if __name__ == "__main__":
    main()
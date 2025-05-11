"""
Simple AWS Braket Quantum Computing Example

This is a minimal working example for AWS Braket that should work
across different versions of the SDK. It solves a simple quantum
problem using AWS's quantum computing services.
"""

import numpy as np
import boto3
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import time
import matplotlib.pyplot as plt


def create_bell_state_circuit():
    """
    Create a Bell state (entangled state) circuit.
    This is one of the simplest non-trivial quantum circuits.
    """
    # Initialize circuit with 2 qubits
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure(0)  # Fixed
    circuit.measure(1)  # Fixed
    return circuit


def create_quantum_fourier_transform_circuit(n_qubits=3):
    circuit = Circuit()
    
    # Apply Hadamard gates to all qubits to create superposition
    for i in range(n_qubits):
        circuit.h(i)
    
    # Apply controlled phase rotations
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            angle = 2 * np.pi / (2 ** (j - i + 1))
            circuit.cphaseshift(control=i, target=j, angle=angle)
    
    # Measure all qubits
    for i in range(n_qubits):
        circuit.measure(i)  # FIXED
    
    return circuit


def create_bernstein_vazirani_circuit(secret_string="101"):
    """
    Create a Bernstein-Vazirani circuit to find a secret string.
    This algorithm demonstrates quantum parallelism and interference.
    """
    n = len(secret_string)
    circuit = Circuit()
    
    # Apply Hadamard gates to all qubits
    for i in range(n):
        circuit.h(i)
    
    # Apply X and Z gates based on the secret string
    for i in range(n):
        if secret_string[i] == "1":
            circuit.z(i)
    
    # Apply second round of Hadamard gates
    for i in range(n):
        circuit.h(i)
    
    # Measure all qubits
    for i in range(n):
        circuit.measure(i, i)
    
    return circuit


def run_circuit_locally(circuit, shots=1000):
    """
    Run the circuit on a local simulator.
    This is free and doesn't require AWS credentials.
    """
    # Create local simulator device
    device = LocalSimulator()
    
    # Run the circuit
    print(f"Running circuit with {len(circuit.qubits)} qubits locally...")
    start_time = time.time()
    result = device.run(circuit, shots=shots).result()
    end_time = time.time()
    
    print(f"Circuit execution time: {end_time - start_time:.2f} seconds")
    
    # Get measurement counts
    counts = result.measurement_counts
    
    return counts


def run_circuit_on_braket_simulator(circuit, shots=1000):
    """
    Run the circuit on AWS Braket's SV1 simulator.
    This requires AWS credentials and incurs costs.
    """
    try:
        # Initialize Braket client
        braket_client = boto3.client('braket')
        
        # Get ARN of the SV1 simulator
        sv1_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        
        # Use SV1 simulator
        from braket.aws import AwsDevice
        device = AwsDevice(sv1_arn)
        
        # Define S3 bucket for results
        s3_folder = ("amazon-braket-quantiumhitter", "quantum-results")
        
        # Run the circuit
        print(f"Running circuit with {len(circuit.qubits)} qubits on SV1 simulator...")
        start_time = time.time()
        task = device.run(circuit, s3_folder, shots=shots)
        print(f"Task ARN: {task.id}")
        
        # Wait for results
        print("Waiting for results...")
        result = task.result()
        end_time = time.time()
        
        print(f"Circuit execution time: {end_time - start_time:.2f} seconds")
        
        # Get measurement counts
        counts = result.measurement_counts
        
        return counts
    
    except Exception as e:
        print(f"Error running circuit on AWS Braket: {str(e)}")
        return {}


def visualize_results(counts, title="Quantum Circuit Results"):
    """
    Visualize the measurement results.
    """
    if not counts:
        print("No results to visualize.")
        return
    
    # Sort counts by bitstring
    sorted_counts = dict(sorted(counts.items()))
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_counts.keys(), sorted_counts.values())
    plt.xlabel('Bitstring')
    plt.ylabel('Counts')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()


def main():
    """Main function to run quantum circuits."""
    print("AWS Braket Quantum Computing Example")
    print("===================================")
    
    # Create a circuit
    print("\nCreating Bell state circuit...")
    bell_circuit = create_bell_state_circuit()
    print(f"Circuit: {bell_circuit}")
    
    # Run locally (this is free and doesn't require AWS credentials)
    bell_counts = run_circuit_locally(bell_circuit)
    print(f"Bell state measurement results: {bell_counts}")
    visualize_results(bell_counts, "Bell State Results")
    
    # Create QFT circuit
    print("\nCreating Quantum Fourier Transform circuit...")
    qft_circuit = create_quantum_fourier_transform_circuit(n_qubits=3)
    print(f"Circuit: {qft_circuit}")
    
    # Run locally
    qft_counts = run_circuit_locally(qft_circuit)
    print(f"QFT measurement results: {qft_counts}")
    visualize_results(qft_counts, "Quantum Fourier Transform Results")
    
    # Create Bernstein-Vazirani circuit with secret string "101"
    print("\nCreating Bernstein-Vazirani circuit...")
    bv_circuit = create_bernstein_vazirani_circuit("101")
    print(f"Circuit: {bv_circuit}")
    
    # Run locally
    bv_counts = run_circuit_locally(bv_circuit)
    print(f"Bernstein-Vazirani measurement results: {bv_counts}")
    print(f"Most frequent bitstring: {max(bv_counts, key=bv_counts.get)}")
    visualize_results(bv_counts, "Bernstein-Vazirani Results")
    
    # Optionally run on AWS Braket SV1 simulator (costs money)
    run_on_aws = False  # Set to True if you want to run on AWS
    if run_on_aws:
        try:
            print("\nRunning circuit on AWS Braket SV1 simulator...")
            aws_counts = run_circuit_on_braket_simulator(bell_circuit)
            print(f"AWS Braket results: {aws_counts}")
            visualize_results(aws_counts, "AWS Braket Results")
        except Exception as e:
            print(f"Error running on AWS: {str(e)}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
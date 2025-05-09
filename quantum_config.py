import numpy as np
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# =====================================================
# Quantum Circuit Definitions
# =====================================================
def create_quantum_device(n_qubits=4):
    """Create quantum device with specified number of qubits"""
    return qml.device("default.qubit", wires=n_qubits)

def H_eigenvalue_circuit(params, features, wires=[0, 1, 2, 3]):
    """
    Hybrid Quantum Circuit for calculating H eigenvalues
    
    Args:
        params: Trainable parameters for the quantum circuit
        features: Classical input features
        wires: Quantum circuit wires/qubits
    """
    # Feature embedding
    for i, feat in enumerate(features):
        qml.RY(feat, wires=i % len(wires))
    
    # Entanglement layers
    for i in range(len(wires)):
        qml.CNOT(wires=[i, (i+1) % len(wires)])
    
    # Parameterized rotation gates
    for i in range(len(wires)):
        qml.RX(params[i], wires=i)
        qml.RY(params[i + len(wires)], wires=i)
        qml.RZ(params[i + 2*len(wires)], wires=i)
    
    # Second entanglement layer
    for i in range(len(wires)):
        qml.CNOT(wires=[i, (i+1) % len(wires)])
    
    # Measurement operations - calculate eigenvalues
    return [qml.expval(qml.PauliZ(i)) for i in range(len(wires))]

# Create the quantum device
dev = create_quantum_device(4)

# Define the QNode
@qml.qnode(dev)
def quantum_circuit(params, features):
    return H_eigenvalue_circuit(params, features)

# =====================================================
# Bayesian Optimization Implementation
# =====================================================
class BayesianOptimizer:
    """Simple Bayesian optimization implementation"""
    
    def __init__(self, param_ranges, objective_function):
        self.param_ranges = param_ranges
        self.objective_function = objective_function
        self.samples = []
        self.values = []
        
    def acquisition_function(self, params):
        """Upper confidence bound acquisition function"""
        if not self.samples:
            return 1.0
        
        # Calculate mean and variance estimates
        distances = [np.linalg.norm(np.array(params) - np.array(s)) for s in self.samples]
        closest_idx = np.argmin(distances)
        
        # Simple exploration-exploitation trade-off
        exploitation = -self.values[closest_idx]  # Negative because we're minimizing
        exploration = 1.0 / (min(distances) + 1e-6)
        
        return exploitation + 0.1 * exploration
        
    def optimize(self, n_iterations=20):
        """Run the Bayesian optimization process"""
        best_params = None
        best_value = float('inf')
        
        for _ in range(n_iterations):
            # Sample new point based on acquisition function
            if not self.samples:
                # First iteration - random sample
                params = [np.random.uniform(low, high) for low, high in self.param_ranges]
            else:
                # Use acquisition function to find most promising point
                res = minimize(
                    lambda x: -self.acquisition_function(x),  # Negate for minimization
                    x0=self.samples[np.argmin(self.values)],
                    bounds=self.param_ranges,
                    method='L-BFGS-B'
                )
                params = res.x
            
            # Evaluate objective function
            value = self.objective_function(params)
            
            # Store sample
            self.samples.append(params)
            self.values.append(value)
            
            # Update best value
            if value < best_value:
                best_value = value
                best_params = params
                
        return best_params, best_value

# =====================================================
# Traffic Prediction Model
# =====================================================
class QuantumTrafficPredictor:
    """Quantum machine learning model for traffic prediction"""
    
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.n_params = n_qubits * 3  # 3 rotation gates per qubit
        self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
        self.feature_scaler = MinMaxScaler()
        
    def preprocess_features(self, features, fit=False):
        """Scale features to appropriate range"""
        features_array = np.array(features).reshape(1, -1)
        if fit:
            return self.feature_scaler.fit_transform(features_array).flatten()
        return self.feature_scaler.transform(features_array).flatten()
        
    def predict(self, features):
        """Make a prediction using the quantum circuit"""
        # Preprocess features - use only the first n_qubits features
        scaled_features = self.preprocess_features(features)[:self.n_qubits]
        
        # Ensure we have enough features
        if len(scaled_features) < self.n_qubits:
            # Pad with zeros if needed
            scaled_features = np.pad(scaled_features, 
                                    (0, self.n_qubits - len(scaled_features)),
                                    'constant')
        
        # Get quantum circuit output
        result = quantum_circuit(self.params, scaled_features)
        
        # Process the result to get a prediction value between 0 and 1
        # Average the expectation values and rescale from [-1,1] to [0,1]
        prediction = (np.mean(result) + 1) / 2
        
        return prediction
        
    def loss_function(self, params, features, targets):
        """Calculate mean squared error loss"""
        self.params = params
        predictions = [self.predict(feature) for feature in features]
        return np.mean((np.array(predictions) - np.array(targets))**2)
        
    def train(self, training_features, training_targets, n_iterations=100):
        """Train the model using Bayesian optimization"""
        # Fit the feature scaler
        all_features = np.array(training_features)
        self.feature_scaler.fit(all_features)
        
        # Scale features
        scaled_features = [self.preprocess_features(f, fit=False) for f in training_features]
        
        # Define objective function for Bayesian optimization
        def objective_function(params):
            return self.loss_function(params, scaled_features, training_targets)
        
        # Set up Bayesian optimizer
        param_ranges = [(0, 2*np.pi) for _ in range(self.n_params)]
        optimizer = BayesianOptimizer(param_ranges, objective_function)
        
        # Run optimization
        best_params, best_loss = optimizer.optimize(n_iterations)
        
        # Update model parameters
        self.params = best_params
        
        return best_loss
        
    def save_model(self, filepath):
        """Save the model parameters to a file"""
        model_data = {
            'params': np.array(self.params).tolist(),
            'feature_scaler_data': {
                'scale_': self.feature_scaler.scale_.tolist(),
                'min_': self.feature_scaler.min_.tolist(),
                'data_min_': self.feature_scaler.data_min_.tolist(),
                'data_max_': self.feature_scaler.data_max_.tolist(),
                'data_range_': self.feature_scaler.data_range_.tolist()
            }
        }
        
        with open(filepath, 'w') as f:
            import json
            json.dump(model_data, f)
        
    def load_model(self, filepath):
        """Load model parameters from a file"""
        with open(filepath, 'r') as f:
            import json
            model_data = json.load(f)
        
        self.params = np.array(model_data['params'])
        
        # Reconstruct the scaler
        scaler_data = model_data['feature_scaler_data']
        self.feature_scaler.scale_ = np.array(scaler_data['scale_'])
        self.feature_scaler.min_ = np.array(scaler_data['min_'])
        self.feature_scaler.data_min_ = np.array(scaler_data['data_min_'])
        self.feature_scaler.data_max_ = np.array(scaler_data['data_max_'])
        self.feature_scaler.data_range_ = np.array(scaler_data['data_range_'])

# =====================================================
# Sample Training Data Generator
# =====================================================
def generate_training_data(n_samples=100):
    """Generate synthetic training data for model development"""
    np.random.seed(42)  # For reproducibility
    
    features = []
    targets = []
    
    for _ in range(n_samples):
        # Generate random features
        vehicle_count = np.random.randint(0, 200)
        weather_condition = np.random.uniform(0, 1)
        time_of_day = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        # Create feature vector
        feature = [
            vehicle_count, 
            weather_condition, 
            time_of_day / 24.0,  # Normalize to [0,1]
            day_of_week / 6.0    # Normalize to [0,1]
        ]
        
        # Generate synthetic target
        # Higher vehicle count, bad weather, and peak hours lead to more congestion
        peak_hour = (time_of_day >= 7 and time_of_day <= 9) or (time_of_day >= 16 and time_of_day <= 18)
        weekend = day_of_week >= 5
        
        # Base congestion level
        congestion = 0.2
        
        # Add factor for vehicle count
        congestion += vehicle_count / 200 * 0.4
        
        # Add weather factor
        congestion += weather_condition * 0.2
        
        # Add peak hour factor
        if peak_hour and not weekend:
            congestion += 0.2
        
        # Add some noise
        congestion += np.random.normal(0, 0.05)
        
        # Clip to [0,1] range
        congestion = max(0, min(1, congestion))
        
        features.append(feature)
        targets.append(congestion)
    
    return features, targets
# from typing import Any, Dict, Tuple
# from fastapi import APIRouter, HTTPException
# from datetime import datetime
# import networkx as nx
# from concurrent.futures import ThreadPoolExecutor

# from quantum.aws_quantum_integration import AWSQuantumProvider
# from quantum.optimizer import AWSHybridQuantumBayesianAlgorithm
# from traffic_graph import TrafficGraph

# class SynchronousTrafficNetworkOptimizer:
#     """
#     A fully synchronous version of the traffic network optimizer.
#     """
    
#     def __init__(self,
#                  n_qubits=8,
#                  shots=200,
#                  device_arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
#                  region="us-west-1",
#                  s3_bucket="amazon-braket-quantiumhitter"):
#         """
#         Initialize the synchronous traffic network optimizer
        
#         Args:
#             n_qubits: Number of qubits to use
#             shots: Number of measurement shots
#             device_arn: ARN of AWS Braket device to use
#             region: AWS region
#             s3_bucket: S3 bucket for result storage
#         """
#         # Set up AWS Quantum Provider
#         self.aws_provider = AWSQuantumProvider(
#             n_qubits=n_qubits,
#             shots=shots,
#             device_arn=device_arn,
#             region=region,
#             s3_bucket=s3_bucket
#         )
        
#         # Initialize quantum algorithm with AWS provider
#         self.quantum_algorithm = AWSHybridQuantumBayesianAlgorithm(
#             aws_provider=self.aws_provider
#         )
    
#     def optimize_traffic_network(self, traffic_graph: nx.Graph) -> Tuple[nx.Graph, Dict[str, Any]]:
#         """
#         Synchronous version of optimize_traffic_network
#         """
#         # Extract subgraphs to analyze
#         subgraphs = self._extract_subgraphs(traffic_graph)
        
#         # Analyze each subgraph
#         subgraph_features = []
#         for i, subgraph in enumerate(subgraphs):
#             print(f"Analyzing subgraph indian")
#             print(f"Analyzing subgraph {i+1}/{len(subgraphs)} with AWS Quantum Computing...")
            
#             # Use extract_graph_features but ensure it's synchronous
#             features = self._extract_graph_features_sync(subgraph)
#             subgraph_features.append(features)
        
#         # Optimize each subgraph based on features
#         optimized_graph = traffic_graph.copy()
        
#         # Update edge weights based on quantum analysis
#         for (u, v, data) in optimized_graph.edges(data=True):
#             # Find which subgraph this edge belongs to
#             subgraph_idx = self._find_edge_subgraph(u, v, subgraphs)
            
#             if subgraph_idx is not None:
#                 features = subgraph_features[subgraph_idx]
                
#                 # Calculate optimized weight based on quantum features
#                 base_weight = data.get('weight', 1.0)
#                 flow_factor = features["flow_rate"]
#                 congestion_factor = features["congestion_factor"]
                
#                 # Lower weight for better flow, higher for congestion
#                 optimized_weight = base_weight * (1 + congestion_factor - flow_factor)
                
#                 # Update edge weight
#                 optimized_graph[u][v]['weight'] = max(0.1, optimized_weight)
                
#                 # Add signalization recommendation
#                 if features["signalization_density"] > 0.7:
#                     optimized_graph[u][v]['needs_signal'] = True
#                     optimized_graph[u][v]['signal_priority'] = features["signalization_density"]
        
#         # Calculate optimization metrics
#         optimization_metrics = {
#             "average_flow_rate": sum(f["flow_rate"] for f in subgraph_features) / len(subgraph_features) if subgraph_features else 0,
#             "average_congestion": sum(f["congestion_factor"] for f in subgraph_features) / len(subgraph_features) if subgraph_features else 0,
#             "signalization_recommendations": sum(1 for _, _, d in optimized_graph.edges(data=True) 
#                                                if d.get('needs_signal', False)),
#             "optimization_score": sum(1 - f["congestion_factor"] + f["flow_rate"] 
#                                    for f in subgraph_features) / len(subgraph_features) if subgraph_features else 0,
#             "quantum_tasks": self.quantum_algorithm.task_ids
#         }
        
#         return optimized_graph, optimization_metrics
    
#     def _extract_graph_features_sync(self, graph: nx.Graph) -> Dict[str, float]:
#         """
#         Synchronous version of extract_graph_features
#         """
#         # Compute eigenvalues
#         eigenvalues = self.quantum_algorithm.compute_h_eigenvalues(graph)
        
#         # Sort eigenvalues
#         sorted_eigenvalues = sorted(eigenvalues)
        
#         # Extract useful features from eigenvalues
#         features = {
#             "spectral_radius": float(max(abs(val) for val in eigenvalues)),
#             "spectral_gap": float(sorted_eigenvalues[-1] - sorted_eigenvalues[-2]) if len(sorted_eigenvalues) >= 2 else 0.0,
#             "average_eigenvalue": float(sum(eigenvalues) / len(eigenvalues)) if eigenvalues else 0.0,
#             "eigenvalue_spread": float(max(eigenvalues) - min(eigenvalues)) if eigenvalues else 0.0,
#             "algebraic_connectivity": float(sorted_eigenvalues[1]) if len(sorted_eigenvalues) >= 2 else 0.0,
#             "network_complexity": float(sum(val**2 for val in eigenvalues)),
#         }
        
#         # Add interpretations for traffic analysis
#         features["flow_rate"] = 0.5 + 0.5 * features["spectral_gap"] / max(1.0, features["spectral_radius"])
#         features["signalization_density"] = min(1.0, features["network_complexity"] / 10.0)
#         features["congestion_factor"] = 1.0 - min(1.0, features["algebraic_connectivity"] * 2.0)
        
#         return features
    
#     def _extract_subgraphs(self, graph: nx.Graph, max_size=8):
#         # Implementation remains the same as in original class
#         if len(graph) <= max_size:
#             return [graph]
        
#         # Use connected components
#         components = list(nx.connected_components(graph))
        
#         subgraphs = []
#         for component in components:
#             subgraph = graph.subgraph(component).copy()
            
#             # If component is still too large, use spectral clustering to break it down
#             if len(subgraph) > max_size:
#                 clusters = self._spectral_clustering(subgraph, max_size)
#                 for cluster in clusters:
#                     if cluster:
#                         subgraphs.append(graph.subgraph(cluster).copy())
#             else:
#                 subgraphs.append(subgraph)
        
#         return subgraphs
    
#     def _spectral_clustering(self, graph: nx.Graph, max_size=8):
#         # Implementation remains the same as in original class
#         import numpy as np
#         laplacian = nx.normalized_laplacian_matrix(graph).todense()
#         eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
#         # Number of clusters needed
#         n_clusters = int(np.ceil(len(graph) / max_size))
        
#         # Use second eigenvector for Fiedler bipartitioning
#         fiedler = eigenvectors[:, 1]
#         median = np.median(fiedler)
        
#         # Simple bipartition
#         cluster1 = [i for i, val in enumerate(fiedler) if val <= median]
#         cluster2 = [i for i, val in enumerate(fiedler) if val > median]
        
#         # Map back to original node IDs
#         nodes = list(graph.nodes())
#         cluster1 = [nodes[i] for i in cluster1]
#         cluster2 = [nodes[i] for i in cluster2]
        
#         # If we need more clusters, recursively split
#         if n_clusters > 2:
#             result = []
#             if len(cluster1) > max_size:
#                 subgraph1 = graph.subgraph(cluster1).copy()
#                 result.extend(self._spectral_clustering(subgraph1, max_size))
#             else:
#                 result.append(cluster1)
                
#             if len(cluster2) > max_size:
#                 subgraph2 = graph.subgraph(cluster2).copy()
#                 result.extend(self._spectral_clustering(subgraph2, max_size))
#             else:
#                 result.append(cluster2)
#             return result
#         else:
#             return [cluster1, cluster2]
    
#     def _find_edge_subgraph(self, u, v, subgraphs):
#         # Implementation remains the same as in original class
#         for i, subgraph in enumerate(subgraphs):
#             if subgraph.has_edge(u, v):
#                 return i
#         return None

# # Update the FastAPI router
# router = APIRouter()
# traffic_network = TrafficGraph()

# # Create a synchronous optimizer with the correct bucket name
# sync_optimizer = SynchronousTrafficNetworkOptimizer(
#     n_qubits=8,
#     shots=200,
#     device_arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
#     region="us-west-1",
#     s3_bucket="amazon-braket-quantiumhitter"  # Make sure this bucket exists in your AWS account
# )
# thread_pool = ThreadPoolExecutor(max_workers=1)

# import asyncio

# @router.post("/optimize-network", response_model=Dict[str, Any])
# async def optimize_network():
#     """
#     Optimize the traffic network using quantum computing.
#     """
#     try:
#         print("Starting network optimization...")
#         graph = traffic_network.graph
        
#         # Run the synchronous optimization in a thread pool
#         loop = asyncio.get_running_loop()
#         optimized_graph, metrics = await loop.run_in_executor(
#             thread_pool,
#             sync_optimizer.optimize_traffic_network,
#             graph
#         )
        
#         # Update the traffic network with optimized values
#         for u, v, data in optimized_graph.edges(data=True):
#             if traffic_network.graph.has_edge(u, v):
#                 traffic_network.update_edge_weight(u, v, data.get('weight', 1.0))
#                 for key, value in data.items():
#                     if key != 'weight':
#                         traffic_network.update_edge_attribute(u, v, key, value)
        
#         return {
#             "timestamp": datetime.now().isoformat(),
#             "optimization_metrics": metrics,
#             "nodes_optimized": len(optimized_graph.nodes()),
#             "edges_optimized": len(optimized_graph.edges()),
#             "signalization_recommendations": sum(1 for _, _, d in optimized_graph.edges(data=True)
#                                             if d.get('needs_signal', False))
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Network optimization error: {str(e)}")
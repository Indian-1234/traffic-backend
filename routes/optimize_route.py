# from typing import Any, Dict
# from fastapi import APIRouter, HTTPException
# from datetime import datetime
# import asyncio
# import networkx as nx
# from concurrent.futures import ThreadPoolExecutor
# import multiprocessing
# import functools

# from quantum.traffic_opt import SynchronousTrafficNetworkOptimizer
# from traffic_graph import TrafficGraph

# router = APIRouter()
# traffic_network = TrafficGraph()

# # Use a thread pool for heavy computation
# thread_pool = ThreadPoolExecutor(max_workers=1)

# # Create a synchronized version of the optimization function
# def run_optimization_in_process(graph_data):
#     """Run optimization in a completely separate process"""
#     # Reconstruct the graph from serialized data
#     graph = nx.node_link_graph(graph_data)
    
#     # Create a fresh optimizer in this process
#     optimizer = SynchronousTrafficNetworkOptimizer(n_qubits=8)
    
#     # We need to create a new event loop for this process
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
    
#     try:
#         # Run the optimization
#         result = loop.run_until_complete(optimizer.optimize_traffic_network(graph))
        
#         # Convert the graph to a serializable format and return with metrics
#         optimized_graph_data = nx.node_link_data(result[0])
#         return (optimized_graph_data, result[1])
#     finally:
#         loop.close()

# @router.post("/optimize-network", response_model=Dict[str, Any])
# async def optimize_network():
#     """
#     Optimize the traffic network using quantum computing.
#     This endpoint runs the optimization in a completely separate process.
#     """
#     try:
#         # Convert the graph to a serializable format
#         graph_data = nx.node_link_data(traffic_network.graph)
        
#         # Create a process pool for this specific task
#         with multiprocessing.Pool(processes=1) as pool:
#             # Run the optimization in a separate process
#             result = await asyncio.get_event_loop().run_in_executor(
#                 None,  # Use default executor
#                 pool.apply,  # Use pool.apply which blocks until the function returns
#                 run_optimization_in_process, 
#                 (graph_data,)
#             )
        
#         # Unpack the results
#         optimized_graph_data, metrics = result
        
#         # Convert back to NetworkX graph
#         optimized_graph = nx.node_link_graph(optimized_graph_data)
        
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
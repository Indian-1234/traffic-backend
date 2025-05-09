import networkx as nx
import numpy as np

class TrafficGraph:
    """Graph representation of traffic network"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id, **attributes):
        """Add a node (intersection or point) to the traffic network"""
        self.graph.add_node(node_id, **attributes)
        
    def add_edge(self, source, target, **attributes):
        """Add an edge (road segment) to the traffic network"""
        self.graph.add_edge(source, target, **attributes)
        
    def update_edge_weight(self, source, target, weight):
        """Update the weight (travel time/congestion) of an edge"""
        if self.graph.has_edge(source, target):
            self.graph[source][target]['weight'] = weight
            
    def compute_shortest_path(self, source, target):
        """Find the shortest path between two nodes"""
        try:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            distance = nx.shortest_path_length(self.graph, source, target, weight='weight')
            return path, distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float('inf')
            
    def compute_eigenvalues(self):
        """Compute the eigenvalues of the graph's adjacency matrix"""
        # Create adjacency matrix from graph
        adj_matrix = nx.adjacency_matrix(self.graph).todense()
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(adj_matrix)
        return eigenvalues
        
    def extract_subgraph(self, center_node, radius):
        """Extract a subgraph centered on a specific node"""
        nodes = nx.ego_graph(self.graph, center_node, radius=radius)
        return self.graph.subgraph(nodes)
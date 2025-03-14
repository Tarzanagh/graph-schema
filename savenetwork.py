# Save enhanced graph to file
def save_enhanced_graph(graph, filename):
    """Simple function to save a NetworkX graph to file"""
    import networkx as nx
    import pickle
    
    # Save using pickle (preserves all attributes)
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"Graph saved to {filename}")
    
    # To load later:
    # with open(filename, 'rb') as f:
    #     loaded_graph = pickle.load(f)

def save_graph(graph, output_file="schema_graph.json"):
    """
    Save the graph to a file for later use.
    
    Args:
        graph: NetworkX graph
        output_file: Path to output file
    """
    import json
    import numpy as np
    
    # Function to make values JSON serializable
    def make_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.float32) or isinstance(value, np.float64):
            return float(value)
        elif isinstance(value, np.int32) or isinstance(value, np.int64):
            return int(value)
        elif isinstance(value, set):
            return list(value)
        elif hasattr(value, 'tolist'):  # For other numpy types
            return value.tolist()
        else:
            return value
    
    # Convert graph to a serializable format
    serializable_graph = {
        'nodes': [],
        'edges': []
    }
    
    for node, attrs in graph.nodes(data=True):
        node_data = {'id': node}
        # Make all attribute values serializable
        for key, value in attrs.items():
            node_data[key] = make_serializable(value)
        serializable_graph['nodes'].append(node_data)
    
    for source, target, attrs in graph.edges(data=True):
        edge_data = {'source': source, 'target': target}
        # Make all attribute values serializable
        for key, value in attrs.items():
            edge_data[key] = make_serializable(value)
        serializable_graph['edges'].append(edge_data)
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(serializable_graph, f, indent=2)
        print(f"Graph saved to {output_file}")
    except TypeError as e:
        print(f"Error: {e}")
        print("Attempting to save with more aggressive serialization...")
        
        # If still failing, try a more aggressive approach
        for nodes in serializable_graph['nodes']:
            for key in list(nodes.keys()):
                if key != 'id':
                    try:
                        # Test if the value is JSON serializable
                        json.dumps(nodes[key])
                    except:
                        # If not, convert to string representation
                        nodes[key] = str(nodes[key])
        
        for edges in serializable_graph['edges']:
            for key in list(edges.keys()):
                if key not in ['source', 'target']:
                    try:
                        # Test if the value is JSON serializable
                        json.dumps(edges[key])
                    except:
                        # If not, convert to string representation
                        edges[key] = str(edges[key])
        
        # Try saving again
        with open(output_file, 'w') as f:
            json.dump(serializable_graph, f, indent=2)
        print(f"Graph saved to {output_file} after conversion")

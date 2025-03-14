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

# Usage:
save_enhanced_graph(enhanced_graph, "enhanced_graph.pkl")

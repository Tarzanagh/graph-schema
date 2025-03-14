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


 def save_graph(graph, output_file = "schema_graph.json"):
        """
        Save the graph to a file for later use.
        
        Args:
            graph: NetworkX graph
            output_file: Path to output file
        """

        
        # Convert graph to a serializable format
        serializable_graph = {
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            serializable_graph['nodes'].append(node_data)
        
        for source, target, attrs in graph.edges(data=True):
            edge_data = {'source': source, 'target': target}
            edge_data.update(attrs)
            serializable_graph['edges'].append(edge_data)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(serializable_graph, f, indent=2)
        
        print(f"Graph saved to {output_file}")

def save_graph_visualization(json_file, output_file):
    """
    Load graph from JSON and save a basic visualization as PNG or PDF.
    
    Args:
        json_file: Path to the JSON file with graph data
        output_file: Path for the output image (should end with .png or .pdf)
    """
    import json
    import networkx as nx
    import matplotlib
    # Use a different backend that handles complex objects better
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Load graph from JSON
    try:
        with open(json_file, 'r') as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes
    for node_data in graph_data['nodes']:
        node_id = node_data.pop('id')
        G.add_node(node_id)
    
    # Add edges
    for edge_data in graph_data['edges']:
        source = edge_data['source']
        target = edge_data['target']
        G.add_edge(source, target)
    
    # Create a new figure with a very simple setup
    plt.figure(figsize=(16, 12), dpi=100)
    plt.title("Schema Graph")
    
    # Generate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Basic drawing with minimal styling
    nx.draw(G, pos, 
            node_size=200, 
            node_color='lightblue',
            with_labels=True, 
            font_size=8)
    
    # Turn off axes
    plt.axis('off')
    
    # Save the figure
    try:
        plt.savefig(output_file, format=output_file.split('.')[-1])
        print(f"Graph visualization saved to {output_file}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
        
    # Close the figure to free memory
    plt.close()

# Usage:
# save_graph_visualization("schema_graph.json", "graph.png")
# or
# save_graph_visualization("schema_graph.json", "graph.pdf")
#
def save_graph_with_graphviz(json_file, output_file):
    """
    Save graph visualization using PyGraphviz for more reliable output.
    
    Args:
        json_file: Path to JSON file with graph data
        output_file: Path for output image (png or pdf)
    """
    import json
    import networkx as nx
    
    # Load graph data
    with open(json_file, 'r') as f:
        graph_data = json.load(f)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges
    for node_data in graph_data['nodes']:
        node_id = node_data.pop('id')
        G.add_node(node_id)
    
    for edge_data in graph_data['edges']:
        source = edge_data['source']
        target = edge_data['target']
        G.add_edge(source, target)
    
    # Convert to PyGraphviz format
    try:
        A = nx.nx_agraph.to_agraph(G)
        
        # Set graph attributes
        A.graph_attr.update(size="12,9", ratio="fill", overlap="false", splines="true")
        A.node_attr.update(shape="ellipse", style="filled", fillcolor="lightblue")
        
        # Save to file
        A.draw(output_file, prog="fdp")  # fdp is a force-directed layout algorithm
        print(f"Graph visualization saved to {output_file}")
    except Exception as e:
        print(f"Error saving with PyGraphviz: {e}")
        print("Make sure PyGraphviz and Graphviz are installed:")
        print("pip install pygraphviz")
        print("On Ubuntu: apt-get install graphviz libgraphviz-dev")
        print("On macOS: brew install graphviz")
        print("On Windows: Download from https://graphviz.org/download/")

# Usage:
# save_graph_with_graphviz("schema_graph.json", "graph.png")
# or
# save_graph_with_graphviz("schema_graph.json", "graph.pdf")


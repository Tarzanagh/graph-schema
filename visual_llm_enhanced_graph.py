def load_and_visualize_graph(json_file="schema_graph.json", output_file=None, show=True):
    """
    Load graph from a JSON file and visualize it.
    
    Args:
        json_file: Path to the JSON file with graph data
        output_file: Optional path to save the visualization
        show: Whether to display the visualization
    """
    import json
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load graph from JSON
    with open(json_file, 'r') as f:
        graph_data = json.load(f)
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_data in graph_data['nodes']:
        node_id = node_data.pop('id')
        G.add_node(node_id, **node_data)
    
    # Add edges with attributes
    for edge_data in graph_data['edges']:
        source = edge_data.pop('source')
        target = edge_data.pop('target')
        G.add_edge(source, target, **edge_data)
    
    # Create a figure
    plt.figure(figsize=(16, 12))
    
    # Extract node types for coloring
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        if node_type == 'table':
            node_colors.append('skyblue')
        elif node_type == 'column':
            if G.nodes[node].get('is_primary_key', False):
                node_colors.append('gold')
            elif G.nodes[node].get('is_foreign_key', False):
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        else:
            node_colors.append('white')
    
    # Set node sizes based on importance/degree
    node_sizes = []
    for node in G.nodes():
        size = 300 * (1 + G.degree(node) / 10)
        node_sizes.append(size)
    
    # Create edge weights and colors
    edge_weights = []
    edge_colors = []
    for u, v, data in G.edges(data=True):
        # Get weight from edge data or default to 1.0
        weight = data.get('weight', 1.0)
        if isinstance(weight, str):
            try:
                weight = float(weight)
            except:
                weight = 1.0
        edge_weights.append(weight * 2)  # Scale for visibility
        
        # Set color based on relationship type
        rel_type = data.get('relationship_type', '')
        if rel_type == 'foreign_key':
            edge_colors.append('green')
        elif rel_type == 'table_column':
            edge_colors.append('blue')
        elif 'diffused_weight' in data:
            edge_colors.append('red')
        else:
            edge_colors.append('gray')
    
    # Generate layout
    try:
        # Try spring layout first
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except:
        try:
            # Fall back to kamada_kawai
            pos = nx.kamada_kawai_layout(G)
        except:
            # Last resort: circular layout
            pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_weights,
        edge_color=edge_colors,
        alpha=0.6,
        arrows=False
    )
    
    # Create node labels
    node_labels = {}
    for node in G.nodes():
        # Shorten long node names
        if '.' in node:
            table, col = node.split('.')
            node_labels[node] = f"{table[:5]}...{col}" if len(table) > 5 else node
        else:
            node_labels[node] = node if len(node) <= 10 else node[:7] + "..."
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=8,
        font_family='sans-serif'
    )
    
    # Add a title
    plt.title("Schema Graph Visualization", fontsize=16)
    
    # Add a legend
    table_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='Table')
    column_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=15, label='Column')
    pk_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=15, label='Primary Key')
    fk_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Foreign Key')
    
    fk_edge = plt.Line2D([0], [0], color='green', linewidth=3, label='Foreign Key Relationship')
    table_col_edge = plt.Line2D([0], [0], color='blue', linewidth=3, label='Table-Column Relationship')
    diffused_edge = plt.Line2D([0], [0], color='red', linewidth=3, label='Diffused Connection')
    other_edge = plt.Line2D([0], [0], color='gray', linewidth=3, label='Other Relationship')
    
    plt.legend(handles=[table_patch, column_patch, pk_patch, fk_patch, 
                      fk_edge, table_col_edge, diffused_edge, other_edge], 
             loc='best', fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return G  # Return the graph in case it's needed for further analysis

# Example usage:
# load_and_visualize_graph("enhanced_schema_graph.json", "visualization.png")

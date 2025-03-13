def visualize_edge_weights(graph, title="Graph Edge Weights", save_path=None):
    """
    Create a heatmap visualization of edge weights in a graph.
    
    Args:
        graph: NetworkX graph with weighted edges
        title: Title for the plot
        save_path: Optional path to save the visualization
    
    Returns:
        The matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    
    # Get all nodes
    nodes = list(graph.nodes())
    
    # Create an adjacency matrix with edge weights
    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    
    # Map node IDs to matrix indices
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Fill the adjacency matrix with edge weights
    for i, j in graph.edges():
        idx_i = node_to_idx[i]
        idx_j = node_to_idx[j]
        weight = graph[i][j].get('weight', graph[i][j].get('semantic_weight', 1.0))
        adj_matrix[idx_i, idx_j] = weight
        adj_matrix[idx_j, idx_i] = weight  # For undirected graphs
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    im = plt.imshow(adj_matrix, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, label='Edge Weight')
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    
    # Only show a subset of node labels if there are many nodes
    if n <= 50:
        plt.xticks(range(n), nodes, rotation=90, fontsize=8)
        plt.yticks(range(n), nodes, fontsize=8)
    else:
        # Show some labels at regular intervals
        step = max(1, n // 20)
        subset_indices = list(range(0, n, step))
        subset_labels = [nodes[i] for i in subset_indices]
        plt.xticks(subset_indices, subset_labels, rotation=90, fontsize=8)
        plt.yticks(subset_indices, subset_labels, fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Edge weight heatmap saved to {save_path}")
    
    return plt.gcf()

def create_table_column_edge_heatmap(graph, title="Table-Column Edge Weights", save_path=None):
    """
    Create a heatmap specifically for table-column relationships.
    
    Args:
        graph: NetworkX graph with tables and columns
        title: Title for the plot
        save_path: Optional path to save the visualization
    
    Returns:
        The matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get tables and columns
    tables = [node for node in graph.nodes() if graph.nodes[node].get('type') == 'table']
    columns = [node for node in graph.nodes() if graph.nodes[node].get('type') == 'column']
    
    # Sort tables and columns alphabetically
    tables.sort()
    columns.sort()
    
    # Create a matrix for the heatmap
    t = len(tables)
    c = len(columns)
    matrix = np.zeros((t, c))
    
    # Fill the matrix with edge weights
    for i, table in enumerate(tables):
        for j, column in enumerate(columns):
            # Check if there's an edge between table and column
            if graph.has_edge(table, column):
                weight = graph[table][column].get('weight', graph[table][column].get('semantic_weight', 1.0))
                matrix[i, j] = weight
            else:
                matrix[i, j] = 0
    
    # Create the figure
    plt.figure(figsize=(max(12, c//4), max(10, t//4)))
    im = plt.imshow(matrix, cmap='viridis', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, label='Edge Weight')
    
    # Set title and labels
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Tables')
    
    # Add table names as y-tick labels
    plt.yticks(range(t), tables, fontsize=8)
    
    # For columns, we might need to rotate them for readability
    # Use abbreviated column names for readability
    abbreviated_columns = [col.split('.')[-1] if '.' in col else col for col in columns]
    plt.xticks(range(c), abbreviated_columns, rotation=90, fontsize=6)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table-column edge weight heatmap saved to {save_path}")
    
    return plt.gcf()

def visualize_graph_with_edge_weights(graph, title="Graph with Edge Weights", save_path=None):
    """
    Visualize a graph where edge thickness and color represent edge weights.
    
    Args:
        graph: NetworkX graph with weighted edges
        title: Title for the plot
        save_path: Optional path to save the visualization
    
    Returns:
        The matplotlib figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # Create the figure
    plt.figure(figsize=(14, 12))
    
    # Create a layout for the graph
    # For database schemas, a hierarchical layout often works better
    try:
        pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    except:
        # Fall back to spring layout if graphviz is not available
        pos = nx.spring_layout(graph, seed=42)
    
    # Get node types for coloring
    node_types = {}
    for node in graph.nodes():
        node_types[node] = graph.nodes[node].get('type', 'unknown')
    
    # Color nodes by type
    node_colors = []
    for node in graph.nodes():
        if node_types[node] == 'table':
            node_colors.append('skyblue')
        elif node_types[node] == 'column':
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightgray')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, 
        node_color=node_colors,
        node_size=300
    )
    
    # Get edge weights
    edge_weights = {}
    for u, v in graph.edges():
        edge_weights[(u, v)] = graph[u][v].get('weight', graph[u][v].get('semantic_weight', 1.0))
    
    # Normalize edge weights for visualization
    if edge_weights:
        max_weight = max(edge_weights.values())
        min_weight = min(edge_weights.values())
        weight_range = max_weight - min_weight
        
        if weight_range > 0:
            normalized_weights = {e: (w - min_weight) / weight_range * 3 + 0.5 for e, w in edge_weights.items()}
        else:
            normalized_weights = {e: 1.0 for e in edge_weights}
            
        # Draw edges with thickness based on weight
        for (u, v), weight in normalized_weights.items():
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v)],
                width=weight,
                alpha=min(1.0, weight * 0.7)
            )
    else:
        # Draw all edges with the same thickness if no weights
        nx.draw_networkx_edges(graph, pos, width=1.0)
    
    # Draw labels for tables and important nodes
    labels = {}
    for node in graph.nodes():
        if node_types[node] == 'table' or node_types[node] == 'unknown':
            labels[node] = node
        else:
            # For columns, just show the column name part
            labels[node] = node.split('.')[-1] if '.' in node else node
    
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    
    # Add a legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Table'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Column')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {save_path}")
    
    return plt.gcf()

# Example usage:
"""
# To visualize edge weights as a heatmap
fig1 = visualize_edge_weights(graph, save_path="edge_weights_heatmap.png")

# To visualize table-column relationships
fig2 = create_table_column_edge_heatmap(graph, save_path="table_column_heatmap.png")

# To visualize the graph with edge weights represented by line thickness
fig3 = visualize_graph_with_edge_weights(graph, save_path="weighted_graph.png")

plt.show()
"""

def create_importance_heatmap(graph, node_importance, title="Node Importance Heatmap"):
    """
    Create a heatmap visualization of node importance from flow diffusion.
    
    Args:
        graph: The graph structure
        node_importance: Dictionary mapping node_id to importance score
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a position layout for the graph
    # You can use different layouts like spring_layout, kamada_kawai_layout, etc.
    pos = nx.spring_layout(graph, seed=42)  # Fixed seed for reproducibility
    
    # Get nodes and their importance scores
    nodes = list(graph.nodes())
    node_types = [graph.nodes[n].get('type', 'unknown') for n in nodes]
    importance = [node_importance.get(n, 0.0) for n in nodes]
    
    # Create a custom colormap (blue gradient)
    cmap = LinearSegmentedColormap.from_list("importance_cmap", ["#FFFFFF", "#0571B0", "#08306B"])
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Draw the graph
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    
    # Draw nodes with color intensity based on importance
    # Normalize importance scores for better visualization
    max_importance = max(importance) if importance else 1.0
    normalized_importance = [i/max_importance for i in importance]
    
    # Draw nodes by type with different shapes and colors
    table_nodes = [n for i, n in enumerate(nodes) if node_types[i] == 'table']
    column_nodes = [n for i, n in enumerate(nodes) if node_types[i] == 'column']
    other_nodes = [n for i, n in enumerate(nodes) if node_types[i] not in ['table', 'column']]
    
    table_importance = [node_importance.get(n, 0.0)/max_importance for n in table_nodes]
    column_importance = [node_importance.get(n, 0.0)/max_importance for n in column_nodes]
    other_importance = [node_importance.get(n, 0.0)/max_importance for n in other_nodes]
    
    # Draw tables as squares
    nx.draw_networkx_nodes(graph, pos, nodelist=table_nodes, node_color=table_importance, 
                         cmap=cmap, node_shape='s', node_size=300)
    
    # Draw columns as circles
    nx.draw_networkx_nodes(graph, pos, nodelist=column_nodes, node_color=column_importance, 
                         cmap=cmap, node_shape='o', node_size=200)
    
    # Draw other nodes as triangles
    if other_nodes:
        nx.draw_networkx_nodes(graph, pos, nodelist=other_nodes, node_color=other_importance, 
                             cmap=cmap, node_shape='^', node_size=150)
    
    # Add labels for important nodes (top N by importance)
    top_n = 20  # Number of top nodes to label
    top_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = {node: node for node, score in top_nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    
    # Add a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Importance Score')
    
    # Add legend for node shapes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Table'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Column')
    ]
    if other_nodes:
        legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                                    markersize=10, label='Other'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    plt.axis('off')
    
    # Return the plot for display or saving
    return plt

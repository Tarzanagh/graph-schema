def visualize_enhanced_graph(graph, output_file=None, show=True):
    """
    Visualize the enhanced schema graph with weights from semantic analysis.
    
    Args:
        graph: NetworkX graph with enhanced weights
        output_file: Optional path to save the visualization
        show: Whether to display the visualization
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # Create a figure
    plt.figure(figsize=(16, 12))
    
    # Extract node types
    node_types = {}
    for node in graph.nodes():
        node_types[node] = graph.nodes[node].get('type', 'unknown')
    
    # Generate positions - use a layout that works well for hierarchical data
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    
    # Get node sizes based on importance
    node_sizes = []
    for node in graph.nodes():
        # Size based on degree
        size = 300 * (1 + graph.degree(node) / 10)
        node_sizes.append(size)
    
    # Get edge weights for line thickness
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        # Use the semantic weight or default to 1.0
        weight = data.get('weight', 1.0)
        edge_weights.append(weight * 2)  # Scale for visibility
    
    # Color nodes by type
    node_colors = []
    for node in graph.nodes():
        node_type = node_types[node]
        if node_type == 'table':
            node_colors.append('skyblue')
        elif node_type == 'column':
            # Check if it's a primary or foreign key
            if graph.nodes[node].get('is_primary_key', False):
                node_colors.append('gold')
            elif graph.nodes[node].get('is_foreign_key', False):
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        else:
            node_colors.append('white')
    
    # Draw the nodes
    nx.draw_networkx_nodes(
        graph, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Create edge colors based on relationship type
    edge_colors = []
    for u, v, data in graph.edges(data=True):
        rel_type = data.get('relationship_type', '')
        if rel_type == 'foreign_key':
            edge_colors.append('green')
        elif rel_type == 'table_column':
            edge_colors.append('blue')
        elif 'diffused_weight' in data:
            # Edges that were enhanced by diffusion
            edge_colors.append('red')
        else:
            edge_colors.append('gray')
    
    # Draw the edges
    nx.draw_networkx_edges(
        graph, pos, 
        width=edge_weights,
        edge_color=edge_colors,
        alpha=0.6,
        arrows=False
    )
    
    # Create node labels
    node_labels = {}
    for node in graph.nodes():
        # Shorten long node names
        if '.' in node:
            table, col = node.split('.')
            node_labels[node] = f"{table[:5]}...{col}" if len(table) > 5 else node
        else:
            node_labels[node] = node if len(node) <= 10 else node[:7] + "..."
    
    # Draw the labels
    nx.draw_networkx_labels(
        graph, pos,
        labels=node_labels,
        font_size=8,
        font_family='sans-serif'
    )
    
    # Add a title
    plt.title("Enhanced Schema Graph with Semantic Weights", fontsize=16)
    
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

def visualize_graph_communities(graph, output_file=None, show=True):
    """
    Visualize the enhanced schema graph with community detection.
    
    Args:
        graph: NetworkX graph with enhanced weights
        output_file: Optional path to save the visualization
        show: Whether to display the visualization
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from community import community_louvain
    
    # Create a figure
    plt.figure(figsize=(16, 12))
    
    # Detect communities using the Louvain method
    # Convert edge weights to positive values as required by community detection
    graph_for_community = graph.copy()
    for u, v, data in graph_for_community.edges(data=True):
        if 'weight' in data:
            # Ensure weight is positive
            graph_for_community[u][v]['weight'] = abs(data['weight'])
    
    # Apply community detection
    partition = community_louvain.best_partition(graph_for_community)
    
    # Get unique communities
    communities = set(partition.values())
    
    # Create a color map for communities
    cmap = plt.cm.get_cmap('tab20', len(communities))
    
    # Generate positions using a force-directed layout
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    
    # Get node sizes based on degree
    node_sizes = []
    for node in graph.nodes():
        size = 300 * (1 + graph.degree(node) / 10)
        node_sizes.append(size)
    
    # Get edge weights for line thickness
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        edge_weights.append(weight * 2)
    
    # Draw nodes colored by community
    for comm in communities:
        # Get nodes in this community
        comm_nodes = [node for node in graph.nodes() if partition[node] == comm]
        
        # Draw nodes for this community
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=comm_nodes,
            node_size=[node_sizes[i] for i, node in enumerate(graph.nodes()) if node in comm_nodes],
            node_color=[cmap(comm)] * len(comm_nodes),
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            label=f"Community {comm}"
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        width=edge_weights,
        alpha=0.3,
        arrows=False
    )
    
    # Create node labels
    node_labels = {}
    for node in graph.nodes():
        if '.' in node:
            table, col = node.split('.')
            node_labels[node] = f"{table[:5]}...{col}" if len(table) > 5 else node
        else:
            node_labels[node] = node if len(node) <= 10 else node[:7] + "..."
    
    # Draw labels
    nx.draw_networkx_labels(
        graph, pos,
        labels=node_labels,
        font_size=8,
        font_family='sans-serif'
    )
    
    # Add a title
    plt.title("Schema Graph Communities", fontsize=16)
    
    # Add a legend for communities
    comm_patches = []
    for comm in communities:
        patch = plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=cmap(comm), markersize=15, 
                           label=f"Community {comm}")
        comm_patches.append(patch)
    
    plt.legend(handles=comm_patches, loc='best', fontsize=10)
    
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

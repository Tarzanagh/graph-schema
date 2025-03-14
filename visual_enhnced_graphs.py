def visualize_node_similarity_graph(graph, filename="node_similarity_graph.png"):
    """
    Visualize the schema graph with weights based only on node similarities
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    plt.figure(figsize=(16, 14))
    
    # Create a copy of the graph for layout calculation
    layout_graph = graph.copy()
    
    # Replace weights with node_similarity values or 1.0
    for u, v, data in layout_graph.edges(data=True):
        if 'node_similarity' in data:
            weight = data['node_similarity']
            if isinstance(weight, (int, float)):
                layout_graph[u][v]['weight'] = weight
            else:
                layout_graph[u][v]['weight'] = 1.0
        else:
            layout_graph[u][v]['weight'] = 1.0
    
    # Create positions using a spring layout
    pos = nx.spring_layout(layout_graph, k=0.3, iterations=50, seed=42)
    
    # Identify node types
    table_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get('type') == 'table']
    pk_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                      if attrs.get('type') == 'column' and attrs.get('is_primary_key', False)]
    fk_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                      if attrs.get('type') == 'column' and attrs.get('is_foreign_key', False)]
    regular_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                          if attrs.get('type') == 'column' 
                          and not attrs.get('is_primary_key', False)
                          and not attrs.get('is_foreign_key', False)]
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=table_nodes, node_size=2000, 
                          node_color="lightblue", edgecolors='black', alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=pk_column_nodes, node_size=800, 
                          node_color="gold", edgecolors='black', alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, nodelist=fk_column_nodes, node_size=800, 
                          node_color="lightgreen", edgecolors='black', alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, nodelist=regular_column_nodes, node_size=800, 
                          node_color="lightgray", edgecolors='black', alpha=0.7)
    
    # Create edge lists and widths based on node_similarity
    edges = []
    edge_widths = []
    max_weight = 1.0
    
    for u, v, data in graph.edges(data=True):
        edges.append((u, v))
        
        # Extract node similarity weight
        weight = 1.0
        if 'node_similarity' in data:
            ns = data['node_similarity']
            if isinstance(ns, (int, float)):
                weight = ns
            elif isinstance(ns, str):
                try:
                    weight = float(ns)
                except:
                    pass
            elif isinstance(ns, (list, tuple, np.ndarray)):
                try:
                    weight = np.mean(ns)
                except:
                    pass
        
        max_weight = max(max_weight, weight)
        edge_widths.append(weight)
    
    # Normalize edge widths
    normalized_widths = [1.0 + 4.0 * (w / max_weight) for w in edge_widths]
    
    # Draw edges with normalized widths
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=normalized_widths,
                          edge_color='blue', alpha=0.6, arrows=False)
    
    # Add labels
    table_labels = {node: node for node in table_nodes}
    column_labels = {}
    for node in pk_column_nodes + fk_column_nodes + regular_column_nodes:
        if '.' in node:
            column_labels[node] = node.split('.')[-1]  
        else:
            column_labels[node] = node
    
    nx.draw_networkx_labels(graph, pos, labels=table_labels, font_size=12, font_weight='bold')
    nx.draw_networkx_labels(graph, pos, labels=column_labels, font_size=8)
    
    # Add legend
    plt.plot([0], [0], 'o', markersize=15, color='lightblue', label='Tables')
    plt.plot([0], [0], 'o', markersize=10, color='gold', label='Primary Key Columns')
    plt.plot([0], [0], 'o', markersize=10, color='lightgreen', label='Foreign Key Columns')
    plt.plot([0], [0], 'o', markersize=10, color='lightgray', label='Regular Columns')
    plt.plot([0], [0], '-', color='blue', linewidth=1.0, label='Low Node Similarity')
    plt.plot([0], [0], '-', color='blue', linewidth=5.0, label='High Node Similarity')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.axis("off")
    plt.title("Schema Graph with Node Similarities", fontsize=16)
    plt.tight_layout()
    
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Node similarity graph visualization saved to '{filename}'")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.close()

def visualize_edge_similarity_graph(graph, filename="edge_similarity_graph.png"):
    """
    Visualize the schema graph with weights based only on edge similarities
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    plt.figure(figsize=(16, 14))
    
    # Create a copy of the graph for layout calculation
    layout_graph = graph.copy()
    
    # Replace weights with edge_similarity values or 1.0
    for u, v, data in layout_graph.edges(data=True):
        if 'edge_similarity' in data:
            weight = data['edge_similarity']
            if isinstance(weight, (int, float)):
                layout_graph[u][v]['weight'] = weight
            else:
                layout_graph[u][v]['weight'] = 1.0
        else:
            layout_graph[u][v]['weight'] = 1.0
    
    # Create positions using a spring layout
    pos = nx.spring_layout(layout_graph, k=0.3, iterations=50, seed=42)
    
    # Identify node types
    table_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get('type') == 'table']
    pk_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                      if attrs.get('type') == 'column' and attrs.get('is_primary_key', False)]
    fk_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                      if attrs.get('type') == 'column' and attrs.get('is_foreign_key', False)]
    regular_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                          if attrs.get('type') == 'column' 
                          and not attrs.get('is_primary_key', False)
                          and not attrs.get('is_foreign_key', False)]
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=table_nodes, node_size=2000, 
                          node_color="lightblue", edgecolors='black', alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=pk_column_nodes, node_size=800, 
                          node_color="gold", edgecolors='black', alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, nodelist=fk_column_nodes, node_size=800, 
                          node_color="lightgreen", edgecolors='black', alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, nodelist=regular_column_nodes, node_size=800, 
                          node_color="lightgray", edgecolors='black', alpha=0.7)
    
    # Create edge lists and widths based on edge_similarity
    edges = []
    edge_widths = []
    max_weight = 1.0
    
    for u, v, data in graph.edges(data=True):
        edges.append((u, v))
        
        # Extract edge similarity weight
        weight = 1.0
        if 'edge_similarity' in data:
            es = data['edge_similarity']
            if isinstance(es, (int, float)):
                weight = es
            elif isinstance(es, str):
                try:
                    weight = float(es)
                except:
                    pass
            elif isinstance(es, (list, tuple, np.ndarray)):
                try:
                    weight = np.mean(es)
                except:
                    pass
        
        max_weight = max(max_weight, weight)
        edge_widths.append(weight)
    
    # Normalize edge widths
    normalized_widths = [1.0 + 4.0 * (w / max_weight) for w in edge_widths]
    
    # Draw edges with normalized widths
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=normalized_widths,
                          edge_color='green', alpha=0.6, arrows=False)
    
    # Add labels
    table_labels = {node: node for node in table_nodes}
    column_labels = {}
    for node in pk_column_nodes + fk_column_nodes + regular_column_nodes:
        if '.' in node:
            column_labels[node] = node.split('.')[-1]  
        else:
            column_labels[node] = node
    
    nx.draw_networkx_labels(graph, pos, labels=table_labels, font_size=12, font_weight='bold')
    nx.draw_networkx_labels(graph, pos, labels=column_labels, font_size=8)
    
    # Add legend
    plt.plot([0], [0], 'o', markersize=15, color='lightblue', label='Tables')
    plt.plot([0], [0], 'o', markersize=10, color='gold', label='Primary Key Columns')
    plt.plot([0], [0], 'o', markersize=10, color='lightgreen', label='Foreign Key Columns')
    plt.plot([0], [0], 'o', markersize=10, color='lightgray', label='Regular Columns')
    plt.plot([0], [0], '-', color='green', linewidth=1.0, label='Low Edge Similarity')
    plt.plot([0], [0], '-', color='green', linewidth=5.0, label='High Edge Similarity')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.axis("off")
    plt.title("Schema Graph with Edge Similarities", fontsize=16)
    plt.tight_layout()
    
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Edge similarity graph visualization saved to '{filename}'")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.close()

def visualize_semantic_weight_graph(graph, filename="semantic_weight_graph.png"):
    """
    Visualize the schema graph with weights based on semantic weights
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    plt.figure(figsize=(16, 14))
    
    # Create a copy of the graph for layout calculation
    layout_graph = graph.copy()
    
    # Replace weights with semantic_weight values or 1.0
    for u, v, data in layout_graph.edges(data=True):
        if 'semantic_weight' in data:
            weight = data['semantic_weight']
            if isinstance(weight, (int, float)):
                layout_graph[u][v]['weight'] = weight
            else:
                layout_graph[u][v]['weight'] = 1.0
        elif 'weight' in data:
            weight = data['weight']
            if isinstance(weight, (int, float)):
                layout_graph[u][v]['weight'] = weight
            else:
                layout_graph[u][v]['weight'] = 1.0
        else:
            layout_graph[u][v]['weight'] = 1.0
    
    # Create positions using a spring layout
    pos = nx.spring_layout(layout_graph, k=0.3, iterations=50, seed=42)
    
    # Identify node types
    table_nodes = [node for node, attrs in graph.nodes(data=True) if attrs.get('type') == 'table']
    pk_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                      if attrs.get('type') == 'column' and attrs.get('is_primary_key', False)]
    fk_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                      if attrs.get('type') == 'column' and attrs.get('is_foreign_key', False)]
    regular_column_nodes = [node for node, attrs in graph.nodes(data=True) 
                          if attrs.get('type') == 'column' 
                          and not attrs.get('is_primary_key', False)
                          and not attrs.get('is_foreign_key', False)]
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=table_nodes, node_size=2000, 
                          node_color="lightblue", edgecolors='black', alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, nodelist=pk_column_nodes, node_size=800, 
                          node_color="gold", edgecolors='black', alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, nodelist=fk_column_nodes, node_size=800, 
                          node_color="lightgreen", edgecolors='black', alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, nodelist=regular_column_nodes, node_size=800, 
                          node_color="lightgray", edgecolors='black', alpha=0.7)
    
    # Create edge lists and widths based on semantic_weight
    edges = []
    edge_widths = []
    max_weight = 1.0
    
    for u, v, data in graph.edges(data=True):
        edges.append((u, v))
        
        # Extract semantic weight
        weight = 1.0
        if 'semantic_weight' in data:
            sw = data['semantic_weight']
            if isinstance(sw, (int, float)):
                weight = sw
            elif isinstance(sw, str):
                try:
                    weight = float(sw)
                except:
                    pass
            elif isinstance(sw, (list, tuple, np.ndarray)):
                try:
                    weight = np.mean(sw)
                except:
                    pass
        elif 'weight' in data:
            w = data['weight']
            if isinstance(w, (int, float)):
                weight = w
            elif isinstance(w, str):
                try:
                    weight = float(w)
                except:
                    pass
            elif isinstance(w, (list, tuple, np.ndarray)):
                try:
                    weight = np.mean(w)
                except:
                    pass
        
        max_weight = max(max_weight, weight)
        edge_widths.append(weight)
    
    # Normalize edge widths
    normalized_widths = [1.0 + 4.0 * (w / max_weight) for w in edge_widths]
    
    # Draw edges with normalized widths
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=normalized_widths,
                          edge_color='red', alpha=0.6, arrows=False)
    
    # Add labels
    table_labels = {node: node for node in table_nodes}
    column_labels = {}
    for node in pk_column_nodes + fk_column_nodes + regular_column_nodes:
        if '.' in node:
            column_labels[node] = node.split('.')[-1]  
        else:
            column_labels[node] = node
    
    nx.draw_networkx_labels(graph, pos, labels=table_labels, font_size=12, font_weight='bold')
    nx.draw_networkx_labels(graph, pos, labels=column_labels, font_size=8)
    
    # Add legend
    plt.plot([0], [0], 'o', markersize=15, color='lightblue', label='Tables')
    plt.plot([0], [0], 'o', markersize=10, color='gold', label='Primary Key Columns')
    plt.plot([0], [0], 'o', markersize=10, color='lightgreen', label='Foreign Key Columns')
    plt.plot([0], [0], 'o', markersize=10, color='lightgray', label='Regular Columns')
    plt.plot([0], [0], '-', color='red', linewidth=1.0, label='Low Semantic Weight')
    plt.plot([0], [0], '-', color='red', linewidth=5.0, label='High Semantic Weight')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.axis("off")
    plt.title("Schema Graph with Semantic Weights", fontsize=16)
    plt.tight_layout()
    
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Semantic weight graph visualization saved to '{filename}'")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.close()


# # After enhancing your graph
# enhanced_graph = llm_service.enhance_edge_semantics(graph, metadata, use_diffusion=True)

# # Visualize the graph in three different ways
# visualize_node_similarity_graph(enhanced_graph, "node_similarity_graph.png")
# visualize_edge_similarity_graph(enhanced_graph, "edge_similarity_graph.png")
# visualize_semantic_weight_graph(enhanced_graph, "semantic_weight_graph.png")

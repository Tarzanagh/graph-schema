def flow_diffusion(self, graph, seed_node, alpha=5.0):
    """
    Run the flow diffusion algorithm to find relevant nodes.
    
    Args:
        graph: The weighted graph representation
        seed_node: The starting node for diffusion
        alpha: Source mass multiplier
        
    Returns:
        Dictionary mapping nodes to their importance scores
    """
    # Initialize node embeddings and mass
    nodes = list(graph.nodes())
    x = {node_id: 0.0 for node_id in nodes}
    m = {node_id: 0.0 for node_id in nodes}
    
    # Set source mass
    if seed_node in m:
        m[seed_node] = alpha
    else:
        # Find best match if seed node not found
        best_match = None
        best_score = -1
        
        for node_id in nodes:
            if seed_node.lower() in node_id.lower():
                # Use degree as a proxy for importance if exact match not found
                score = graph.degree(node_id)
                if score > best_score:
                    best_match = node_id
                    best_score = score
        
        if best_match:
            m[best_match] = alpha
        else:
            # Fall back to the first node if no match
            m[nodes[0]] = alpha
    
    # Set sink capacities based on node type
    t = {}
    for node_id in nodes:
        if graph.nodes[node_id].get('type') == 'table':
            t[node_id] = 0.3  # Tables can hold more mass
        else:
            t[node_id] = 0.15  # Columns hold less mass
    
    # Run flow diffusion for specified iterations
    for iteration in range(self.max_iterations * 2):
        # Find nodes with excess mass
        overflow_nodes = [node for node, mass in m.items() if mass > t[node]]
        
        if not overflow_nodes:
            break
            
        # Pick an overflow node uniformly at random as per the original algorithm
        i = random.choice(overflow_nodes)
        
        # Calculate weighted degree
        w_i = sum(graph[i][j].get('weight', 1.0) for j in graph.neighbors(i))
        
        if w_i == 0:
            continue  # Skip isolated nodes
            
        # Update node embedding
        x[i] += (m[i] - t[i]) / w_i
        
        # Set excess mass to distribute
        excess = m[i] - t[i]
        m[i] = t[i]  # Set mass to capacity (important invariant)
        
        # Distribute excess mass to neighbors
        for j in graph.neighbors(i):
            # Find edge weight - use semantic_weight if available
            edge_weight = graph[i][j].get('semantic_weight', graph[i][j].get('weight', 1.0))
            
            # Update mass at neighbor
            m[j] += excess * edge_weight / w_i
    
    return x

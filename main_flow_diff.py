# 1. First, modify the flow_diffusion method to reduce sink capacity and increase iterations

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
    
    # Set source mass - increase the alpha for more mass
    if seed_node in m:
        m[seed_node] = alpha * 2.0  # Doubled source mass
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
            m[best_match] = alpha * 2.0  # Doubled source mass
        else:
            # Fall back to the first node if no match
            m[nodes[0]] = alpha * 2.0
    
    # Set sink capacities - REDUCE to 0.5 to force more spreading
    t = {node_id: 0.5 for node_id in nodes}  # Reduced from 1.0 to 0.5
    
    # Run flow diffusion for specified iterations - increase max iterations
    max_iterations = self.max_iterations * 2  # Double the iterations
    
    for _ in range(max_iterations):
        # Find nodes with excess mass
        overflow_nodes = [node for node, mass in m.items() if mass > t[node]]
        
        if not overflow_nodes:
            break
            
        # Pick an overflow node (deterministic for reproducibility)
        i = overflow_nodes[0]  
        
        # Calculate weighted degree
        w_i = sum(graph[i][j].get('weight', 1.0) for j in graph.neighbors(i))
        
        if w_i == 0:
            continue  # Skip isolated nodes
            
        # Update node embedding
        x[i] += (m[i] - t[i]) / w_i
        
        # Set excess mass to distribute
        excess = m[i] - t[i]
        m[i] = t[i]
        
        # Distribute excess mass to neighbors
        for j in graph.neighbors(i):
            # Find edge weight - use semantic_weight if available
            edge_weight = graph[i][j].get('semantic_weight', graph[i][j].get('weight', 1.0))
            
            # Update mass at neighbor
            m[j] += excess * edge_weight / w_i
    
    return x

# 2. Next, modify the extract_paths_from_subcluster method to get longer paths

def extract_paths_from_subcluster(self, graph, subcluster, limit=10, max_path_length=8):
    """
    Extract meaningful paths from a subcluster with longer paths.
    
    Args:
        graph: The graph
        subcluster: Set of node IDs in the subcluster
        limit: Maximum number of paths to return
        max_path_length: Maximum length of paths to find
        
    Returns:
        List of paths (as strings)
    """
    paths = []
    path_scores = {}
    
    # Find table nodes in the subcluster
    table_nodes = [node for node in subcluster if graph.nodes[node].get('type') == 'table']
    
    # Find column nodes in the subcluster
    column_nodes = [node for node in subcluster if graph.nodes[node].get('type') == 'column']
    
    # Find all nodes in the cluster
    all_nodes = list(subcluster)
    
    # Start paths from table nodes
    for start_node in table_nodes:
        # BFS to find paths through the subcluster
        visited = {start_node: 0}  # Node: depth visited at
        queue = [(start_node, [start_node], 0)]  # (node, path, length)
        
        while queue and len(paths) < limit * 3:  # Increased multiplier to find more candidate paths
            current, path, length = queue.pop(0)
            
            # If path reaches another table or max length, consider it complete
            if (length >= 3 and current in table_nodes and current != start_node) or length >= max_path_length:
                # Convert path to string format
                path_str = '.'.join(path)
                paths.append(path_str)
                # Score based on node importance, path length and number of tables
                num_tables = sum(1 for n in path if n in table_nodes)
                path_scores[path_str] = num_tables * 2.0 / (length * 0.8)  # Reward paths with more tables
                continue
            
            # Continue BFS
            for neighbor in graph.neighbors(current):
                # Allow visiting the same node again but at a deeper level to find more paths
                # This creates longer paths through the graph
                if neighbor in subcluster and (neighbor not in visited or visited[neighbor] > length + 2):
                    visited[neighbor] = length + 1
                    # Don't add duplicate nodes in sequence (avoid A->B->A->B loops)
                    if len(path) < 2 or neighbor != path[-2]:
                        queue.append((neighbor, path + [neighbor], length + 1))
    
    # If not enough paths yet, try starting from column nodes
    if len(paths) < limit:
        for start_node in column_nodes[:min(5, len(column_nodes))]:
            visited = {start_node: 0}
            queue = [(start_node, [start_node], 0)]
            
            while queue and len(paths) < limit * 3:
                current, path, length = queue.pop(0)
                
                if length >= 3:
                    path_str = '.'.join(path)
                    if path_str not in paths:
                        paths.append(path_str)
                        # Score based on path properties
                        num_tables = sum(1 for n in path if n in table_nodes)
                        num_columns = sum(1 for n in path if n in column_nodes)
                        path_scores[path_str] = (num_tables * 1.5 + num_columns) / (length * 0.7)
                
                if length < max_path_length:
                    for neighbor in graph.neighbors(current):
                        # Allow revisiting nodes at deeper levels
                        if neighbor in subcluster and (neighbor not in visited or visited[neighbor] > length + 2):
                            visited[neighbor] = length + 1
                            # Avoid simple back-and-forth
                            if len(path) < 2 or neighbor != path[-2]:
                                queue.append((neighbor, path + [neighbor], length + 1))
    
    # Try to find multi-hop table paths (table -> column -> table -> column)
    if len(paths) < limit * 2:
        for start_table in table_nodes:
            # Find paths connecting multiple tables
            visited = set()
            stack = [(start_table, [start_table], set([start_table]))]  # DFS with path and visited set
            
            while stack and len(paths) < limit * 3:
                current, path, path_set = stack.pop()
                
                # If we've found a path connecting 3+ tables, keep it
                tables_in_path = [n for n in path if n in table_nodes]
                if len(tables_in_path) >= 3 and len(path) >= 5:
                    path_str = '.'.join(path)
                    paths.append(path_str)
                    path_scores[path_str] = len(tables_in_path) * 3.0  # High score for multi-table paths
                
                # Continue exploration if path isn't too long
                if len(path) < max_path_length:
                    neighbors = list(graph.neighbors(current))
                    # Sort neighbors to prioritize unvisited tables
                    neighbors.sort(key=lambda n: (n not in path_set, n in table_nodes), reverse=True)
                    
                    for neighbor in neighbors:
                        if neighbor in subcluster:
                            # For longer paths, allow some revisits but avoid short cycles
                            new_path = path + [neighbor]
                            new_path_set = path_set.copy()
                            new_path_set.add(neighbor)
                            
                            # Check if adding this would create a cycle in recent nodes
                            should_add = True
                            if len(path) >= 2:
                                recent = path[-2:]
                                if neighbor in recent and (recent + [neighbor]).count(neighbor) > 1:
                                    should_add = False
                            
                            if should_add:
                                stack.append((neighbor, new_path, new_path_set))
    
    # Sort paths by score and take top ones
    scored_paths = [(p, path_scores.get(p, 0)) for p in paths]
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    
    # Prioritize longer paths in the final selection
    final_paths = []
    # First add some longer paths (length > 4)
    for p, score in scored_paths:
        path_nodes = p.split('.')
        if len(path_nodes) > 4 and len(final_paths) < limit * 0.7:
            final_paths.append(p)
    
    # Then fill with remaining best paths
    for p, _ in scored_paths:
        if p not in final_paths and len(final_paths) < limit:
            final_paths.append(p)
    
    # If still not enough paths, generate some
    if len(final_paths) < limit:
        # Create simple table.column paths
        for table in table_nodes:
            for col in graph.neighbors(table):
                if col in column_nodes and len(final_paths) < limit:
                    path = f"{table}.{col}"
                    if path not in final_paths:
                        final_paths.append(path)
    
    return final_paths

# 3. Finally, modify the find_subcluster method to include more nodes

def find_subcluster(self, graph, node_importance, min_importance=0.05, max_size=20):
    """
    Identify a subcluster of important nodes based on flow diffusion results.
    
    Args:
        graph: The graph representation
        node_importance: Dictionary mapping nodes to importance scores
        min_importance: Minimum importance threshold for inclusion
        max_size: Maximum size of the subcluster
        
    Returns:
        Set of node IDs in the identified subcluster
    """
    # Filter nodes with sufficient importance (lowered threshold)
    active_nodes = {node: score for node, score in node_importance.items() if score > min_importance}
    
    if not active_nodes:
        return set()
    
    # Sort by importance and take top nodes to form core of subcluster
    sorted_nodes = sorted(active_nodes.items(), key=lambda x: x[1], reverse=True)
    core_nodes = {node for node, _ in sorted_nodes[:min(8, len(sorted_nodes))]}  # Increased core size
    
    # Expand to include two-hop neighbors with positive importance
    subcluster = set(core_nodes)
    frontier = set(core_nodes)
    
    # First hop
    next_frontier = set()
    for node in frontier:
        for neighbor in graph.neighbors(node):
            if node_importance.get(neighbor, 0) > 0:
                subcluster.add(neighbor)
                next_frontier.add(neighbor)
    
    # Second hop - to get longer paths
    if len(subcluster) < max_size:
        for node in next_frontier:
            for neighbor in graph.neighbors(node):
                if (neighbor not in subcluster and 
                    node_importance.get(neighbor, 0) > 0 and
                    len(subcluster) < max_size * 1.5):  # Allow temporarily exceeding for better connectivity
                    subcluster.add(neighbor)
    
    # If subcluster is too large, prioritize:
    # 1. Nodes with higher importance
    # 2. Table nodes (they're connection points)
    # 3. Nodes with higher degree within the subcluster
    if len(subcluster) > max_size:
        # Calculate each node's internal degree
        internal_degree = {}
        for node in subcluster:
            internal_degree[node] = sum(1 for n in graph.neighbors(node) if n in subcluster)
        
        # Score each node
        node_scores = {}
        for node in subcluster:
            # Base score is the importance
            score = node_importance.get(node, 0) * 10
            
            # Bonus for table nodes
            if graph.nodes[node].get('type') == 'table':
                score += 5
            
            # Bonus for internal connectivity
            score += internal_degree.get(node, 0) * 0.5
            
            node_scores[node] = score
        
        # Keep highest scoring nodes
        ranked_nodes = sorted([(n, node_scores[n]) for n in subcluster], key=lambda x: x[1], reverse=True)
        subcluster = {n for n, _ in ranked_nodes[:max_size]}
    
    return subcluster

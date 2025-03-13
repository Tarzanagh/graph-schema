def enhance_edge_semantics(self, graph, metadata, use_diffusion=True, n_kernels=4):
    """
    Enhance graph edge weights using multiple kernels for different feature types.
    
    Args:
        graph: NetworkX graph of the database schema
        metadata: Dictionary containing schema metadata
        use_diffusion: Whether to apply network diffusion similar to SIMLR (default: True)
        n_kernels: Number of kernels to use (default: 4)
        
    Returns:
        Updated graph with semantically enhanced edge weights
    """
    enhanced_graph = graph.copy()
    
    # 1. Extract node feature vectors for each node
    node_features = self._extract_node_features(enhanced_graph, metadata)
    
    # 2. Extract edge feature vectors for each edge
    edge_features = self._extract_edge_features(enhanced_graph, metadata)
    
    # 3. Compute kernel matrices for node features
    node_kernel_matrices = []
    node_kernel_params = [0.5, 0.8, 1.0, 1.5]  # Different gamma values for kernels
    
    nodes = list(enhanced_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    for k in range(len(node_features[nodes[0]])):  # For each feature type
        # Extract feature matrix for kth feature
        feature_matrix = np.array([node_features[node][k] for node in nodes])
        
        # Compute distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Apply Gaussian kernel
        kernel = np.exp(-node_kernel_params[k % len(node_kernel_params)] * (distances ** 2))
        node_kernel_matrices.append(kernel)
    
    # 4. Compute combined edge weights using both node and edge features
    edge_kernel_params = [0.3, 0.6, 0.9, 1.2]  # Different gamma values for edge kernels
    
    # Initialize alpha weights for kernels
    alpha_node = np.ones(len(node_kernel_matrices)) / len(node_kernel_matrices)
    alpha_edge = np.ones(len(edge_features[list(enhanced_graph.edges())[0]])) / len(edge_features[list(enhanced_graph.edges())[0]])
    
    # Use LLM to refine alpha weights for specific column-to-column edges
    alpha_node, alpha_edge = self._refine_kernel_weights(enhanced_graph, metadata, node_features, edge_features)
    
    # Apply weights to edges
    for u, v in enhanced_graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        
        # Node similarity component (from kernel matrices)
        node_sim = sum(alpha_node[k] * node_kernel_matrices[k][i, j] for k in range(len(node_kernel_matrices)))
        
        # Edge feature component
        edge_sim = sum(alpha_edge[k] * edge_features[(u, v)][k] for k in range(len(edge_features[(u, v)])))
        
        # Combined weight (balance between node similarity and edge features)
        combined_weight = 0.5 * node_sim + 0.5 * edge_sim
        
        # Store features and weights
        enhanced_graph[u][v]['node_similarity'] = node_sim
        enhanced_graph[u][v]['edge_similarity'] = edge_sim
        enhanced_graph[u][v]['semantic_weight'] = combined_weight
        enhanced_graph[u][v]['weight'] = combined_weight
    
    # 5. Apply network diffusion if requested
    if use_diffusion:
        enhanced_graph = self._apply_network_diffusion(enhanced_graph, nodes, node_to_idx)
    
    return enhanced_graph

def _apply_network_diffusion(self, graph, nodes=None, node_to_idx=None):
    """
    Apply network diffusion similar to SIMLR approach to enhance the graph.
    
    Args:
        graph: NetworkX graph to enhance
        nodes: List of nodes (if None, will be computed)
        node_to_idx: Dictionary mapping nodes to indices (if None, will be computed)
        
    Returns:
        Enhanced graph with diffused weights
    """
    diffused_graph = graph.copy()
    
    # If nodes and node_to_idx are not provided, compute them
    if nodes is None:
        nodes = list(diffused_graph.nodes())
    
    if node_to_idx is None:
        node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    n = len(nodes)
    
    # Create adjacency matrix with edge weights
    A = np.zeros((n, n))
    for u, v, data in diffused_graph.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        A[i, j] = data.get('weight', 1.0)
        A[j, i] = data.get('weight', 1.0)  # Ensure symmetry
    
    # Apply network diffusion (adapted from SIMLR)
    k = 10  # Neighborhood size parameter
    A = A - np.diag(np.diag(A))  # Remove self-loops
    
    # Find k nearest neighbors for each node
    sorted_indices = np.argsort(-A, axis=1)
    P = np.zeros_like(A)
    
    for i in range(n):
        neighbors = sorted_indices[i, :min(k, n-1)]
        P[i, neighbors] = A[i, neighbors]
    
    # Make P symmetric
    P = (P + P.T) / 2
    
    # Add self-loops
    P = P + np.eye(n)
    
    # Normalize
    D_inv = np.diag(1.0 / np.maximum(np.sum(P, axis=1), 1e-10))
    P = D_inv @ P
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    
    # Apply diffusion
    alpha = 0.8
    beta = 2
    d = (1 - alpha) * eigenvalues / (1 - alpha * (eigenvalues ** beta))
    W = eigenvectors @ np.diag(d) @ eigenvectors.T
    
    # Update graph with diffused weights
    for i in range(n):
        for j in range(n):
            if i != j and W[i, j] > 0:
                u, v = nodes[i], nodes[j]
                if diffused_graph.has_edge(u, v):
                    diffused_graph[u][v]['diffused_weight'] = W[i, j]
                    diffused_graph[u][v]['weight'] = W[i, j]
    
    return diffused_graph




def enhance_edge_semantics(self, graph, metadata, use_diffusion=True, n_kernels=4):
    """
    Enhance graph edge weights using multiple kernels for different feature types.
    
    Args:
        graph: NetworkX graph of the database schema
        metadata: Dictionary containing schema metadata
        use_diffusion: Whether to apply network diffusion similar to SIMLR (default: True)
        n_kernels: Number of kernels to use (default: 4)
        
    Returns:
        Updated graph with semantically enhanced edge weights
    """
    enhanced_graph = graph.copy()
    
    # 1. Extract node feature vectors for each node
    node_features = self._extract_node_features(enhanced_graph, metadata)
    
    # 2. Extract edge feature vectors for each edge
    edge_features = self._extract_edge_features(enhanced_graph, metadata)
    
    # 3. Compute kernel matrices for node features
    node_kernel_matrices = []
    node_kernel_params = [0.5, 0.8, 1.0, 1.5]  # Different gamma values for kernels
    
    nodes = list(enhanced_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    for k in range(len(node_features[nodes[0]])):  # For each feature type
        # Extract feature matrix for kth feature
        feature_matrix = np.array([node_features[node][k] for node in nodes])
        
        # Compute distance matrix
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Apply Gaussian kernel
        kernel = np.exp(-node_kernel_params[k % len(node_kernel_params)] * (distances ** 2))
        node_kernel_matrices.append(kernel)
    
    # 4. Compute combined edge weights using both node and edge features
    edge_kernel_params = [0.3, 0.6, 0.9, 1.2]  # Different gamma values for edge kernels
    
    # Initialize alpha weights for kernels
    alpha_node = np.ones(len(node_kernel_matrices)) / len(node_kernel_matrices)
    alpha_edge = np.ones(len(edge_features[list(enhanced_graph.edges())[0]])) / len(edge_features[list(enhanced_graph.edges())[0]])
    
    # Use LLM to refine alpha weights for specific column-to-column edges
    alpha_node, alpha_edge = self._refine_kernel_weights(enhanced_graph, metadata, node_features, edge_features)
    
    # Apply weights to edges
    for u, v in enhanced_graph.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        
        # Node similarity component (from kernel matrices)
        node_sim = sum(alpha_node[k] * node_kernel_matrices[k][i, j] for k in range(len(node_kernel_matrices)))
        
        # Edge feature component
        edge_sim = sum(alpha_edge[k] * edge_features[(u, v)][k] for k in range(len(edge_features[(u, v)])))
        
        # Combined weight (balance between node similarity and edge features)
        combined_weight = 0.5 * node_sim + 0.5 * edge_sim
        
        # Store features and weights
        enhanced_graph[u][v]['node_similarity'] = node_sim
        enhanced_graph[u][v]['edge_similarity'] = edge_sim
        enhanced_graph[u][v]['semantic_weight'] = combined_weight
        enhanced_graph[u][v]['weight'] = combined_weight
    
    # 5. Apply network diffusion if requested (similar to SIMLR approach)
    if use_diffusion:
        # Create adjacency matrix with edge weights
        A = np.zeros((n, n))
        for u, v, data in enhanced_graph.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            A[i, j] = data.get('weight', 1.0)
            A[j, i] = data.get('weight', 1.0)  # Ensure symmetry
        
        # Apply network diffusion (adapted from SIMLR)
        k = 10  # Neighborhood size parameter
        A = A - np.diag(np.diag(A))  # Remove self-loops
        
        # Find k nearest neighbors for each node
        sorted_indices = np.argsort(-A, axis=1)
        P = np.zeros_like(A)
        
        for i in range(n):
            neighbors = sorted_indices[i, :min(k, n-1)]
            P[i, neighbors] = A[i, neighbors]
        
        # Make P symmetric
        P = (P + P.T) / 2
        
        # Add self-loops
        P = P + np.eye(n)
        
        # Normalize
        D_inv = np.diag(1.0 / np.maximum(np.sum(P, axis=1), 1e-10))
        P = D_inv @ P
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        
        # Apply diffusion
        alpha = 0.8
        beta = 2
        d = (1 - alpha) * eigenvalues / (1 - alpha * (eigenvalues ** beta))
        W = eigenvectors @ np.diag(d) @ eigenvectors.T
        
        # Update graph with diffused weights
        for i in range(n):
            for j in range(n):
                if i != j and W[i, j] > 0:
                    u, v = nodes[i], nodes[j]
                    if enhanced_graph.has_edge(u, v):
                        enhanced_graph[u][v]['diffused_weight'] = W[i, j]
                        enhanced_graph[u][v]['weight'] = W[i, j]
    
    return enhanced_graph

def _extract_node_features(self, graph, metadata):
    """
    Extract multiple feature vectors for each node in the graph.
    
    Args:
        graph: NetworkX graph
        metadata: Schema metadata
        
    Returns:
        Dictionary mapping node names to lists of feature vectors
    """
    node_features = {}
    
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('type', '')
        node_name = node if '.' not in node else node.split('.')[1]
        table_name = node.split('.')[0] if '.' in node else ''
        
        # 1. Textual Semantic Features
        textual_feature = np.zeros(10)
        # Simple embedding: character counts, word length, etc.
        textual_feature[0] = len(node_name) / 20  # Normalized length
        textual_feature[1] = len(node_name.split('_')) / 5  # Number of parts
        textual_feature[2] = float('id' in node_name.lower())
        textual_feature[3] = float('name' in node_name.lower())
        textual_feature[4] = float('date' in node_name.lower())
        textual_feature[5] = float('key' in node_name.lower())
        textual_feature[6] = float('code' in node_name.lower())
        textual_feature[7] = float('status' in node_name.lower())
        textual_feature[8] = float('amount' in node_name.lower())
        textual_feature[9] = float('price' in node_name.lower())
        
        # 2. Structural Position Features
        structural_feature = np.zeros(10)
        neighbors = list(graph.neighbors(node))
        structural_feature[0] = len(neighbors) / 20  # Normalized degree
        structural_feature[1] = graph.degree(node) / 50  # Normalized degree
        structural_feature[2] = 1.0 if node_type == 'table' else 0.0
        structural_feature[3] = 1.0 if node_type == 'column' else 0.0
        structural_feature[4] = sum(1 for n in neighbors if graph.nodes[n].get('type') == 'column') / 20
        structural_feature[5] = sum(1 for n in neighbors if graph.nodes[n].get('type') == 'table') / 5
        
        # Calculate centrality
        if 'centrality' in node_data:
            structural_feature[6] = node_data['centrality']
        
        # 3. Type and Metadata Features
        type_feature = np.zeros(10)
        data_type = node_data.get('data_type', '').lower()
        type_feature[0] = float('int' in data_type or 'number' in data_type)
        type_feature[1] = float('varchar' in data_type or 'char' in data_type or 'text' in data_type)
        type_feature[2] = float('date' in data_type or 'time' in data_type)
        type_feature[3] = float('bool' in data_type)
        type_feature[4] = float('float' in data_type or 'double' in data_type or 'decimal' in data_type)
        type_feature[5] = float(node_data.get('is_primary_key', False))
        type_feature[6] = float(node_data.get('is_foreign_key', False))
        type_feature[7] = float(node_data.get('is_unique', False))
        type_feature[8] = float(node_data.get('is_indexed', False))
        type_feature[9] = float(node_data.get('not_null', False))
        
        # 4. Statistical Features
        statistical_feature = np.zeros(10)
        statistical_feature[0] = node_data.get('nullability', 0.0)
        statistical_feature[1] = node_data.get('avg_length', 0.0) / 100  # Normalized
        statistical_feature[2] = node_data.get('cardinality', 0.0) / 1000  # Normalized
        statistical_feature[3] = node_data.get('avg_value', 0.0) / 1000  # Normalized
        statistical_feature[4] = node_data.get('std_dev', 0.0) / 100  # Normalized
        statistical_feature[5] = node_data.get('min_value', 0.0) / 1000  # Normalized
        statistical_feature[6] = node_data.get('max_value', 0.0) / 1000  # Normalized
        
        # 5. Domain Knowledge Features
        domain_feature = np.zeros(10)
        domain_feature[0] = float('customer' in node_name.lower() or 'customer' in table_name.lower())
        domain_feature[1] = float('order' in node_name.lower() or 'order' in table_name.lower())
        domain_feature[2] = float('product' in node_name.lower() or 'product' in table_name.lower())
        domain_feature[3] = float('payment' in node_name.lower() or 'payment' in table_name.lower())
        domain_feature[4] = float('user' in node_name.lower() or 'user' in table_name.lower())
        domain_feature[5] = float('transaction' in node_name.lower() or 'transaction' in table_name.lower())
        domain_feature[6] = float('inventory' in node_name.lower() or 'inventory' in table_name.lower())
        domain_feature[7] = float('employee' in node_name.lower() or 'employee' in table_name.lower())
        domain_feature[8] = float('account' in node_name.lower() or 'account' in table_name.lower())
        domain_feature[9] = float('location' in node_name.lower() or 'location' in table_name.lower())
        
        # Store all feature vectors for this node
        node_features[node] = [
            textual_feature, 
            structural_feature,
            type_feature,
            statistical_feature,
            domain_feature
        ]
    
    return node_features

def _extract_edge_features(self, graph, metadata):
    """
    Extract multiple feature vectors for each edge in the graph.
    
    Args:
        graph: NetworkX graph
        metadata: Schema metadata
        
    Returns:
        Dictionary mapping edge tuples to lists of feature vectors
    """
    edge_features = {}
    
    for u, v, data in graph.edges(data=True):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        u_type = u_data.get('type', '')
        v_type = v_data.get('type', '')
        u_name = u if '.' not in u else u.split('.')[1]
        v_name = v if '.' not in v else v.split('.')[1]
        rel_type = data.get('relationship_type', '')
        
        # 1. Schema Structural Edge Features
        structural_feature = np.zeros(10)
        structural_feature[0] = float(rel_type == 'foreign_key')
        structural_feature[1] = float(rel_type == 'table_column')
        structural_feature[2] = float(rel_type == 'same_table')
        structural_feature[3] = float(rel_type == 'join_path')
        structural_feature[4] = float(rel_type == 'view_dependency')
        
        # 2. Semantic Similarity Edge Features
        semantic_feature = np.zeros(10)
        # Name similarity (simple Jaccard-like metric)
        u_tokens = set(u_name.lower().split('_'))
        v_tokens = set(v_name.lower().split('_'))
        if u_tokens and v_tokens:  # Avoid division by zero
            semantic_feature[0] = len(u_tokens.intersection(v_tokens)) / len(u_tokens.union(v_tokens))
        
        # Check for common substrings
        common_substring_len = 0
        for i in range(min(len(u_name), len(v_name))):
            if u_name[i].lower() == v_name[i].lower():
                common_substring_len += 1
        semantic_feature[1] = common_substring_len / max(len(u_name), len(v_name)) if max(len(u_name), len(v_name)) > 0 else 0
        
        # Calculate edit distance (normalized)
        edit_distance = self._levenshtein_distance(u_name.lower(), v_name.lower())
        semantic_feature[2] = 1.0 - (edit_distance / max(len(u_name), len(v_name))) if max(len(u_name), len(v_name)) > 0 else 0
        
        # Check for common prefixes/suffixes
        prefixes = {'id', 'name', 'date', 'num', 'code', 'key', 'status', 'type'}
        u_prefixes = {prefix for prefix in prefixes if u_name.lower().startswith(prefix)}
        v_prefixes = {prefix for prefix in prefixes if v_name.lower().startswith(prefix)}
        semantic_feature[3] = float(bool(u_prefixes.intersection(v_prefixes)))
        
        suffixes = {'id', 'name', 'date', 'count', 'total', 'code', 'key', 'status', 'type'}
        u_suffixes = {suffix for suffix in suffixes if u_name.lower().endswith(suffix)}
        v_suffixes = {suffix for suffix in suffixes if v_name.lower().endswith(suffix)}
        semantic_feature[4] = float(bool(u_suffixes.intersection(v_suffixes)))
        
        # 3. Statistical Co-occurrence Features
        statistical_feature = np.zeros(10)
        # Simple co-occurrence metrics (assuming these are stored in edge data)
        statistical_feature[0] = data.get('co_occurrence', 0.0)
        statistical_feature[1] = data.get('pmi', 0.0)  # Pointwise mutual information
        statistical_feature[2] = data.get('conditional_prob', 0.0)
        
        # 4. Domain-Specific Compatibility Features
        compatibility_feature = np.zeros(10)
        # Type compatibility
        u_type_str = u_data.get('data_type', '').lower()
        v_type_str = v_data.get('data_type', '').lower()
        
        # 1 for exact match, 0.5 for compatible types, 0 for incompatible
        if u_type_str == v_type_str:
            compatibility_feature[0] = 1.0
        elif ('int' in u_type_str and 'int' in v_type_str) or \
             ('char' in u_type_str and 'char' in v_type_str) or \
             ('date' in u_type_str and 'date' in v_type_str) or \
             (('float' in u_type_str or 'decimal' in u_type_str) and ('float' in v_type_str or 'decimal' in v_type_str)):
            compatibility_feature[0] = 0.5
        
        # Primary key to foreign key relationship
        compatibility_feature[1] = float(u_data.get('is_primary_key', False) and v_data.get('is_foreign_key', False))
        
        # Foreign key to primary key relationship
        compatibility_feature[2] = float(u_data.get('is_foreign_key', False) and v_data.get('is_primary_key', False))
        
        # 5. Contextual Edge Features
        contextual_feature = np.zeros(10)
        # Could be based on node descriptions, schema documentation, etc.
        contextual_feature[0] = data.get('description_similarity', 0.0)
        
        # Table-column relationship
        if u_type == 'table' and v_type == 'column':
            contextual_feature[1] = 1.0
        elif u_type == 'column' and v_type == 'table':
            contextual_feature[1] = 1.0
        
        # Column-column relationship in same table
        if u_type == 'column' and v_type == 'column':
            u_table = u.split('.')[0] if '.' in u else ''
            v_table = v.split('.')[0] if '.' in v else ''
            contextual_feature[2] = float(u_table == v_table and u_table != '')
        
        # 6. Domain Knowledge Edge Features
        domain_feature = np.zeros(10)
        # Common domain concepts
        domain_concepts = {
            'order': {'customer', 'product', 'date', 'status', 'total'},
            'customer': {'name', 'address', 'email', 'phone'},
            'product': {'name', 'price', 'category', 'inventory'},
            'employee': {'name', 'department', 'salary', 'hire_date'},
            'transaction': {'date', 'amount', 'type', 'account'}
        }
        
        # Check if nodes belong to same domain concept
        for concept, terms in domain_concepts.items():
            u_match = any(term in u_name.lower() for term in terms)
            v_match = any(term in v_name.lower() for term in terms)
            if u_match and v_match:
                domain_feature[0] = 1.0
                break
        
        # Store all feature vectors for this edge
        edge_features[(u, v)] = [
            structural_feature,
            semantic_feature,
            statistical_feature,
            compatibility_feature,
            contextual_feature,
            domain_feature
        ]
    
    return edge_features

def _refine_kernel_weights(self, graph, metadata, node_features, edge_features):
    """
    Use LLM to refine kernel weights for specific edges.
    
    Args:
        graph: NetworkX graph
        metadata: Schema metadata
        node_features: Dictionary of node features
        edge_features: Dictionary of edge features
        
    Returns:
        Tuple of (node_kernel_weights, edge_kernel_weights)
    """
    # Format schema for prompt
    schema_text = SchemaGraphBuilder.format_schema_for_prompt(metadata)
    
    # Identify column-to-column edges for LLM analysis
    column_pairs = []
    for u, v, data in graph.edges(data=True):
        if graph.nodes[u].get('type') == 'column' and graph.nodes[v].get('type') == 'column':
            rel_type = data.get('relationship_type', '')
            column_pairs.append((u, v, rel_type))
    
    # Sample a subset for analysis (to avoid too long prompts)
    if len(column_pairs) > 20:
        import random
        sampled_pairs = random.sample(column_pairs, 20)
    else:
        sampled_pairs = column_pairs
    
    # Create prompt
    prompt = f"""
    Your task is to determine the importance of different feature types when assessing relationships between database columns.
    
    {schema_text}
    
    I want you to evaluate the importance of these feature types for measuring column relationships:
    1. Textual Semantic Features (column names, text similarity)
    2. Structural Position Features (positions in the schema)
    3. Type and Metadata Features (data types, constraints)
    4. Statistical Features (distributions, cardinality)
    5. Domain Knowledge Features (business context)
    
    And for edge relationships:
    1. Schema Structural Features (foreign keys, table-column relationships)
    2. Semantic Similarity Features (name similarities, common patterns)
    3. Statistical Co-occurrence Features (join frequencies)
    4. Compatibility Features (type compatibility)
    5. Contextual Features (documentation, descriptions)
    6. Domain Knowledge Features (business rules, common patterns)
    
    For the given database schema, assign weights to each feature type (weights should sum to 1.0 for each category).
    
    Respond in this exact format:
    <NODE_WEIGHTS>
    textual: 0.X
    structural: 0.X
    type: 0.X
    statistical: 0.X
    domain: 0.X
    </NODE_WEIGHTS>
    
    <EDGE_WEIGHTS>
    structural: 0.X
    semantic: 0.X
    statistical: 0.X
    compatibility: 0.X
    contextual: 0.X
    domain: 0.X
    </EDGE_WEIGHTS>
    """
    
    # Call LLM
    llm_response = self.call_llm(prompt, max_tokens=1000, temperature=0.2)
    
    # Parse node weights
    node_weights = np.ones(len(node_features[list(graph.nodes())[0]])) / len(node_features[list(graph.nodes())[0]])
    edge_weights = np.ones(len(edge_features[list(graph.edges())[0]])) / len(edge_features[list(graph.edges())[0]])
    
    # Extract node weights
    node_match = re.search(r'<NODE_WEIGHTS>(.*?)</NODE_WEIGHTS>', llm_response, re.DOTALL)
    if node_match:
        node_text = node_match.group(1).strip()
        try:
            weights = []
            for line in node_text.split('\n'):
                if ':' in line:
                    weight = float(re.search(r'0\.\d+', line).group(0))
                    weights.append(weight)
            if len(weights) == len(node_weights) and abs(sum(weights) - 1.0) < 0.1:
                node_weights = np.array(weights)
                node_weights = node_weights / sum(node_weights)  # Normalize
        except:
            pass  # Keep default weights if parsing fails
    
    # Extract edge weights
    edge_match = re.search(r'<EDGE_WEIGHTS>(.*?)</EDGE_WEIGHTS>', llm_response, re.DOTALL)
    if edge_match:
        edge_text = edge_match.group(1).strip()
        try:
            weights = []
            for line in edge_text.split('\n'):
                if ':' in line:
                    weight = float(re.search(r'0\.\d+', line).group(0))
                    weights.append(weight)
            if len(weights) == len(edge_weights) and abs(sum(weights) - 1.0) < 0.1:
                edge_weights = np.array(weights)
                edge_weights = edge_weights / sum(edge_weights)  # Normalize
        except:
            pass  # Keep default weights if parsing fails
    
    return node_weights, edge_weights

def _levenshtein_distance(self, s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    if len(s1) < len(s2):
        return self._levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

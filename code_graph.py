import re
import json
import argparse
import numpy as np
from collections import defaultdict

# Note: These imports are for semantic similarity functionality
# If unavailable, the system will fall back to keyword-based matching
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers package not found. Semantic similarity will not be available.")
    print("To enable semantic similarity, install with: pip install sentence-transformers")

try:
    import spacy

    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Warning: spacy package not found. Fallback semantic similarity will not be available.")
    print("To enable fallback semantic similarity, install with: pip install spacy")
    print("Then download a model with: python -m spacy download en_core_web_md")


class SchemaGraphPathFilter:
    """
    A flexible system for filtering relevant paths from a schema graph based on queries.
    Supports multiple filtering strategies that can be combined as needed.
    """

    def __init__(self, paths_file, enable_semantic=True, semantic_model=None, similarity_threshold=0.5):
        """
        Initialize with paths from a JSON file

        Args:
            paths_file: Path to JSON file containing graph paths
            enable_semantic: Whether to enable semantic similarity functionality
            semantic_model: Name of the sentence-transformers model to use
                           (default: 'all-MiniLM-L6-v2' if None)
            similarity_threshold: Minimum similarity score (0-1) for semantic matching
        """
        with open(paths_file, 'r') as f:
            self.paths = json.load(f)

        print(f"Loaded {len(self.paths)} paths from {paths_file}")

        # Initialize indexes for faster lookups
        self._build_indexes()

        # Set up semantic similarity if enabled
        self.enable_semantic = enable_semantic
        self.similarity_threshold = similarity_threshold
        self.semantic_model = None
        self.nlp = None

        if enable_semantic:
            self._initialize_semantic_models(semantic_model)

    def _build_indexes(self):
        """Build various indexes to speed up filtering operations"""
        # Index paths by node
        self.node_index = defaultdict(list)
        # Index paths by relationship type
        self.relationship_index = defaultdict(list)
        # Index paths by source-target pair
        self.endpoint_index = defaultdict(list)
        # Index paths by length
        self.length_index = defaultdict(list)
        # Index paths by node types (table/column)
        self.node_type_index = defaultdict(list)
        # Store text representation of each path for semantic search
        self.path_texts = []

        for i, path in enumerate(self.paths):
            # Index by all nodes in the path
            for node in path['path']:
                self.node_index[node].append(i)

            # Index by relationship types
            for edge in path['edges']:
                rel_type = edge.get('relationship_type')
                if rel_type:
                    self.relationship_index[rel_type].append(i)

            # Index by source-target pair
            source = path['source']
            target = path['target']
            self.endpoint_index[(source, target)].append(i)

            # Index by path length
            self.length_index[path['length']].append(i)

            # Index by node types
            source_type = path['source_type']
            target_type = path['target_type']
            self.node_type_index[(source_type, target_type)].append(i)

            # Create text representation of path for semantic search
            nodes_text = " ".join(path['path'])
            relationships_text = " ".join([edge.get('relationship', '') for edge in path['edges']])
            path_text = f"{nodes_text} {relationships_text}"
            self.path_texts.append(path_text)

    def _initialize_semantic_models(self, model_name=None):
        """Initialize semantic similarity models if available"""
        if not self.enable_semantic:
            return

        # Try to initialize sentence-transformers (preferred method)
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                model_name = model_name or 'all-MiniLM-L6-v2'  # Fast and effective model
                print(f"Loading sentence-transformers model: {model_name}")
                self.semantic_model = SentenceTransformer(model_name)

                # Instead of pre-computing embeddings for entire paths,
                # we'll pre-compute embeddings for individual entities (tables, columns)
                print("Computing embeddings for entities...")

                # Extract unique entity names
                self.all_entities = set()
                for path in self.paths:
                    # Add all nodes as entities
                    for node in path['path']:
                        self.all_entities.add(node)

                # Convert to list for encoding
                entity_list = list(self.all_entities)
                if entity_list:
                    # Compute embeddings for all entities
                    self.entity_embeddings = {}
                    batched_entities = [entity_list[i:i + 64] for i in range(0, len(entity_list), 64)]

                    for batch in batched_entities:
                        batch_embeddings = self.semantic_model.encode(batch, show_progress_bar=False)
                        for i, entity in enumerate(batch):
                            self.entity_embeddings[entity] = batch_embeddings[i]

                    print(f"Generated embeddings for {len(self.entity_embeddings)} unique entities")
                else:
                    print("Warning: No entities available for embedding.")
                    self.entity_embeddings = {}

                return
            except Exception as e:
                print(f"Error initializing sentence-transformers: {e}")
                self.semantic_model = None

        # Fallback to spaCy if available
        if HAS_SPACY:
            try:
                print("Falling back to spaCy for semantic similarity")
                self.nlp = spacy.load("en_core_web_md")  # Medium-sized model with word vectors
                return
            except Exception as e:
                print(f"Error initializing spaCy: {e}")
                self.nlp = None

        print("Semantic similarity is not available. Using keyword matching only.")

    def filter_by_keywords(self, keywords, match_all=False):
        """
        Filter paths containing specified keywords

        Args:
            keywords: List of keywords to search for in node names
            match_all: If True, all keywords must be present; if False, any keyword is sufficient

        Returns:
            List of filtered paths
        """
        if not keywords:
            return self.paths

        # Preprocess keywords for more flexible matching
        processed_keywords = [k.lower() for k in keywords]

        filtered_paths = []
        for path in self.paths:
            # Create a string representation of the path for text search
            path_text = " ".join(path['path']).lower()

            if match_all:
                # All keywords must be present
                if all(kw in path_text for kw in processed_keywords):
                    filtered_paths.append(path)
            else:
                # Any keyword is sufficient
                if any(kw in path_text for kw in processed_keywords):
                    filtered_paths.append(path)

        return filtered_paths

    def filter_by_node_types(self, source_type=None, target_type=None):
        """
        Filter paths by source and/or target node types

        Args:
            source_type: Type of source node (e.g., 'table', 'column')
            target_type: Type of target node (e.g., 'table', 'column')

        Returns:
            List of filtered paths
        """
        filtered_paths = []

        for path in self.paths:
            matches = True

            if source_type and path['source_type'] != source_type:
                matches = False

            if target_type and path['target_type'] != target_type:
                matches = False

            if matches:
                filtered_paths.append(path)

        return filtered_paths

    def filter_by_endpoints(self, source=None, target=None):
        """
        Filter paths by exact source and/or target nodes

        Args:
            source: Name of source node
            target: Name of target node

        Returns:
            List of filtered paths
        """
        filtered_paths = []

        for path in self.paths:
            matches = True

            if source and path['source'] != source:
                matches = False

            if target and path['target'] != target:
                matches = False

            if matches:
                filtered_paths.append(path)

        return filtered_paths

    def filter_by_relationships(self, relationship_types, require_all=False):
        """
        Filter paths containing specific relationship types

        Args:
            relationship_types: List of relationship types to look for
            require_all: If True, all specified relationship types must be present

        Returns:
            List of filtered paths
        """
        if not relationship_types:
            return self.paths

        filtered_paths = []

        for path in self.paths:
            # Extract relationship types from the path
            path_rel_types = [edge.get('relationship_type') for edge in path['edges']]

            if require_all:
                # All specified relationship types must be present
                if all(rel_type in path_rel_types for rel_type in relationship_types):
                    filtered_paths.append(path)
            else:
                # Any specified relationship type is sufficient
                if any(rel_type in path_rel_types for rel_type in relationship_types):
                    filtered_paths.append(path)

        return filtered_paths

    def filter_by_length(self, min_length=None, max_length=None):
        """
        Filter paths by length (number of edges)

        Args:
            min_length: Minimum path length (inclusive)
            max_length: Maximum path length (inclusive)

        Returns:
            List of filtered paths
        """
        filtered_paths = []

        for path in self.paths:
            length = path['length']

            if min_length is not None and length < min_length:
                continue

            if max_length is not None and length > max_length:
                continue

            filtered_paths.append(path)

        return filtered_paths

    def filter_by_node_presence(self, required_nodes, optional_nodes=None):
        """
        Filter paths containing specific nodes

        Args:
            required_nodes: List of nodes that must be present in the path
            optional_nodes: List of nodes where at least one should be present

        Returns:
            List of filtered paths
        """
        filtered_paths = []

        for path in self.paths:
            path_nodes = set(path['path'])

            # Check required nodes
            if required_nodes and not all(node in path_nodes for node in required_nodes):
                continue

            # Check optional nodes
            if optional_nodes and not any(node in path_nodes for node in optional_nodes):
                continue

            filtered_paths.append(path)

        return filtered_paths

    def rank_paths_by_relevance(self, query_terms, max_results=None):
        """
        Rank paths by relevance to query terms

        Args:
            query_terms: List of terms to match against paths
            max_results: Maximum number of results to return

        Returns:
            List of paths sorted by relevance score
        """
        if not query_terms:
            return self.paths[:max_results] if max_results else self.paths

        scored_paths = []

        for path in self.paths:
            score = 0
            path_text = " ".join(path['path']).lower()

            # Score based on term presence
            for term in query_terms:
                term = term.lower()
                if term in path_text:
                    # Exact matches get higher score
                    if any(term == node.lower() for node in path['path']):
                        score += 2
                    # Partial matches
                    else:
                        score += 1

            # Adjust score by path length (shorter paths preferred)
            adjusted_score = score / (0.5 * path['length'] + 1)

            scored_paths.append((adjusted_score, path))

        # Sort by score in descending order
        scored_paths.sort(reverse=True, key=lambda x: x[0])

        # Return ranked paths (limited if max_results specified)
        if max_results:
            return [p for _, p in scored_paths[:max_results]]
        else:
            return [p for _, p in scored_paths]

    def extract_query_components(self, query):
        """
        Parse a natural language query to extract key components

        Args:
            query: Natural language query string

        Returns:
            Dictionary with extracted query components
        """
        query = query.lower()
        components = {
            'keywords': [],
            'tables': [],
            'columns': [],
            'relationships': []
        }

        # Extract tables (assuming they're mentioned as "table X" or "X table")
        table_patterns = [r'table (\w+)', r'(\w+) table']
        for pattern in table_patterns:
            for match in re.finditer(pattern, query):
                components['tables'].append(match.group(1))

        # Extract columns (assuming they're mentioned as "column X" or "X column")
        column_patterns = [r'column (\w+)', r'(\w+) column']
        for pattern in column_patterns:
            for match in re.finditer(pattern, query):
                components['columns'].append(match.group(1))

        # Extract relationship types based on common terms
        relationship_map = {
            'primary key': 'pk_table',
            'foreign key': 'fk_table',
            'reference': 'pk_fk_column',
            'same table': 'same_table',
            'table column': 'table_column',
            'table to table': 'pk_fk_table'
        }

        for term, rel_type in relationship_map.items():
            if term in query:
                components['relationships'].append(rel_type)

        # Extract general keywords (words longer than 3 characters)
        for word in re.findall(r'\b\w{4,}\b', query):
            if (word not in components['tables'] and
                    word not in components['columns'] and
                    word not in ['table', 'column', 'primary', 'foreign', 'key', 'reference']):
                components['keywords'].append(word)

        return components

    def filter_by_semantic_similarity(self, query_text, top_k=None):
        """
        Filter paths by semantic similarity matching between query entities and path entities

        Args:
            query_text: Query text to extract entities from and compare against path entities
            top_k: Number of top results to return (if None, uses similarity_threshold)

        Returns:
            List of filtered paths sorted by entity-level semantic similarity
        """
        if not self.enable_semantic or (self.semantic_model is None and self.nlp is None):
            print("Semantic similarity not available. Falling back to keyword matching.")
            # Fall back to keyword-based matching
            keywords = query_text.lower().split()
            return self.filter_by_keywords(keywords)

        # Extract query components to get potential entities
        query_components = self.extract_query_components(query_text)

        # Combine all potential entities from the query
        query_entities = (
                query_components.get('tables', []) +
                query_components.get('columns', []) +
                query_components.get('keywords', [])
        )

        # Make sure we have something to match
        if not query_entities:
            print("No distinct entities found in query. Extracting words as entities.")
            # Extract words as potential entities (filter out common words)
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                          'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'about',
                          'against', 'between', 'into', 'through', 'during', 'before', 'after',
                          'above', 'below', 'under', 'over', 'again', 'further', 'then', 'once',
                          'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                          'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                          'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                          'can', 'will', 'just', 'don', 'should', 'now', 'between', 'find', 'get'}

            query_words = re.findall(r'\b\w+\b', query_text.lower())
            query_entities = [word for word in query_words if word not in stop_words and len(word) > 3]

        scored_paths = []

        # Using sentence-transformers (preferred method)
        if self.semantic_model is not None and self.entity_embeddings:
            # Encode the query entities
            query_entity_embeddings = {}
            for entity in query_entities:
                query_entity_embeddings[entity] = self.semantic_model.encode(entity)

            # Score each path based on entity similarity
            for path_idx, path in enumerate(self.paths):
                path_score = 0
                path_entities = path['path']

                # Skip empty paths
                if not path_entities:
                    continue

                # For each query entity, find its best match in the path
                for query_entity, query_embedding in query_entity_embeddings.items():
                    best_similarity = 0

                    # Compare with each path entity
                    for path_entity in path_entities:
                        if path_entity in self.entity_embeddings:
                            path_embedding = self.entity_embeddings[path_entity]

                            # Calculate cosine similarity
                            similarity = np.dot(query_embedding, path_embedding) / (
                                    np.linalg.norm(query_embedding) * np.linalg.norm(path_embedding) + 1e-8
                            )

                            best_similarity = max(best_similarity, similarity)

                    # Add the best similarity to the path score
                    path_score += best_similarity

                # Normalize by number of query entities
                if query_entity_embeddings:
                    path_score /= len(query_entity_embeddings)

                # Adjust score by path length (prefer shorter paths)
                path_length = path['length']
                adjusted_score = path_score / (0.1 * path_length + 1.0)

                scored_paths.append((adjusted_score, path))

        # Using spaCy fallback for entity similarity
        elif self.nlp is not None:
            # Process query entities with spaCy
            query_entity_docs = {entity: self.nlp(entity) for entity in query_entities}

            # Score each path
            for path_idx, path in enumerate(self.paths):
                path_score = 0
                path_entities = path['path']

                # Skip empty paths
                if not path_entities:
                    continue

                # For each query entity, find its best match in the path
                for query_entity, query_doc in query_entity_docs.items():
                    best_similarity = 0

                    # Compare with each path entity
                    for path_entity in path_entities:
                        path_doc = self.nlp(path_entity)
                        similarity = query_doc.similarity(path_doc)
                        best_similarity = max(best_similarity, similarity)

                    # Add the best similarity to the path score
                    path_score += best_similarity

                # Normalize by number of query entities
                if query_entity_docs:
                    path_score /= len(query_entity_docs)

                # Adjust by path length
                path_length = path['length']
                adjusted_score = path_score / (0.1 * path_length + 1.0)

                scored_paths.append((adjusted_score, path))

        # Fall back to keyword matching if no semantic scores
        if not scored_paths:
            print("No semantic scores generated. Falling back to keyword matching.")
            return self.filter_by_keywords(query_entities)

        # Sort by similarity score (descending)
        scored_paths.sort(reverse=True, key=lambda x: x[0])

        if top_k is not None:
            # Return top-k results
            return [path for score, path in scored_paths[:top_k]]
        else:
            # Filter by threshold
            return [path for score, path in scored_paths if score >= self.similarity_threshold]

        # Sort by similarity score (descending)
        scored_paths.sort(reverse=True, key=lambda x: x[0])

        if top_k is not None:
            # Return top-k results
            return [path for score, path in scored_paths[:top_k]]
        else:
            # Filter by threshold
            return [path for score, path in scored_paths if score >= self.similarity_threshold]

    def filter_paths_for_query(self, query, filters=None, max_results=20, use_semantic=True):
        """
        Filter paths relevant to a query using multiple filtering strategies

        Args:
            query: Query string or query components dictionary
            filters: List of filter names to apply (e.g., ['keywords', 'relationships'])
                    If None, all filters are applied
            max_results: Maximum number of results to return
            use_semantic: Whether to use semantic similarity (requires initialized models)

        Returns:
            List of relevant paths
        """
        # Default filters to apply if none specified
        all_filters = ['semantic', 'keywords', 'node_types', 'relationships', 'endpoints', 'ranking']
        filters = filters or all_filters

        # Parse query if it's a string
        if isinstance(query, str):
            query_components = self.extract_query_components(query)
        else:
            query_components = query

        # Start with all paths
        relevant_paths = self.paths

        # Safety check - if we have no paths, return empty list
        if not relevant_paths:
            print("Warning: No paths available in the graph.")
            return []

        # Parse query if it's a string
        if isinstance(query, str):
            query_components = self.extract_query_components(query)
        else:
            query_components = query

        # Start with all paths
        relevant_paths = self.paths

        # Apply semantic similarity filter first if enabled
        if 'semantic' in filters and use_semantic and isinstance(query, str):
            # Use semantic similarity with original query text
            semantic_paths = self.filter_by_semantic_similarity(query, top_k=max(50, max_results * 2))
            if semantic_paths:
                relevant_paths = semantic_paths
                print(f"Applied semantic similarity filter: {len(relevant_paths)} paths")

        # Apply keyword filter
        if 'keywords' in filters and (query_components.get('keywords') or
                                      query_components.get('tables') or
                                      query_components.get('columns')):
            # Combine all text keywords
            all_keywords = (query_components.get('keywords', []) +
                            query_components.get('tables', []) +
                            query_components.get('columns', []))
            keyword_paths = self.filter_by_keywords(all_keywords)

            if keyword_paths:
                relevant_paths = keyword_paths
                print(f"Applied keyword filter: {len(relevant_paths)} paths")

        if 'node_types' in filters:
            # Filter for table-to-table paths if query mentions tables
            if query_components.get('tables') and len(query_components.get('tables', [])) >= 2:
                relevant_paths = [p for p in relevant_paths if
                                  p['source_type'] == 'table' and p['target_type'] == 'table']

            # Filter for column-to-column paths if query mentions columns
            elif query_components.get('columns') and len(query_components.get('columns', [])) >= 2:
                relevant_paths = [p for p in relevant_paths if
                                  p['source_type'] == 'column' and p['target_type'] == 'column']

        if 'relationships' in filters and query_components.get('relationships'):
            # Filter by relationship types extracted from query
            filtered_by_rel = self.filter_by_relationships(query_components['relationships'])
            # Only apply if we get results, otherwise keep previous results
            if filtered_by_rel:
                relevant_paths = filtered_by_rel

        if 'endpoints' in filters:
            # Try to identify specific endpoints from tables/columns mentioned
            if len(query_components.get('tables', [])) >= 2:
                # If two tables are mentioned, they might be endpoints
                table1, table2 = query_components['tables'][:2]
                filtered_by_endpoints = self.filter_by_endpoints(source=table1, target=table2)

                # If no results, try reversing the order
                if not filtered_by_endpoints:
                    filtered_by_endpoints = self.filter_by_endpoints(source=table2, target=table1)

                # Apply if we got results
                if filtered_by_endpoints:
                    relevant_paths = filtered_by_endpoints

        # Final ranking of results
        if 'ranking' in filters:
            # Combine all terms for ranking
            all_terms = []
            for category in ['keywords', 'tables', 'columns']:
                all_terms.extend(query_components.get(category, []))

            relevant_paths = self.rank_paths_by_relevance(all_terms, max_results)

        # Limit results if needed
        if max_results and len(relevant_paths) > max_results:
            relevant_paths = relevant_paths[:max_results]

        return relevant_paths

    def explain_path(self, path, query_entities=None):
        """
        Generate a human-readable explanation of a path, optionally highlighting
        semantic matches to query entities

        Args:
            path: Path object
            query_entities: Optional list of query entities to highlight matches for

        Returns:
            String explanation of the path
        """
        nodes = path['path']
        edges = path['edges']

        explanation = f"Path from {path['source']} to {path['target']} (length: {path['length']}):\n"

        # Add path sequence
        explanation += "  " + " → ".join(nodes) + "\n\n"

        # Add semantic match information if provided
        if query_entities and self.semantic_model is not None and hasattr(self, 'entity_embeddings'):
            explanation += "Semantic matches:\n"

            for query_entity in query_entities:
                query_emb = self.semantic_model.encode(query_entity)

                # Find best matches for this query entity
                matches = []
                for path_entity in nodes:
                    if path_entity in self.entity_embeddings:
                        path_emb = self.entity_embeddings[path_entity]
                        similarity = np.dot(query_emb, path_emb) / (
                                np.linalg.norm(query_emb) * np.linalg.norm(path_emb) + 1e-8
                        )
                        if similarity >= 0.5:  # Only include significant matches
                            matches.append((path_entity, similarity))

                # Sort matches by similarity
                matches.sort(key=lambda x: x[1], reverse=True)

                # Show top 3 matches at most
                if matches:
                    match_str = ", ".join([f"{entity} ({sim:.2f})" for entity, sim in matches[:3]])
                    explanation += f"  '{query_entity}' matches: {match_str}\n"

            explanation += "\n"

        # Add edge descriptions
        explanation += "Relationships:\n"
        for i, edge in enumerate(edges):
            source = edge['source']
            target = edge['target']
            rel_type = edge['relationship_type']
            rel_desc = edge['relationship']

            explanation += f"  {i + 1}. {source} → {target} ({rel_type}): {rel_desc}\n"

        return explanation


def main():
    parser = argparse.ArgumentParser(description='Filter relevant paths from a database schema graph')
    parser.add_argument('paths_file', help='JSON file containing paths')
    parser.add_argument('--query', '-q', help='Natural language query')
    parser.add_argument('--filters', '-f', nargs='+',
                        choices=['semantic', 'keywords', 'node_types', 'relationships', 'endpoints', 'ranking', 'all'],
                        default=['all'], help='Filters to apply')
    parser.add_argument('--max-results', '-m', type=int, default=10,
                        help='Maximum number of results to return')
    parser.add_argument('--explain', '-e', action='store_true',
                        help='Provide detailed explanations of paths')
    parser.add_argument('--semantic-model', help='Name of sentence-transformers model to use')
    parser.add_argument('--semantic-threshold', type=float, default=0.5,
                        help='Similarity threshold for semantic matching (0-1)')
    parser.add_argument('--no-semantic', action='store_true',
                        help='Disable semantic similarity matching')
    parser.add_argument('--show-matches', action='store_true',
                        help='Show semantic matches in path explanations')

    args = parser.parse_args()

    # Load and initialize the path filter
    path_filter = SchemaGraphPathFilter(
        args.paths_file,
        enable_semantic=not args.no_semantic,
        semantic_model=args.semantic_model,
        similarity_threshold=args.semantic_threshold
    )

    # Process filters list
    if 'all' in args.filters:
        filters = None  # Use all filters
    else:
        filters = args.filters

    # Filter paths based on query
    if args.query:
        relevant_paths = path_filter.filter_paths_for_query(
            args.query,
            filters=filters,
            max_results=args.max_results,
            use_semantic=not args.no_semantic
        )

        print(f"\nFound {len(relevant_paths)} relevant paths for query: '{args.query}'")

        # Get query entities for explanation if showing matches
        query_entities = None
        if args.show_matches:
            query_components = path_filter.extract_query_components(args.query)
            query_entities = (
                    query_components.get('tables', []) +
                    query_components.get('columns', []) +
                    query_components.get('keywords', [])
            )

        # Display results
        for i, path in enumerate(relevant_paths):
            print(f"\n--- Result {i + 1} ---")
            if args.explain:
                print(path_filter.explain_path(path, query_entities if args.show_matches else None))
            else:
                print(f"Path: {' → '.join(path['path'])}")
    else:
        print("No query provided. Please use --query to specify a query.")


if __name__ == "__main__":
    main()
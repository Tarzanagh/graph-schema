"""
flow_diffusion.py - Implementation of the Flow Diffusion algorithm
"""

import math
import re
import random
from collections import defaultdict

class WeightedFlowDiffusion:
    """
    Implementation of the Weighted Flow Diffusion algorithm for finding
    relevant subclusters in a database schema graph.
    """
    
    def __init__(self, gamma=0.02, max_iterations=30):
        """
        Initialize the weighted flow diffusion algorithm.
        
        Args:
            gamma: Parameter for Gaussian kernel weighting (default: 0.02)
            max_iterations: Maximum iterations for flow diffusion (default: 30)
        """
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.cache = {}  # For memoization
    
    def compute_query_relevance(self, graph, query):
        """
        Compute query relevance scores for all nodes using optimized methods.
        
        Args:
            graph: The schema graph
            query: The query text
            
        Returns:
            Dictionary mapping nodes to their query relevance scores
        """
        # Cache key for memoization
        cache_key = f"relevance_{hash(query)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        relevance_scores = {}
        
        # Scan all nodes once
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', '')
            
            if node_type == 'table':
                # For table nodes, use table name for relevance
                node_terms = set(re.findall(r'\b\w+\b', node_id.lower()))
                relevance = len(query_terms.intersection(node_terms)) * 2  # Give tables higher weight
                
            elif node_type == 'column':
                # For column nodes, use column name
                table = attrs.get('table', '')
                column = attrs.get('column_name', '')
                
                # Combine both table and column terms
                node_terms = set(re.findall(r'\b\w+\b', column.lower()))
                table_terms = set(re.findall(r'\b\w+\b', table.lower()))
                
                # Weight column matches higher than table matches for columns
                col_overlap = len(query_terms.intersection(node_terms)) * 1.5
                table_overlap = len(query_terms.intersection(table_terms)) * 0.5
                
                relevance = col_overlap + table_overlap
            else:
                relevance = 0
            
            # Bonus for data type mentions in query
            if node_type == 'column' and attrs.get('data_type'):
                data_type = attrs.get('data_type', '').lower()
                if data_type in query.lower():
                    relevance += 0.5
            
            # Bonus for primary keys as they're often important for joins
            if attrs.get('is_primary_key', False):
                relevance += 0.3
                
            relevance_scores[node_id] = relevance
        
        # Cache the result
        self.cache[cache_key] = relevance_scores
        return relevance_scores
    
    def find_seed_nodes(self, graph, query, limit=3):
        """
        Find the best seed nodes to start flow diffusion from.
        
        Args:
            graph: The schema graph
            query: The query text
            limit: Maximum number of seed nodes to return
            
        Returns:
            List of (node_id, score) tuples for the best seed nodes
        """
        # Compute query relevance
        node_relevance = self.compute_query_relevance(graph, query)
        
        # Find highest scoring nodes
        scored_nodes = [(node, score) for node, score in node_relevance.items() if score > 0]
        top_nodes = sorted(scored_nodes, key=lambda x: x[1], reverse=True)[:limit]
        
        return top_nodes
    
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
        
        # Set sink capacities (here using uniform capacity of 1.0)
        t = {node_id: 1.0 for node_id in nodes}
        
        # Run flow diffusion for specified iterations
        for _ in range(self.max_iterations):
            # Find nodes with excess mass
            overflow_nodes = [node for node, mass in m.items() if mass > t[node]]
            
            if not overflow_nodes:
                break
                
            # Pick a random overflow node
            i = overflow_nodes[0]  # Deterministic for better reproducibility
            
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
                edge_weight = graph[i][j].get('semantic_weight', 
                             graph[i][j].get('weight', 1.0))
                
                # Update mass at neighbor
                m[j] += excess * edge_weight / w_i
        
        return x
    
    def find_subcluster(self, graph, node_importance, min_importance=0.1):
        """
        Identify a subcluster of important nodes based on flow diffusion results.
        
        Args:
            graph: The graph representation
            node_importance: Dictionary mapping nodes to importance scores
            min_importance: Minimum importance threshold for inclusion
            
        Returns:
            Set of node IDs in the identified subcluster
        """
        # Filter nodes with sufficient importance
        active_nodes = {node: score for node, score in node_importance.items() if score > min_importance}
        
        if not active_nodes:
            return set()
        
        # Select core nodes (highest importance)
        sorted_nodes = sorted(active_nodes.items(), key=lambda x: x[1], reverse=True)
        core_nodes = {node for node, _ in sorted_nodes[:min(5, len(sorted_nodes))]}
        
        # Expand to include one-hop neighbors with positive importance
        subcluster = set(core_nodes)
        for node in core_nodes:
            for neighbor in graph.neighbors(node):
                if node_importance.get(neighbor, 0) > 0:
                    subcluster.add(neighbor)
        
        return subcluster
    
    def find_relevant_subclusters(self, graph, query, limit=3):
        """
        Find relevant subclusters for a query using weighted flow diffusion.
        
        Args:
            graph: The schema graph
            query: The query text
            limit: Maximum number of subclusters to identify
            
        Returns:
            List of subclusters (each a set of node IDs)
        """
        # Find the best seed nodes
        seed_nodes = self.find_seed_nodes(graph, query)
        
        if not seed_nodes:
            return []
        
        subclusters = []
        
        # Find subclusters from different seed nodes
        for seed_node, _ in seed_nodes[:limit]:
            # Run flow diffusion from this seed
            node_importance = self.flow_diffusion(graph, seed_node)
            
            # Identify the subcluster
            subcluster = self.find_subcluster(graph, node_importance)
            
            if subcluster:
                subclusters.append(subcluster)
        
        # Deduplicate if subclusters overlap too much
        unique_subclusters = []
        for subcluster in subclusters:
            # Check if this subcluster is too similar to existing ones
            is_unique = True
            for existing in unique_subclusters:
                overlap = len(subcluster.intersection(existing)) / len(subcluster.union(existing))
                if overlap > 0.7:  # More than 70% overlap
                    is_unique = False
                    break
            
            if is_unique:
                unique_subclusters.append(subcluster)
        
        return unique_subclusters
    
    def get_tables_from_subcluster(self, graph, subcluster):
        """
        Extract relevant tables from a subcluster.
        
        Args:
            graph: The schema graph
            subcluster: Set of node IDs in the subcluster
            
        Returns:
            Set of table names
        """
        tables = set()
        
        for node in subcluster:
            if graph.nodes[node].get('type') == 'table':
                tables.add(node)
            elif graph.nodes[node].get('type') == 'column':
                # Get the table from column node
                table = node.split('.')[0] if '.' in node else None
                if table:
                    tables.add(table)
        
        return tables
    
    def get_columns_from_subcluster(self, graph, subcluster):
        """
        Extract relevant columns from a subcluster.
        
        Args:
            graph: The schema graph
            subcluster: Set of node IDs in the subcluster
            
        Returns:
            Dictionary mapping tables to their relevant columns
        """
        columns_by_table = defaultdict(list)
        
        for node in subcluster:
            if graph.nodes[node].get('type') == 'column':
                if '.' in node:
                    table, column = node.split('.')
                    columns_by_table[table].append(column)
        
        return dict(columns_by_table)
    
    def get_joins_from_subcluster(self, graph, subcluster):
        """
        Extract potential joins from a subcluster.
        
        Args:
            graph:

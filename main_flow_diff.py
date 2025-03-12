"""
flow_diffusion.py - Flow Diffusion for subcluster finding in database schemas
"""

import math
import re
import random
from collections import defaultdict
import os
import json
import argparse
import time

from init_schema_graph import SchemaGraphBuilder
from LLM_service import LLMService

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
        import pdb; pdb.set_trace()
        # Run flow diffusion for specified iterations
        for _ in range(self.max_iterations):
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
                edge_weight = graph[i][j].get('semantic_weight',  graph[i][j].get('weight', 1.0))
                
                # Update mass at neighbor
                m[j] += excess * edge_weight / w_i
        
        return x
    
    def find_subcluster(self, graph, node_importance, min_importance=0.1, max_size=15):
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
        # Filter nodes with sufficient importance
        active_nodes = {node: score for node, score in node_importance.items() if score > min_importance}
        
        if not active_nodes:
            return set()
        
        # Sort by importance and take top nodes to form core of subcluster
        sorted_nodes = sorted(active_nodes.items(), key=lambda x: x[1], reverse=True)
        core_nodes = {node for node, _ in sorted_nodes[:min(5, len(sorted_nodes))]}
        
        # Expand to include one-hop neighbors with positive importance
        subcluster = set(core_nodes)
        for node in core_nodes:
            for neighbor in graph.neighbors(node):
                if node_importance.get(neighbor, 0) > 0:
                    subcluster.add(neighbor)
        
        # Limit subcluster size if needed
        if len(subcluster) > max_size:
            # Keep most important nodes
            subcluster = {node for node, _ in sorted_nodes[:max_size]}
        
        return subcluster
    
    def find_multiple_subclusters(self, graph, query, num_clusters=4):
        """
        Find multiple non-overlapping subclusters for a query using weighted flow diffusion.
        
        Args:
            graph: The schema graph
            query: The query text
            num_clusters: Number of subclusters to find
            
        Returns:
            List of subclusters (each a set of node IDs)
        """
        # Find the best seed nodes
        seed_nodes = self.find_seed_nodes(graph, query, limit=num_clusters*2)
        
        if not seed_nodes:
            return []
        
        subclusters = []
        used_nodes = set()
        
        # Try to find the requested number of subclusters
        for seed_node, _ in seed_nodes:
            # Skip if seed node already in a subcluster
            if seed_node in used_nodes:
                continue
                
            # Run flow diffusion from this seed
            node_importance = self.flow_diffusion(graph, seed_node)
            
            # Prioritize nodes not already in other subclusters
            for node in used_nodes:
                node_importance[node] = node_importance.get(node, 0) * 0.2  # Reduce importance
            
            # Identify the subcluster
            subcluster = self.find_subcluster(graph, node_importance)
            
            if subcluster:
                subclusters.append(subcluster)
                used_nodes.update(subcluster)
            
            # Stop if we have enough subclusters
            if len(subclusters) >= num_clusters:
                break
        
        # If we still don't have enough subclusters, try with lower thresholds
        if len(subclusters) < num_clusters:
            for seed_node, _ in seed_nodes:
                if len(subclusters) >= num_clusters:
                    break
                    
                if seed_node in used_nodes:
                    continue
                    
                node_importance = self.flow_diffusion(graph, seed_node)
                
                # Use a lower threshold
                subcluster = self.find_subcluster(graph, node_importance, min_importance=0.05)
                
                if subcluster and not subcluster.issubset(used_nodes):
                    # Remove already used nodes
                    subcluster = subcluster - used_nodes
                    if len(subcluster) >= 3:  # Only add if still substantial
                        subclusters.append(subcluster)
                        used_nodes.update(subcluster)
        
        # If we still don't have enough, create dummy subclusters
        while len(subclusters) < num_clusters:
            # Create a dummy subcluster with unused high-degree nodes
            remaining_nodes = [n for n in graph.nodes() if n not in used_nodes]
            if not remaining_nodes:
                break
                
            # Sort by degree
            remaining_nodes.sort(key=lambda n: graph.degree(n), reverse=True)
            dummy_subcluster = set(remaining_nodes[:5])
            subclusters.append(dummy_subcluster)
            used_nodes.update(dummy_subcluster)
        
        return subclusters
    
    def extract_paths_from_subcluster(self, graph, subcluster, limit=10):
        """
        Extract meaningful paths from a subcluster.
        
        Args:
            graph: The graph
            subcluster: Set of node IDs in the subcluster
            limit: Maximum number of paths to return
            
        Returns:
            List of paths (as strings)
        """
        paths = []
        path_scores = {}
        
        # Find table nodes in the subcluster
        table_nodes = [node for node in subcluster if graph.nodes[node].get('type') == 'table']
        
        # Find column nodes in the subcluster
        column_nodes = [node for node in subcluster if graph.nodes[node].get('type') == 'column']
        
        # Start paths from table nodes
        for start_node in table_nodes:
            # BFS to find paths through the subcluster
            visited = set([start_node])
            queue = [(start_node, [start_node], 0)]  # (node, path, length)
            
            while queue and len(paths) < limit * 2:
                current, path, length = queue.pop(0)
                
                # If path reaches a certain length or ends at another table, consider it complete
                if (length >= 2 and current in table_nodes and current != start_node) or length >= 4:
                    # Convert path to string format
                    path_str = '.'.join(path)
                    paths.append(path_str)
                    # Score based on node importance and path length
                    path_scores[path_str] = 1.0 / (length + 1)  # Shorter paths score higher
                    continue
                
                # Continue BFS
                for neighbor in graph.neighbors(current):
                    if neighbor in subcluster and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor], length + 1))
        
        # If not enough paths yet, try starting from column nodes
        if len(paths) < limit:
            for start_node in column_nodes[:min(5, len(column_nodes))]:
                visited = set([start_node])
                queue = [(start_node, [start_node], 0)]
                
                while queue and len(paths) < limit * 2:
                    current, path, length = queue.pop(0)
                    
                    if length >= 2:
                        path_str = '.'.join(path)
                        if path_str not in paths:
                            paths.append(path_str)
                            path_scores[path_str] = 0.8 / (length + 1)  # Slightly lower base score
                    
                    if length < 4:
                        for neighbor in graph.neighbors(current):
                            if neighbor in subcluster and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, path + [neighbor], length + 1))
        
        # Sort paths by score and take top ones
        scored_paths = [(p, path_scores.get(p, 0)) for p in paths]
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        top_paths = [p for p, _ in scored_paths[:limit]]
        
        # If still not enough paths, generate some simple ones
        if len(top_paths) < limit:
            # Create simple table.column paths
            for table in table_nodes:
                for col in graph.neighbors(table):
                    if col in column_nodes and len(top_paths) < limit:
                        path = f"{table}.{col.split('.')[-1]}"
                        if path not in top_paths:
                            top_paths.append(path)
        
        # If still not enough, add some placeholder paths
        while len(top_paths) < limit:
            placeholder = f"path_{len(top_paths)+1}"
            top_paths.append(placeholder)
        
        return top_paths

#!/usr/bin/env python
"""
main.py - Main script for query processing with subclusters and SQL generation
"""


def main():
    """Main function to run the flow diffusion-based query processor."""
    # Load configuration from JSON file
    config_file =  "config.json"    
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Required keys in config
    required_keys = ['api_url', 'api_key']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Initialize LLM service
    llm_service = LLMService(
        api_url=config['api_url'], 
        api_key=config['api_key'],
        model=config.get('model')  # Optional
    )
    

    # load schema graph 
    graph = SchemaGraphBuilder.load_graph('enhanced_schema_graph.json')
    schema_details = SchemaGraphBuilder.extract_schema_details(graph)

    query = config['query']
    print('query:', query)

    # Decompose the query
    subqueries = llm_service.decompose_query(query, schema_details) 

    # Step 2: Initialize the flow diffusion algorithm
    flow_diffusion = WeightedFlowDiffusion(gamma=0.02, max_iterations=30)
    
    # Step 3: Find seed nodes based on the all the subqueries
    for subquery in subqueries:
        print(f"\nProcessing subquery: {subquery}")
        seed_nodes = flow_diffusion.find_seed_nodes(graph, subquery, limit=3)
        print(f"Found {len(seed_nodes)} seed nodes:")
        for node, score in seed_nodes:
            print(f"  - {node} (score: {score:.2f})")

        # Find subclusters for the query
        best_seed, _ = seed_nodes[0]
        print(f"\nRunning flow diffusion from seed node: {best_seed}")
        node_importance = flow_diffusion.flow_diffusion(graph, best_seed)
        
        # Print top important nodes
        top_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop important nodes:")
        for node, importance in top_nodes:
            print(f"  - {node} (importance: {importance:.2f})")
        
        # Step 5: Find subclusters
        subclusters = flow_diffusion.find_multiple_subclusters(graph, subquery, num_clusters=1)
        print("subclusters:")
        print(subclusters)
        
        
    import pdb; pdb.set_trace()
    # Step 4: Find subclusters for the query


if __name__ == "__main__":
    main()


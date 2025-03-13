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

        #  "path": ["city → address → customer → rental → inventory → film → film_category → category"],
        
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
        t = {node_id: 0.1 for node_id in nodes}  # Reduced from 1.0 to 0.5
        
        # Run flow diffusion for specified iterations - increase max iterations
        max_iterations = self.max_iterations * 2  # Double the iterations
        
        for _ in range(max_iterations):
            import pdb; pdb.set_trace()

            # Find nodes with excess mass
            overflow_nodes = sorted([node for node, mass in m.items() if mass > t[node]])
            print(overflow_nodes)
            if not overflow_nodes:
                break
              
            # Pick an overflow node (deterministic for reproducibility)
            i = overflow_nodes[0]
            print('...mass',m[i], i) 
            xx = [-1]
            for key in m.keys():
                if m[key]!= 0:
                    xx.append((key,round(m[key],3) ))
            if len(xx)>1: print(xx)


            
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
    

    # load schema graph & query
    graph = SchemaGraphBuilder.load_graph('enhanced_schema_graph.json')
    schema_details = SchemaGraphBuilder.extract_schema_details(graph)
    query = config['query']
    print('query:', query)


    # Initialize the flow diffusion algorithm
    flow_diffusion = WeightedFlowDiffusion(gamma=0.02, max_iterations=30)



    def process_query(query):

        # Find seed nodes
        seed_nodes = flow_diffusion.find_seed_nodes(graph, query, limit=3)
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
    
        # Find subclusters
        subclusters = flow_diffusion.find_multiple_subclusters(graph, query, num_clusters=1)
        print("subclusters:")
        print(subclusters)
        return subclusters

    subclusters = process_query(query)
    
    import pdb; pdb.set_trace()


    # Decompose the query
    subqueries = llm_service.decompose_query(query, schema_details,2) 

    for subquery in subqueries:
        print(f"\nProcessing subquery: {subquery}")
        subclusters = process_query(subquery)
        
        


if __name__ == "__main__":
    main()


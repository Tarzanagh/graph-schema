#!/usr/bin/env python
"""
build_graph.py - Simple script to build and optionally visualize a schema graph
"""

import argparse
import os
import json
import time
import networkx as nx
from schema_graph_builder import SchemaGraphBuilder

def main():
    """Build a schema graph from a paths JSON file"""
    parser = argparse.ArgumentParser(description="Build a schema graph from a paths JSON file")
    parser.add_argument("--paths", "-p", required=True, help="Path to JSON file with paths")
    parser.add_argument("--output", "-o", help="Path to save the graph (JSON format)")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the graph")
    parser.add_argument("--db", "-d", help="Path to SQLite database file (alternative to paths)")
    
    args = parser.parse_args()
    
    # Check if paths file exists
    if args.paths and not os.path.exists(args.paths):
        print(f"Error: Paths file '{args.paths}' not found")
        return 1
    
    # Check if database file exists
    if args.db and not os.path.exists(args.db):
        print(f"Error: Database file '{args.db}' not found")
        return 1
    
    # Build the graph
    start_time = time.time()
    
    if args.db:
        print(f"Building graph from database: {args.db}")
        graph = SchemaGraphBuilder.build_from_database(args.db)
    else:
        print(f"Building graph from paths file: {args.paths}")
        graph = SchemaGraphBuilder.build_from_paths(args.paths)
    
    print(f"Graph built in {time.time() - start_time:.2f} seconds")
    
    # Print some statistics
    table_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'table']
    column_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'column']
    
    print(f"\nGraph Statistics:")
    print(f"- Tables: {len(table_nodes)}")
    print(f"- Columns: {len(column_nodes)}")
    print(f"- Edges: {graph.number_of_edges()}")
    
    # Print table information
    print("\nTables:")
    for table in sorted(table_nodes):
        table_columns = [n for n in graph.neighbors(table) if '.' in n and n.startswith(f"{table}.")]
        print(f"- {table} ({len(table_columns)} columns)")
    
    # Save the graph if requested
    if args.output:
        print(f"\nSaving graph to {args.output}")
        SchemaGraphBuilder.save_graph(graph, args.output)
    
    # Visualize the graph if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            print("\nVisualizing graph...")
            
            # Create a smaller graph with just tables for better visualization
            simplified_graph = nx.Graph()
            
            # Add table nodes
            for table in table_nodes:
                simplified_graph.add_node(table, type='table')
            
            # Add edges between tables that have a foreign key relationship
            for u, v, data in graph.edges(data=True):
                if data.get('relationship_type') == 'pk_fk_table':
                    if u in table_nodes and v in table_nodes:
                        simplified_graph.add_edge(u, v)
            
            # Position nodes using spring layout
            pos = nx.spring_layout(simplified_graph)
            
            # Set figure size
            plt.figure(figsize=(12, 10))
            
            # Draw nodes
            nx.draw_networkx_nodes(simplified_graph, pos, 
                                  node_size=2000, 
                                  node_color='lightblue')
            
            # Draw edges
            nx.draw_networkx_edges(simplified_graph, pos, 
                                  width=1.5, 
                                  alpha=0.7)
            
            # Draw labels
            nx.draw_networkx_labels(simplified_graph, pos, 
                                   font_size=12, 
                                   font_weight='bold')
            
            plt.title("Database Schema Graph (Tables Only)")
            plt.axis('off')
            
            # Save or show
            if args.output:
                image_path = os.path.splitext(args.output)[0] + ".png"
                plt.savefig(image_path)
                print(f"Graph visualization saved to {image_path}")
            else:
                plt.show()
                
        except ImportError:
            print("\nVisualization requires matplotlib. Install it with: pip install matplotlib")
    
    return 0

if __name__ == "__main__":
    exit(main())

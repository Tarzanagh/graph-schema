import networkx as nx
from flow_diffusion import WeightedFlowDiffusion

def run_flow_diffusion_example(db_file, query):
    """
    Run flow diffusion on a database schema
    
    Args:
        db_file: Path to SQLite database file
        query: Natural language query to analyze
    """
    print(f"Analyzing query: {query}")
    
    # Step 1: Build the graph from database
    # You can use the db_graph_builder script we created earlier
    from db_graph_builder import build_graph_from_database
    graph = build_graph_from_database(db_file)
    
    # Step 2: Initialize the flow diffusion algorithm
    flow_diffusion = WeightedFlowDiffusion(gamma=0.02, max_iterations=30)
    
    # Step 3: Find seed nodes based on the query
    seed_nodes = flow_diffusion.find_seed_nodes(graph, query, limit=3)
    print(f"Found {len(seed_nodes)} seed nodes:")
    for node, score in seed_nodes:
        print(f"  - {node} (score: {score:.2f})")
    
    # Step 4: Run flow diffusion from the best seed node
    if seed_nodes:
        best_seed, _ = seed_nodes[0]
        print(f"\nRunning flow diffusion from seed node: {best_seed}")
        node_importance = flow_diffusion.flow_diffusion(graph, best_seed)
        
        # Print top important nodes
        top_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop important nodes:")
        for node, importance in top_nodes:
            print(f"  - {node} (importance: {importance:.2f})")
        
        # Step 5: Find subclusters
        subclusters = flow_diffusion.find_multiple_subclusters(graph, query, num_clusters=4)
        print(f"\nFound {len(subclusters)} subclusters:")
        
        for i, subcluster in enumerate(subclusters):
            print(f"\nSubcluster {i+1} ({len(subcluster)} nodes):")
            
            # Print table nodes in this subcluster
            table_nodes = [node for node in subcluster if graph.nodes[node].get('type') == 'table']
            print(f"  Tables: {', '.join(table_nodes[:5])}" + 
                  (f" and {len(table_nodes)-5} more" if len(table_nodes) > 5 else ""))
            
            # Step 6: Extract paths from subcluster
            paths = flow_diffusion.extract_paths_from_subcluster(graph, subcluster)
            print(f"  Paths:")
            for j, path in enumerate(paths[:5]):
                print(f"    - {path}")
            if len(paths) > 5:
                print(f"    - ... and {len(paths)-5} more paths")
    else:
        print("No seed nodes found for this query.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python run_flow_diffusion.py <database_file> <query>")
        sys.exit(1)
        
    db_file = sys.argv[1]
    query = sys.argv[2]
    run_flow_diffusion_example(db_file, query)

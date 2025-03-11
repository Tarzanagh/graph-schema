#!/usr/bin/env python3
"""
Main script to run the JSON Path Analyzer with configuration from a JSON file.
"""

import json
import os
import sys
from json_path_analyzer import QueryProcessor

def main():
    """Main function to run the Path Analyzer with configuration from a JSON file."""
    # Load configuration from JSON file
    config_file = "config.json"
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Check if paths file exists
    paths_json_file = config.get("paths_file")
    if not os.path.exists(paths_json_file):
        print(f"Error: Paths file not found: {paths_json_file}")
        print("Make sure to generate the paths file first.")
        sys.exit(1)
    
    # Get API key from config or environment
    api_key = config.get("api_key") or os.environ.get("API_KEY", "")
    if not api_key:
        print("Error: API key is required. Set it in the config file or as API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize the query processor with the configuration
    processor = QueryProcessor(
        api_url=config["api_url"],
        api_key=api_key,
        model=config["model"]
    )
    
    # Process the query
    query = config["query"]
    print(f"Processing query: {query}")
    print(f"Using paths from: {paths_json_file}")
    results = processor.process_query(paths_json_file, query)
    
    # Print the results
    print_results(results)
    
    # Also save results to file
    output_file = config.get("output_file", "query_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def print_results(results):
    """Print results in a readable format."""
    print("\n" + "="*80)
    print(f"QUERY: {results['original_query']}")
    print("="*80)
    
    print("\nSUBQUERIES:")
    for i, subquery in enumerate(results["subqueries"]):
        print(f"  {chr(65+i)}. {subquery}")
    
    for result in results["subquery_results"]:
        print("\n" + "-"*80)
        print(f"SUBQUERY {result['subquery_id']}: {result['subquery_text']}")
        print("-"*80)
        
        print("\nTOP RELEVANT PATHS:")
        for i, path in enumerate(result["top_paths"]):
            relevance = result["path_relevance_scores"].get(path, 0.0)
            print(f"  {i+1}. {path} (Relevance: {relevance:.2f})")
        
        print("\nTOP SQL STATEMENTS:")
        for i, sql in enumerate(result["top_sqls"]):
            rewards = result["sql_rewards"].get(sql, {})
            print(f"\n  SQL #{i+1}:")
            print(f"  {sql.replace(chr(10), chr(10)+'  ')}")
            print("\n  Rewards:")
            for reward_type, score in rewards.items():
                print(f"    {reward_type}: {score:.2f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
main.py - Main script for query processing with subclusters and SQL generation
"""

import os
import json
import argparse
import time

from revised_query_processor import QueryProcessor

def main():
    """Main function to run the flow diffusion-based query processor."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process queries using Flow Diffusion-based Subcluster Analyzer')
    parser.add_argument('--config', type=str, help='Path to config file with API credentials')
    parser.add_argument('--paths', type=str, required=True, help='Path to JSON file containing paths')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--output', type=str, help='Path to save results as JSON')
    args = parser.parse_args()
    
    # Load configuration from JSON file
    config_file = args.config or "config.json"    
    
    # Check if paths file exists
    paths_json_file = args.paths
    if not os.path.exists(paths_json_file):
        print(f"Error: Paths file '{paths_json_file}' not found")
        return
    
    # Initialize the query processor with the configuration
    processor = QueryProcessor.from_config(config_file)
    
    # Process the query
    query = args.query
    if not query:
        query = input("Enter your query: ")
        
    print(f"Processing query: {query}")
    print(f"Using paths from: {paths_json_file}")
    
    start_time = time.time()
    results = processor.process_query(paths_json_file, query)
    total_time = time.time() - start_time
    
    print(f"Total processing time: {total_time:.2f} seconds")
    
    # Print summary results
    print("\n" + "="*80)
    print(f"QUERY: {results['original_query']}")
    print("="*80)
    
    print("\nSUBQUERIES:")
    for i, subquery in enumerate(results["subqueries"]):
        print(f"  {chr(65+i)}. {subquery}")
    
    for result in results["subquery_results"]:
        print("\n" + "-"*80)
        print(f"SUBQUERY {result['subquery_id']}: {result['subquery_text']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print("-"*80)
        
        print("\nSUBCLUSTERS:")
        for subcluster_result in result["subclusters"]:
            print(f"  {subcluster_result['subcluster_id']} (size: {subcluster_result['subcluster_size']})")
            
            print("\n  TOP SQL STATEMENTS:")
            for i, sql in enumerate(subcluster_result["sqls"][:3]):  # Show top 3 for brevity
                rewards = subcluster_result["rewards"].get(sql, {})
                avg_reward = sum(rewards.values()) / len(rewards) if rewards else 0
                
                print(f"\n  SQL #{i+1} (Avg Reward: {avg_reward:.2f}):")
                print(f"  {sql.replace(chr(10), chr(10)+'  ')}")
                print("\n  Rewards:")
                for reward_type, score in rewards.items():
                    print(f"    {reward_type}: {score:.2f}")
    
    # Also save results to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

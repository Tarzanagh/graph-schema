#!/usr/bin/env python3
"""
Main script to run the JSON Path Analyzer.
This script executes the query processing pipeline on a given query using paths from a JSON file.
"""

import json
import os
import sys
import argparse
from json_path_analyzer import QueryProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process queries using Path Analyzer')
    parser.add_argument('--config', type=str, help='Path to config file with API credentials')
    parser.add_argument('--paths', type=str, required=True, help='Path to JSON file containing paths')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--output', type=str, help='Path to save results as JSON')
    return parser.parse_args()

def main():
    """Main function to run the Path Analyzer."""
    # Parse command line arguments
    args = parse_args()
    
    # Check if paths file exists
    if not os.path.exists(args.paths):
        print(f"Error: Paths file not found: {args.paths}")
        sys.exit(1)
    
    # Initialize the query processor
    if args.config and os.path.exists(args.config):
        processor = QueryProcessor.from_config(args.config)
    else:
        # Use environment variables for API credentials if no config file
        api_url = os.environ.get("API_URL", "https://api.openai.com/v1/chat/completions")
        api_key = os.environ.get("API_KEY", "")
        model = os.environ.get("MODEL", "gpt-4")
        
        if not api_key:
            print("Error: API key is required. Set it in the config file or as API_KEY environment variable.")
            sys.exit(1)
        
        processor = QueryProcessor(api_url, api_key, model)
    
    # Get the query
    query = args.query
    if not query:
        query = input("Enter your query: ")
    
    # Process the query
    print(f"Processing query: {query}")
    print(f"Using paths from: {args.paths}")
    results = processor.process_query(args.paths, query)
    
    # Print the results or save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print_results(results)

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

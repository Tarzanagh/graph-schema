#!/usr/bin/env python3

import json
import os
import sys
import argparse
import requests
import sqlparse
import difflib
import re

# Your existing classes would be here...

def main():
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

if __name__ == "__main__":
    main()

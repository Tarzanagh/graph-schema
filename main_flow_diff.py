#!/usr/bin/env python
"""
main.py - Main entry point for the database query processor
"""

import os
import sys
import time
import argparse
import json
from query_processor import QueryProcessor

def main():
    """Entry point for processing database queries"""
    parser = argparse.ArgumentParser(description='Process database queries using LLM and weighted flow diffusion')
    parser.add_argument('--db', '-d', required=True, help='Path to SQLite database file')
    parser.add_argument('--query', '-q', help='Natural language query to process')
    parser.add_argument('--llm-endpoint', '-e', required=True, help='URL endpoint for LLM API')
    parser.add_argument('--llm-key', '-k', help='API key for LLM (if required)')
    parser.add_argument('--execute', '-x', action='store_true', help='Execute the generated SQL')
    parser.add_argument('--output', '-o', help='Output file path for results (JSON format)')
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db):
        print(f"Error: Database file '{args.db}' not found.")
        return 1
    
    # Initialize the query processor
    processor = QueryProcessor(
        db_file=args.db,
        llm_endpoint=args.llm_endpoint,
        llm_api_key=args.llm_key
    )
    
    # Get query from command line or prompt user
    query = args.query
    if not query:
        query = input("Enter your query: ")
    
    # Process the query
    results = processor.process_query(query)
    
    # Print the results
    print("\n" + "="*50)
    print(f"Original query: {results['original_query']}")
    print(f"Decomposed into {len(results['subqueries'])} subqueries")
    print("="*50)
    
    for subquery_result in results['subquery_results']:
        print(f"\n{subquery_result['subquery_id']}: {subquery_result['subquery_text']}")
        print(f"Processing time: {subquery_result['processing_time']:.2f} seconds")
        
        # Print subcluster information
        print("\nRelevant subclusters identified:")
        for i, subcluster in enumerate(subquery_result['subclusters']):
            print(f"  Subcluster {i+1}:")
            print(f"    Tables: {', '.join(subcluster['tables'][:3])}{'...' if len(subcluster['tables']) > 3 else ''}")
            print(f"    Primary table: {subcluster['primary_table']}")
        
        # Print SQL
        if subquery_result['sql_statements']:
            print("\nGenerated SQL statements:")
            for i, sql in enumerate(subquery_result['sql_statements']):
                print(f"\nSQL {i+1}:\n{sql}")
                
                # Execute SQL if requested
                if args.execute:
                    print("\nExecuting SQL...")
                    result = processor.execute_sql(sql)
                    
                    if "error" in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"Results: {len(result['rows'])} rows")
                        if result['rows']:
                            print(f"Columns: {', '.join(result['columns'])}")
                            for j, row in enumerate(result['rows'][:5]):
                                print(f"Row {j+1}: {row}")
                            if len(result['rows']) > 5:
                                print(f"... and {len(result['rows']) - 5} more rows")
        else:
            print("\nNo SQL statements generated.")
        
        print("\n" + "-"*50)
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

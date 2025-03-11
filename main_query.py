"""
query_processor.py - Main module that orchestrates the entire query processing pipeline
"""
 
import sqlite3
import time
import os
from schema_graph import create_schema_graph
from llm_service import LLMService
from flow_diffusion import WeightedFlowDiffusion

class QueryProcessor:
    """
    Main class for processing natural language queries using a combination of
    LLM for decomposition and SQL generation, and flow diffusion for subcluster finding.
    """
    
    def __init__(self, db_file, llm_endpoint, llm_api_key=None):
        """
        Initialize the query processor.
        
        Args:
            db_file: Path to the SQLite database file
            llm_endpoint: URL for the LLM API
            llm_api_key: API key for LLM authentication
        """
        # Create schema graph
        print(f"Creating schema graph from database: {db_file}")
        self.graph, self.metadata = create_schema_graph(db_file)
        
        # Initialize components
        print(f"Initializing LLM service with endpoint: {llm_endpoint}")
        self.llm_service = LLMService(llm_endpoint, llm_api_key)
        
        print("Initializing weighted flow diffusion")
        self.flow_diffusion = WeightedFlowDiffusion()
        
        # Store database file for later use
        self.db_file = db_file
        
        # Enhance edge weights with LLM semantic analysis
        print("Enhancing graph with semantic edge weights...")
        self.graph = self.llm_service.enhance_edge_semantics(self.graph, self.metadata)
        print("Graph enhancement complete")
    
    def process_query(self, query):
        """
        Process a natural language query to generate SQL statements.
        
        Args:
            query: The natural language query
            
        Returns:
            Dictionary with processing results including subqueries and generated SQL
        """
        print(f"\nProcessing query: {query}")
        start_time = time.time()
        
        # Step 1: Use LLM to decompose the query into subqueries
        print("Decomposing query into subqueries...")
        subqueries = self.llm_service.decompose_query(
            query, self.metadata['schema_details']
        )
        print(f"Query decomposed into {len(subqueries)} subqueries")
        
        # Step 2 & 3: Process each subquery to find subclusters and generate SQL
        subquery_results = []
        
        for i, subquery in enumerate(subqueries):
            subquery_start = time.time()
            print(f"\nProcessing subquery {i+1}: {subquery}")
            
            # Find relevant subclusters using weighted flow diffusion
            print("Finding relevant subclusters...")
            subclusters = self.flow_diffusion.find_relevant_subclusters(
                self.graph, subquery, limit=3
            )
            print(f"Found {len(subclusters)} relevant subclusters")
            
            # Extract structured information from subclusters
            subcluster_info = []
            for j, subcluster in enumerate(subclusters):
                info = self.flow_diffusion.extract_subcluster_info(self.graph, subcluster)
                subcluster_info.append(info)
                
                print(f"Subcluster {j+1}:")
                print(f"  - Tables: {info['tables']}")
                print(f"  - Primary table: {info['primary_table']}")
                print(f"  - Columns: {sum(len(cols) for cols in info['columns_by_table'].values())} columns across {len(info['columns_by_table'])} tables")
                print(f"  - Joins: {len(info['joins'])} potential joins")
            
            # Use LLM to generate SQL for this subquery (without using paths)
            print("Generating SQL statements...")
            sql_statements = self.llm_service.generate_sqls_from_subquery(
                subquery, self.metadata['schema_details']
            )
            print(f"Generated {len(sql_statements)} SQL statements")
            
            # Combine results for this subquery
            subquery_results.append({
                "subquery_id": f"Subquery {chr(65 + i)}",  # A, B, C, etc.
                "subquery_text": subquery,
                "subclusters": subcluster_info,
                "sql_statements": sql_statements,
                "processing_time": time.time() - subquery_start
            })
        
        # Return the processing results
        result = {
            "original_query": query,
            "subqueries": subqueries,
            "subquery_results": subquery_results,
            "total_processing_time": time.time() - start_time
        }
        
        print(f"\nQuery processing completed in {result['total_processing_time']:.2f} seconds")
        return result
    
    def execute_sql(self, sql):
        """
        Execute SQL statement and return results.
        
        Args:
            sql: SQL statement to execute
            
        Returns:
            Dictionary with columns and rows or error message
        """
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Return both column names and rows
            return {
                "columns": column_names,
                "rows": rows
            }
        except Exception as e:
            return {
                "error": str(e)
            }

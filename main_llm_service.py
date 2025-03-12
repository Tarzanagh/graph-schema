import os
from llm_service import LLMService

def main():
    # Configure your LLM API endpoint
    # For Ollama (local deployment)
    api_endpoint = "http://localhost:11434/api/generate"
    
    # For OpenAI or other cloud APIs
    # api_endpoint = "https://api.openai.com/v1/completions"
    # api_key = os.getenv("OPENAI_API_KEY")  # Recommended to use environment variables
    
    # Initialize LLM Service
    llm_service = LLMService(api_endpoint)
    
    # Example 1: Simple LLM Call
    print("--- Example 1: Basic LLM Call ---")
    prompt = "Write a two-sentence explanation of machine learning"
    response = llm_service.call_llm(prompt, max_tokens=200)
    print("LLM Response:", response)
    
    # Example 2: Query Decomposition
    print("\n--- Example 2: Query Decomposition ---")
    schema_details = {
        "tables": [
            {
                "name": "customers",
                "columns": ["customer_id", "name", "email", "join_date"]
            },
            {
                "name": "orders",
                "columns": ["order_id", "customer_id", "total_amount", "order_date"]
            }
        ]
    }
    query = "Find the top customers who made the most purchases in the last quarter"
    subqueries = llm_service.decompose_query(query, schema_details)
    print("Decomposed Subqueries:")
    for i, subquery in enumerate(subqueries, 1):
        print(f"{i}. {subquery}")
    
    # Example 3: SQL Generation
    print("\n--- Example 3: SQL Query Generation ---")
    sql_queries = llm_service.generate_sqls_from_subquery(
        subquery="Find the top customers who made the most purchases in the last quarter", 
        schema_details=schema_details
    )
    print("Generated SQL Queries:")
    for i, sql in enumerate(sql_queries, 1):
        print(f"SQL {i}:\n{sql}\n")

if __name__ == "__main__":
    main()

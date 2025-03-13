"""
llm_service.py - Module for interacting with LLM APIs (like Llama)
"""
import os
import requests
import re
import json
from typing import List, Dict, Any, Optional
from init_schema_graph import SchemaGraphBuilder
from GraphBuild import visualize_schema_graph, visualize_schema_graph_communities

class LLMService:
    """Service for interacting with LLM APIs."""
    
    def __init__(self, 
                 api_url: str, 
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None):
        """
        Initialize the LLM service.
        
        Args:
            api_url (str): The URL of the LLM API
            api_key (str, optional): API key for authentication
            model (str, optional): Specific model to use
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        
        # Validate critical parameters
        if not self.api_url:
            raise ValueError("API URL must be provided")
        if not self.api_key:
            raise ValueError("API key must be provided or set in environment variable LLM_API_KEY")
    
    def call_llm(self, 
                prompt: str, 
                max_tokens: int = 1000, 
                temperature: float = 0.3) -> str:
        """
        Make a call to the LLM API.
        
        Args:
            prompt (str): The prompt text to send to the LLM
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Text response from the LLM
        """
        try:
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert at solving assigned tasks."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated subqueries from the response
            content = response_data["choices"][0]["message"]["content"]
            return content
        

            
        except requests.RequestException as e:
            print(f"Error calling LLM API: {e}")

    def decompose_query(self, query, schema_details, nquery):
        """
        Use LLM to decompose a complex query into simpler subqueries.
        
        Args:
            query: The original natural language question
            schema_details: Database schema information
            
        Returns:
            List of subqueries
        """
        # Format the schema details
        schema_text = SchemaGraphBuilder.format_schema_for_prompt(schema_details)
        
        # Create prompt for LLM
        prompt = f"""
        Your task is to decompose a complex user query into simpler user subqueries.
        
        {schema_text}
        
        QUERY: {query}
        
        Break this down into {nquery} distinct user subqueries, where each subquery addresses a specific part of the overall question.
        Format your response as:
        <SUBQUERIES>
        1. First subquery text
        2. Second subquery text
        ...
        </SUBQUERIES>
        
        Only include the user subqueries within the tags - no explanations or other text.
        """
        
        # Call LLM
        llm_response = self.call_llm(prompt, max_tokens=500, temperature=0.3)
        
        # Parse subqueries from the response
        subqueries_match = re.search(r'<SUBQUERIES>(.*?)</SUBQUERIES>', llm_response, re.DOTALL)
        
        if subqueries_match:
            subqueries_text = subqueries_match.group(1).strip()
            subquery_list = re.findall(r'^\d+\.\s*(.*?)$', subqueries_text, re.MULTILINE)
            
            if subquery_list:
                return subquery_list
        
        # Fallback: if parsing fails, return the original query as a single subquery
        return [query]
    
    def enhance_edge_semantics(self, graph, metadata):
        """
        Use LLM to enhance graph edge weights based on semantic relationships between columns.
        
        Args:
            graph: NetworkX graph of the database schema
            metadata: Dictionary containing schema metadata
            
        Returns:
            Updated graph with semantically enhanced edge weights
        """
        enhanced_graph = graph.copy()
        schema_text = SchemaGraphBuilder.format_schema_for_prompt(metadata)
        
        # Get all column pairs connected by edges
        column_pairs = []
        for source, target, data in graph.edges(data=True):
            if graph.nodes[source].get('type') == 'column' and graph.nodes[target].get('type') == 'column':
                rel_type = data.get('relationship_type', '')
                column_pairs.append((source, target, rel_type))
        
        # Process in batches to avoid too large prompts
        batch_size = 10
        for i in range(0, len(column_pairs), batch_size):
            batch = column_pairs[i:i+batch_size]
            print('batch:', batch)
            
            # Create prompt
            prompt = f"""
            Your task is to analyze semantic relationships between database columns and assign relationship strengths.
            
            {schema_text}
            
            For each pair of columns below, analyze their semantic relationship and assign a strength score from 0.1 to 2.0:
            - 0.1-0.5: Weak relationship (minimal semantic connection)
            - 0.6-1.0: Moderate relationship (some semantic connection)
            - 1.1-1.5: Strong relationship (clear semantic connection)
            - 1.6-2.0: Very strong relationship (direct semantic connection)
            
            Consider column names, data types, and their role in the database.
            
            COLUMN PAIRS:
            """
            
            for source, target, rel_type in batch:
                source_name = source if '.' not in source else source.split('.')[1]
                target_name = target if '.' not in target else target.split('.')[1]
                source_table = source.split('.')[0] if '.' in source else 'Unknown'
                target_table = target.split('.')[0] if '.' in target else 'Unknown'
                
                prompt += f"{source_table}.{source_name} <-> {target_table}.{target_name} (Relationship: {rel_type})\n"
            
            prompt += """
            For each pair, respond in this exact format:
            <PAIR>column1 <-> column2: strength_score</PAIR>
            
            For example:
            <PAIR>users.user_id <-> orders.user_id: 1.8</PAIR>
            """
            
            # Call LLM

            llm_response = self.call_llm(prompt, max_tokens=1000, temperature=0.2)
            
            # Parse results
            pair_matches = re.findall(r'<PAIR>(.*?)</PAIR>', llm_response, re.DOTALL)
            
            for pair_match in pair_matches:
                try:
                    # Extract columns and score
                    match = re.match(r'(.*?)\s+<->\s+(.*?):\s*([\d\.]+)', pair_match.strip())
                    if match:
                        col1_str, col2_str, score_str = match.groups()
                        
                        # Clean up column names to match graph format
                        col1 = col1_str.strip()
                        col2 = col2_str.strip()
                        
                        # Handle possible format variations
                        if '.' not in col1:
                            for source, _, _ in batch:
                                if source.endswith(col1):
                                    col1 = source
                                    break
                        
                        if '.' not in col2:
                            for _, target, _ in batch:
                                if target.endswith(col2):
                                    col2 = target
                                    break
                        
                        # Get existing edge and update weight
                        try:
                            score = float(score_str)
                            if enhanced_graph.has_edge(col1, col2):
                                enhanced_graph[col1][col2]['semantic_weight'] = score
                                # Combine with existing weight if any
                                if 'weight' in enhanced_graph[col1][col2]:
                                    enhanced_graph[col1][col2]['weight'] *= score
                                else:
                                    enhanced_graph[col1][col2]['weight'] = score
                        except ValueError:
                            continue
                except Exception as e:
                    print(f"Error processing pair {pair_match}: {e}")
                    continue
        
        return enhanced_graph
                
    def generate_sqls_from_subquery(self, subquery, schema_details):
        """
        Generate multiple SQL statements from a single subquery without using paths.
        
        Args:
            subquery: The subquery text
            schema_details: Database schema information
            
        Returns:
            List of SQL statements
        """
        # Format the schema details
        schema_text = SchemaGraphBuilder.format_schema_for_prompt(schema_details)
        
        # Create prompt for LLM
        prompt = f"""
        Your task is to generate SQL statements that answer a database query.
        
        {schema_text}
        
        QUERY: {subquery}
        
        Generate 3 different, valid SQL statements that correctly answer this query using the schema information above.
        Each SQL should be complete and executable. Ensure variety in the solutions:
        - First SQL: Simple approach with minimal JOINs
        - Second SQL: More optimized approach with appropriate indexes
        - Third SQL: Alternative approach using different tables or conditions
        
        Format your response as:
        <SQL_1>
        Your first SQL code here
        </SQL_1>
        
        <SQL_2>
        Your second SQL code here
        </SQL_2>
        
        <SQL_3>
        Your third SQL code here
        </SQL_3>
        
        Only include the SQL statements within the tags - no explanations or other text.
        """
        
        # Call LLM
        llm_response = self.call_llm(prompt, max_tokens=1000, temperature=0.5)
        
        sqls = []
        
        # Extract all SQL statements
        for i in range(1, 4):
            # Extract SQL from the tags
            sql_match = re.search(rf'<SQL_{i}>(.*?)</SQL_{i}>', llm_response, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1).strip()
                sqls.append(sql)
        
        return sqls




def main():
    
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

    query = config['query']
    print('query:', query)
    
    # Example LLM call
    # response = llm_service.call_llm("Write a two-sentence explanation of machine learning")
    # print(response)

    # load schema graph 
    graph = SchemaGraphBuilder.load_graph('schema_graph.json')
    schema_details = SchemaGraphBuilder.extract_schema_details(graph)

    # Decompose the query
    subqueries = llm_service.decompose_query(query, schema_details, 4) 
    import pdb; pdb.set_trace()

    # Enhanced graph with semantic similarties 
    enhanced_graph = llm_service.enhance_edge_semantics(graph, schema_details)
    SchemaGraphBuilder.save_graph(enhanced_graph, output_file = "enhanced_schema_graph.json")
    visualize_schema_graph(enhanced_graph)
    visualize_schema_graph_communities(enhanced_graph)
    enhanced_schema_details = SchemaGraphBuilder.extract_schema_details(enhanced_graph)

    # Decompose the query with enhanced graph
    subqueries_enhanced = llm_service.decompose_query(query, enhanced_schema_details, 4) 
    
    import pdb; pdb.set_trace()

    #Saving the results
    json_string = json.dumps(subqueries)
    with open("subqueries.json", "w") as f:
        f.write(json_string)

    json_string = json.dumps(subqueries_enhanced)
    with open("subqueries_enhanced.json", "w") as f:
        f.write(json_string)



 

if __name__ == "__main__":
    main()

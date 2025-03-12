"""
query_processor.py - Main class for query processing with subclusters and SQL generation
"""

import json
import re
import requests
import sqlparse
import difflib
import os
import time
import networkx as nx

class QueryProcessor:
    """Main class for query processing pipeline with decomposition, subcluster finding, and SQL generation."""
    
    def __init__(self, api_url, api_key, model):
        """
        Initialize the query processor.
        
        Args:
            api_url: URL for the LLM API
            api_key: API key for authentication
            model: Model name to use
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        
        # Initialize components for each stage of the pipeline
        self.query_decomposer = QueryDecomposer(api_url, api_key, model)
        self.subcluster_finder = SubclusterFinder()
        self.sql_generator = SQLGenerator(api_url, api_key, model)
    
    @classmethod
    def from_config(cls, config_file=None):
        """
        Create a QueryProcessor from a configuration file.
        
        Args:
            config_file: Path to a JSON configuration file with API settings
            
        Returns:
            QueryProcessor instance
        """
        # Default configuration
        config = {
            "api_url": "https://api.openai.com/v1/chat/completions",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "model": "gpt-4"
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        return cls(
            api_url=config["api_url"],
            api_key=config["api_key"],
            model=config["model"]
        )
    
    def build_schema_graph(self, paths_json_file):
        """
        Build a schema graph from paths JSON file.
        
        Args:
            paths_json_file: Path to JSON file containing available paths
            
        Returns:
            NetworkX graph representing the schema
        """
        # Load paths data
        paths_data = self._load_paths_json(paths_json_file)
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Extract nodes and edges from paths
        for path in paths_data:
            elements = path.split('.')
            
            # Add nodes
            for i, element in enumerate(elements):
                if '.' in element:  # This is a column
                    table, column = element.split('.')
                    node_id = element
                    node_type = 'column'
                    graph.add_node(node_id, type=node_type, table=table, column_name=column)
                else:  # This is a table
                    node_id = element
                    node_type = 'table'
                    graph.add_node(node_id, type=node_type)
                
                # Add edges between consecutive elements
                if i > 0:
                    prev_element = elements[i-1]
                    # Determine relationship type
                    rel_type = 'table_column'  # Default
                    
                    if '.' in prev_element and '.' in element:
                        rel_type = 'column_column'
                    elif '.' not in prev_element and '.' not in element:
                        rel_type = 'table_table'
                    elif '.' in prev_element and '.' not in element:
                        rel_type = 'column_table'
                    
                    graph.add_edge(prev_element, element, relationship_type=rel_type, weight=1.0)
        
        # Add reverse edges for undirected connectivity
        edges_to_add = []
        for u, v, data in graph.edges(data=True):
            edges_to_add.append((v, u, data.copy()))
        
        # Add reverse edges
        for u, v, data in edges_to_add:
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, **data)
        
        return graph
    
    def process_query(self, paths_json_file, question):
        """
        Process a natural language query through the pipeline:
        1. Build schema graph from paths
        2. Decompose query into subqueries (max 4) using LLM
        3. For each subquery, find subclusters using flow diffusion
        4. For each subcluster, generate 10 SQL statements
        5. Evaluate SQL statements with rewards
        
        Args:
            paths_json_file: Path to JSON file containing available paths
            question: Natural language question
            
        Returns:
            Dictionary with detailed processing results
        """
        print(f"Processing query: {question}")
        start_time = time.time()
        
        # Step 1: Build schema graph from paths
        print("Building schema graph...")
        graph = self.build_schema_graph(paths_json_file)
        print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Load paths data for SQL generation
        paths_data = self._load_paths_json(paths_json_file)
        
        # Step 2: Decompose the query into subqueries (max 4)
        print("Decomposing query into subqueries...")
        subqueries = self.query_decomposer.decompose_query(question, max_subqueries=4)
        print(f"Query decomposed into {len(subqueries)} subqueries")
        
        # Step 3, 4, 5: Process each subquery
        subquery_results = []
        
        for i, subquery in enumerate(subqueries):
            subquery_start = time.time()
            print(f"\nProcessing subquery {i+1}: {subquery}")
            
            # Step 3: Find subclusters for this subquery using flow diffusion
            print("Finding subclusters...")
            subcluster_results = self.subcluster_finder.find_subclusters(graph, subquery)
            subclusters = subcluster_results["subclusters"]
            print(f"Found {len(subclusters)} subclusters")
            
            # Create list to store results for each subcluster
            subcluster_detail_results = []
            
            # Step 4 & 5: For each subcluster, generate and evaluate SQL statements
            for j, subcluster in enumerate(subclusters):
                print(f"Processing subcluster {j+1} with {len(subcluster)} nodes")
                
                # Extract paths from subcluster
                paths = self.subcluster_finder.extract_paths_from_subcluster(graph, subcluster)
                print(f"Extracted {len(paths)} paths from subcluster")
                
                # Generate SQL statements for this subcluster and evaluate them
                print("Generating SQL statements...")
                sql_results = self.sql_generator.generate_top_subsqls(paths_data, subquery, paths, limit=10)
                print(f"Generated {len(sql_results['sqls'])} SQL statements")
                
                # Store subcluster results
                subcluster_detail_results.append({
                    "subcluster_id": f"Subcluster {j+1}",
                    "subcluster_size": len(subcluster),
                    "paths": paths,
                    "sqls": sql_results["sqls"],
                    "rewards": sql_results["rewards"]
                })
            
            # Combine results for this subquery
            subquery_results.append({
                "subquery_id": f"Subquery {chr(65 + i)}",  # A, B, C, etc.
                "subquery_text": subquery,
                "subclusters": subcluster_detail_results,
                "processing_time": time.time() - subquery_start
            })
        
        # Return the processing results
        result = {
            "original_query": question,
            "subqueries": subqueries,
            "subquery_results": subquery_results,
            "total_processing_time": time.time() - start_time
        }
        
        print(f"\nQuery processing completed in {result['total_processing_time']:.2f} seconds")
        return result
    
    def _load_paths_json(self, paths_json_file):
        """Load paths data from JSON file."""
        with open(paths_json_file, 'r') as file:
            return json.load(file)


class QueryDecomposer:
    """Component for decomposing a natural language query into subqueries."""
    
    def __init__(self, api_url, api_key, model):
        """Initialize the query decomposer."""
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
    
    def decompose_query(self, question, max_subqueries=4):
        """
        Decompose a complex query into simpler subqueries.
        
        Args:
            question: The original natural language question
            max_subqueries: Maximum number of subqueries to generate
            
        Returns:
            List of subqueries
        """
        prompt = f"""
        Please decompose the following complex query into simpler subqueries:
        
        QUERY: {question}
        
        Break this down into 2-{max_subqueries} distinct subqueries, where each subquery addresses a specific part of the overall question.
        Format your response as:
        <SUBQUERIES>
        1. First subquery text
        2. Second subquery text
        ...
        </SUBQUERIES>
        """
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert at breaking down complex queries into simpler components."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated subqueries from the response
            content = response_data["choices"][0]["message"]["content"]
            
            # Extract subqueries from the tags
            subqueries_match = re.search(r'<SUBQUERIES>(.*?)</SUBQUERIES>', content, re.DOTALL)
            
            if subqueries_match:
                subqueries_text = subqueries_match.group(1).strip()
                # Extract numbered items, removing the numbers
                subquery_list = re.findall(r'^\d+\.\s*(.*?)$', subqueries_text, re.MULTILINE)
                
                if subquery_list:
                    # Limit to max_subqueries
                    return subquery_list[:max_subqueries]
            
            # Fallback: if parsing fails, return the original query as a single subquery
            return [question]
                
        except Exception as e:
            print(f"Error decomposing query: {e}")
            # Fallback to a basic decomposition
            return [question]


class SubclusterFinder:
    """Component for finding subclusters using weighted flow diffusion."""
    
    def __init__(self):
        """Initialize the subcluster finder."""
        from flow_diffusion import WeightedFlowDiffusion
        self.flow_diffusion = WeightedFlowDiffusion()
    
    def find_subclusters(self, graph, subquery, num_clusters=4):
        """
        Find subclusters for a subquery using weighted flow diffusion.
        
        Args:
            graph: NetworkX graph of the database schema
            subquery: The subquery text
            num_clusters: Number of subclusters to find
            
        Returns:
            Dictionary with subclusters and their importance scores
        """
        # Find subclusters using flow diffusion
        subclusters = self.flow_diffusion.find_multiple_subclusters(graph, subquery, num_clusters)
        
        # Return the subclusters
        return {
            "subclusters": subclusters
        }
    
    def extract_paths_from_subcluster(self, graph, subcluster, limit=10):
        """
        Extract paths from a subcluster for SQL generation.
        
        Args:
            graph: NetworkX graph
            subcluster: Set of node IDs in the subcluster
            limit: Maximum number of paths to return
            
        Returns:
            List of paths
        """
        return self.flow_diffusion.extract_paths_from_subcluster(graph, subcluster, limit)


class SQLGenerator:
    """Component for generating SQL for subqueries based on relevant paths."""
    
    def __init__(self, api_url, api_key, model):
        """Initialize the SQL generator."""
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
    
    def generate_top_subsqls(self, paths_data, subquery, relevant_paths, limit=10):
        """
        Generate top SQL statements for a subquery based on relevant paths and automatically evaluate them.
        
        Args:
            paths_data: JSON data containing available paths
            subquery: The subquery text
            relevant_paths: List of relevant paths for this subquery
            limit: Maximum number of SQL statements to generate (default: 10)
            
        Returns:
            Dictionary with generated SQLs and their computed rewards
        """
        # Convert paths data and relevant paths to strings for the prompt
        paths_str = json.dumps(paths_data, indent=2)
        relevant_paths_str = json.dumps(relevant_paths, indent=2)
        
        prompt = f"""
        Generate {limit} different SQL statements for the following subquery using the given relevant paths.
        
        AVAILABLE PATHS:
        {paths_str}
        
        RELEVANT PATHS FOR THIS SUBQUERY:
        {relevant_paths_str}
        
        SUBQUERY: {subquery}
        
        Generate {limit} different, valid SQL statements that correctly answer this subquery using the provided paths.
        Each SQL should be complete and executable. Ensure variety in the solutions.
        
        Format your response as:
        <SQL_1>
        Your first SQL code here
        </SQL_1>
        
        <SQL_2>
        Your second SQL code here
        </SQL_2>
        
        ... and so on for all {limit} SQL statements
        """
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert SQL generator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,  # Slightly higher temperature for more variety
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated SQLs from the response
            content = response_data["choices"][0]["message"]["content"]
            
            sqls = []
            
            # Extract all SQL statements
            for i in range(1, limit + 1):
                # Extract SQL from the tags
                sql_match = re.search(rf'<SQL_{i}>(.*?)</SQL_{i}>', content, re.DOTALL)
                if sql_match:
                    sql = sql_match.group(1).strip()
                    sqls.append(sql)
            
            # If fewer than limit SQLs were generated, add placeholders
            if len(sqls) < limit:
                for i in range(len(sqls), limit):
                    placeholder_sql = f"-- Placeholder SQL #{i+1} for: {subquery}\nSELECT * FROM table WHERE condition_{i+1};"
                    sqls.append(placeholder_sql)
            
            # Now evaluate each SQL statement
            rewards = self._evaluate_sql_statements(sqls, subquery, paths_data, relevant_paths)
            
            return {
                "sqls": sqls,
                "rewards": rewards
            }
                
        except Exception as e:
            print(f"Error generating SQLs: {e}")
            # Return placeholders on error
            placeholder_sqls = []
            placeholder_rewards = {}
            
            for i in range(limit):
                sql = f"-- Error generating SQL #{i+1} for: {subquery}\nSELECT * FROM error_table WHERE error_condition_{i+1};"
                placeholder_sqls.append(sql)
                
                placeholder_rewards[sql] = {
                    "exec_correctness": 0.0,  # Cannot execute
                    "schema_linking": 0.0,
                    "dialect_function": 0.0,
                    "query_planning": 0.0,
                    "efficiency": 0.0,
                    "divergence": 0.0
                }
                
            return {
                "sqls": placeholder_sqls,
                "rewards": placeholder_rewards
            }
    
    def _evaluate_sql_statements(self, sql_statements, subquery, paths_data, relevant_paths):
        """
        Evaluate SQL statements based on execution and other metrics.
        
        Args:
            sql_statements: List of SQL statements to evaluate
            subquery: The original subquery
            paths_data: JSON data containing available paths
            relevant_paths: List of relevant paths for this subquery
            
        Returns:
            Dictionary mapping each SQL statement to its reward metrics
        """
        rewards = {}
        
        for sql in sql_statements:
            # Initialize rewards structure
            sql_rewards = {
                "exec_correctness": 0.0,
                "schema_linking": 0.0,
                "dialect_function": 0.0,
                "query_planning": 0.0,
                "efficiency": 0.0,
                "divergence": 0.0
            }
            
            # Test 1: Check if the SQL is valid syntax
            try:
                # Try to parse the SQL using sqlparse
                parsed = sqlparse.parse(sql)
                if parsed:
                    # If it parses successfully, it has at least valid syntax
                    sql_rewards["exec_correctness"] = 0.5  # Base score for valid syntax
                else:
                    sql_rewards["exec_correctness"] = 0.0
            except Exception:
                sql_rewards["exec_correctness"] = 0.0
            
            # Test 2: Check for correct schema linking
            sql_rewards["schema_linking"] = self._evaluate_schema_linking(sql, paths_data, relevant_paths)
            
            # Test 3: Check for dialect functions
            sql_rewards["dialect_function"] = self._evaluate_dialect_functions(sql)
            
            # Test 4: Evaluate query planning
            sql_rewards["query_planning"] = self._evaluate_query_planning(sql)
            
            # Test 5: Evaluate efficiency
            sql_rewards["efficiency"] = self._evaluate_efficiency(sql)
            
            # Test 6: Evaluate divergence (compare to other generated SQLs)
            sql_rewards["divergence"] = self._evaluate_divergence(sql, sql_statements)
            
            # Store the rewards for this SQL
            rewards[sql] = sql_rewards
        
        return rewards
    
    def _evaluate_schema_linking(self, sql, paths_data, relevant_paths):
        """Evaluate how well the SQL uses proper schema elements from paths"""
        # Extract tables and columns from SQL
        tables_pattern = r'FROM\s+([a-zA-Z0-9_]+)'
        columns_pattern = r'SELECT\s+(.*?)\s+FROM'
        
        tables_match = re.search(tables_pattern, sql, re.IGNORECASE)
        columns_match = re.search(columns_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        # If no tables or columns found, give a low score
        if not tables_match or not columns_match:
            return 0.1
        
        # Check if tables mentioned in SQL appear in the relevant paths
        tables = tables_match.group(1).split(',')
        tables = [t.strip() for t in tables]
        
        # Check if tables are in the relevant paths
        table_match_score = 0.0
        for table in tables:
            if any(table in path for path in relevant_paths):
                table_match_score += 1.0
        
        if tables:
            table_match_score /= len(tables)
        
        # Extract columns from the SELECT clause
        columns_text = columns_match.group(1)
        # Handle wildcard case
        if '*' in columns_text:
            column_match_score = 0.5  # Wildcard gets a medium score
        else:
            columns = columns_text.split(',')
            columns = [c.strip() for c in columns]
            
            # Check if columns are likely to be correct based on paths
            column_match_score = 0.0
            for column in columns:
                # Remove table prefix if present
                if '.' in column:
                    column = column.split('.')[1]
                
                # Check if column appears in paths
                if any(column in path for path in relevant_paths):
                    column_match_score += 1.0
            
            if columns:
                column_match_score /= len(columns)
        
        # Evaluate join conditions
        join_score = 0.0
        if 'JOIN' in sql.upper():
            # Simple heuristic: joins exist and seem to connect relevant tables
            join_score = 0.8
            
            # Check for ON conditions which suggest proper joins
            if 'ON' in sql.upper() and '=' in sql:
                join_score = 1.0
        
        # Combine scores, weighted toward tables and joins
        final_score = (table_match_score * 0.4) + (column_match_score * 0.3) + (join_score * 0.3)
        return final_score
    
    def _evaluate_dialect_functions(self, sql):
        """Evaluate correct usage of dialect-specific functions"""
        sql_upper = sql.upper()
        
        # List of common SQL dialect functions
        dialect_functions = [
            'ST_DISTANCE', 'ST_CONTAINS', 'ST_WITHIN',  # Spatial
            'DATE_TRUNC', 'TO_TIMESTAMP', 'EXTRACT',    # Date/time
            'REGEXP_MATCHES', 'REGEXP_REPLACE',         # Regex
            'ARRAY_AGG', 'UNNEST', 'JSON_EXTRACT',      # Arrays/JSON
            'PERCENTILE_CONT', 'MEDIAN', 'STDDEV',      # Statistical
            'RANK', 'DENSE_RANK', 'ROW_NUMBER'          # Window functions
        ]
        
        function_count = 0
        for func in dialect_functions:
            if func in sql_upper:
                function_count += 1
        
        # Check for proper syntax around functions
        syntax_pattern = r'(\w+)\s*\([\w\s,*]+\)'
        syntax_matches = re.findall(syntax_pattern, sql)
        syntax_score = min(1.0, len(syntax_matches) * 0.2)  # Scale based on function count
        
        # If no functions are used, assign a neutral score
        if function_count == 0:
            return 0.5
        
        # Score based on proper function usage and syntax
        return min(1.0, 0.3 + (function_count * 0.15) + (syntax_score * 0.55))
    
    def _evaluate_query_planning(self, sql):
        """Evaluate multi-step or nested SQL structures"""
        sql_upper = sql.upper()
        
        # Check for advanced SQL structures
        has_cte = 'WITH' in sql_upper
        has_subquery = '(' in sql and 'SELECT' in sql_upper[sql_upper.find('('):] if '(' in sql_upper else False
        has_union = 'UNION' in sql_upper
        has_group_by = 'GROUP BY' in sql_upper
        has_having = 'HAVING' in sql_upper
        has_order_by = 'ORDER BY' in sql_upper
        has_limit = 'LIMIT' in sql_upper
        
        # Count the structural elements
        structure_count = sum([has_cte, has_subquery, has_union, has_group_by, has_having, has_order_by, has_limit])
        
        # Check for nested subqueries (more complex planning)
        nesting_level = 0
        inside_subquery = False
        for char in sql:
            if char == '(':
                if inside_subquery:
                    nesting_level += 1
                elif 'SELECT' in sql_upper[sql.find('(') - 10:sql.find('(')]:
                    inside_subquery = True
            elif char == ')' and inside_subquery:
                if nesting_level > 0:
                    nesting_level -= 1
                else:
                    inside_subquery = False
        
        # More complex nesting gets higher scores
        nesting_score = min(1.0, nesting_level * 0.2)
        
        # Simple scoring based on complexity
        base_score = 0.3
        complexity_score = min(0.7, structure_count * 0.1)
        
        return base_score + complexity_score + nesting_score * 0.3
    
    def _evaluate_efficiency(self, sql):
        """Evaluate SQL for efficiency based on static analysis"""
        sql_upper = sql.upper()
        
        # Check for common inefficient patterns
        has_select_star = 'SELECT *' in sql_upper
        has_cartesian_join = 'FROM' in sql_upper and 'JOIN' in sql_upper and 'ON' not in sql_upper
        has_distinct = 'DISTINCT' in sql_upper
        has_or_conditions = ' OR ' in sql_upper
        has_not_in = 'NOT IN' in sql_upper
        has_subquery_in_where = 'WHERE' in sql_upper and '(' in sql_upper and 'SELECT' in sql_upper[sql_upper.find('WHERE'):] 
        
        # Count potential inefficiencies
        inefficiencies = sum([has_select_star, has_cartesian_join, has_distinct, has_or_conditions, has_not_in, has_subquery_in_where])
        
        # Check for efficiency optimizations
        has_index_hint = 'INDEX' in sql_upper
        has_limit = 'LIMIT' in sql_upper
        has_specific_columns = not has_select_star
        has_proper_joins = 'JOIN' in sql_upper and 'ON' in sql_upper
        
        # Count optimizations
        optimizations = sum([has_index_hint, has_limit, has_specific_columns, has_proper_joins])
        
        # Calculate efficiency score
        base_score = 0.5
        inefficiency_penalty = min(0.5, inefficiencies * 0.1)
        optimization_bonus = min(0.5, optimizations * 0.125)
        
        return max(0.0, min(1.0, base_score - inefficiency_penalty + optimization_bonus))
    
    def _evaluate_divergence(self, sql, all_sqls):
        """Evaluate how much this SQL diverges from other solutions"""
        similarity_scores = []
        
        # Compare this SQL to all others
        for other_sql in all_sqls:
            if sql == other_sql:
                continue  # Skip comparing to itself
                
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, sql, other_sql).ratio()
            similarity_scores.append(similarity)
        
        # If no other SQLs to compare to, give a medium score
        if not similarity_scores:
            return 0.5
        
        # Average similarity to other SQL statements
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Divergence is the inverse of similarity (high similarity = low divergence)
        # But we want a little divergence (to encourage exploration), so we'll score
        # highest for moderate divergence
        if avg_similarity < 0.3:
            # Too different, possibly incorrect - moderate score
            return 0.5
        elif avg_similarity > 0.8:
            # Too similar to other solutions - low score
            return 0.3
        else:
            # Good balance - high score
            return 0.9

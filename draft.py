#!/usr/bin/env python3

import json
import re
import requests
import sqlparse
import difflib
import os
import sys
import argparse

class QueryProcessor:
    """Main class for query processing pipeline with decomposition, path finding, and SQL generation."""
    
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
        self.path_finder = PathFinder(api_url, api_key, model)
        self.sql_generator = SQLGenerator(api_url, api_key, model)
    
    
    def process_query(self, paths_json_file, question):
        """
        Process a natural language query through the pipeline:
        1. Decompose into subqueries
        2. Find top 10 relevant paths for each subquery with relevance scores
        3. Generate top 10 SQL for each subquery based on paths with rewards
        
        Args:
            paths_json_file: Path to JSON file containing available paths
            question: Natural language question
            
        Returns:
            Dictionary with detailed processing results
        """
        # Load paths data
        paths_data = self._load_paths_json(paths_json_file)
        
        # Step 1: Decompose the query into subqueries
        subqueries = self.query_decomposer.decompose_query(question)
        
        # Step 2 & 3: Process each subquery to find paths and generate SQL
        subquery_results = []
        
        for i, subquery in enumerate(subqueries):
            # Find top 10 relevant paths for this subquery with relevance scores
            path_results = self.path_finder.find_top_relevant_paths(paths_data, subquery, limit=10)
            
            # Generate top 10 SQL for this subquery using the found paths
            sql_results = self.sql_generator.generate_top_subsqls(paths_data, subquery, path_results["paths"], limit=10)
            
            # Combine results for this subquery
            subquery_results.append({
                "subquery_id": f"Subquery {chr(65 + i)}",  # A, B, C, etc.
                "subquery_text": subquery,
                "top_paths": path_results["paths"],
                "path_relevance_scores": path_results["relevance_scores"],
                "top_sqls": sql_results["sqls"],
                "sql_rewards": sql_results["rewards"]
            })
        
        # Return the processing results
        return {
            "original_query": question,
            "subqueries": subqueries,
            "subquery_results": subquery_results
        }
    
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
    
    def decompose_query(self, question):
        """
        Decompose a complex query into simpler subqueries.
        
        Args:
            question: The original natural language question
            
        Returns:
            List of subqueries
        """
        prompt = f"""
        Please decompose the following complex query into simpler subqueries:
        
        QUERY: {question}
        
        Break this down into 2-4 distinct subqueries, where each subquery addresses a specific part of the overall question.
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
            print(response_data)
            # Extract subqueries from the tags
            subqueries_match = re.search(r'<SUBQUERIES>(.*?)</SUBQUERIES>', content, re.DOTALL)
            
            if subqueries_match:
                subqueries_text = subqueries_match.group(1).strip()
                # Extract numbered items, removing the numbers
                subquery_list = re.findall(r'^\d+\.\s*(.*?)$', subqueries_text, re.MULTILINE)
                
                if subquery_list:
                    return subquery_list
            
            # Fallback: if parsing fails, return the original query as a single subquery
            return [question]
                
        except Exception as e:
            print(f"Error decomposing query: {e}")
            # Fallback to a basic decomposition
            return [question]


class PathFinder:
    """Component for finding relevant paths for a subquery."""
    
    def __init__(self, api_url, api_key, model):
        """Initialize the path finder."""
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
    
    def find_top_relevant_paths(self, paths_data, subquery, limit=10):
        """
        Find top relevant paths for a given subquery and calculate relevance scores.
        
        Args:
            paths_data: JSON data containing available paths
            subquery: The subquery text
            limit: Maximum number of paths to return (default: 10)
            
        Returns:
            Dictionary with paths and their relevance scores
        """
        # Convert paths data to a string for the prompt
        paths_str = json.dumps(paths_data, indent=2)
        
        prompt = f"""
        Given the following paths and a subquery, identify the {limit} most relevant paths that would help answer the subquery.
        
        AVAILABLE PATHS:
        {paths_str}
        
        SUBQUERY: {subquery}
        
        For each relevant path, assign a relevance score between 0.0 and 1.0, where 1.0 means the path is perfectly relevant.
        
        Format your response as:
        <RELEVANT_PATHS>
        path1: score1
        path2: score2
        ...
        path{limit}: score{limit}
        </RELEVANT_PATHS>
        
        Return exactly {limit} paths, sorted by relevance score in descending order.
        """
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert at finding relevant paths for queries."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        try:
            import pdb; pdb.set_trace()
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated paths from the response
            content = response_data["choices"][0]["message"]["content"]
            
            # Extract paths from the tags
            paths_match = re.search(r'<RELEVANT_PATHS>(.*?)</RELEVANT_PATHS>', content, re.DOTALL)
            
            paths = []
            relevance_scores = {}
            
            if paths_match:
                paths_text = paths_match.group(1).strip()
                # Extract path and score pairs
                path_score_pairs = re.findall(r'^(.*?):\s*([\d.]+)$', paths_text, re.MULTILINE)
                
                for path, score in path_score_pairs:
                    path = path.strip()
                    paths.append(path)
                    relevance_scores[path] = float(score)
            
            # If no paths were found or fewer than limit, generate placeholders
            if len(paths) < limit:
                # Generate placeholder paths
                for i in range(len(paths), limit):
                    placeholder_path = f"path_{i+1}"
                    paths.append(placeholder_path)
                    relevance_scores[placeholder_path] = max(0.0, 1.0 - (i * 0.1))
            
            # Ensure we only return the top 'limit' paths
            if len(paths) > limit:
                # Sort paths by score
                paths_with_scores = [(path, relevance_scores[path]) for path in paths]
                paths_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take only the top 'limit' paths
                top_paths = [p for p, _ in paths_with_scores[:limit]]
                top_relevance_scores = {p: relevance_scores[p] for p in top_paths}
                
                paths = top_paths
                relevance_scores = top_relevance_scores
            
            return {
                "paths": paths,
                "relevance_scores": relevance_scores
            }
                
        except Exception as e:
            print(f"Error finding relevant paths: {e}")
            # Return placeholders on error
            placeholders = {}
            placeholder_paths = []
            
            for i in range(limit):
                path = f"error_path_{i+1}"
                placeholder_paths.append(path)
                placeholders[path] = max(0.1, 1.0 - (i * 0.1))
            
            return {
                "paths": placeholder_paths,
                "relevance_scores": placeholders
            }


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
        
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process queries using Path Analyzer')
    parser.add_argument('--config', type=str, help='Path to config file with API credentials')
    parser.add_argument('--paths', type=str, required=True, help='Path to JSON file containing paths')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--output', type=str, help='Path to save results as JSON')
    return parser.parse_args()


def main():
    """Main function to run the Path Analyzer with configuration from a JSON file."""
    # Load configuration from JSON file
    config_file = "config.json"    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Check if paths file exists
    paths_json_file = config.get("paths_file")

    
    # Get API key from config or environment
    api_key = config.get("api_key") or os.environ.get("API_KEY", "")

    
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

# python PathAnalyzer.py  --paths database_schema_paths.json
# Replace this problematic section:
paths_match = re.search(r'<RELEVANT_PATHS>(.*?)</RELEVANT_PATHS>', content, re.DOTALL)

paths = []
relevance_scores = {}

if paths_match:
    paths_text = paths_match.group(1).strip()
    # Extract path and score pairs
    path_score_pairs = re.findall(r'^(.*?):\s*([\d.]+)$', paths_text, re.MULTILINE)
    
    for path, score in path_score_pairs:
        path = path.strip()
        paths.append(path)
        relevance_scores[path] = float(score)

# With this more robust version:
paths = []
relevance_scores = {}

# Try multiple pattern formats to extract paths
# First try original tag format
paths_match = re.search(r'<RELEVANT_PATHS>(.*?)</RELEVANT_PATHS>', content, re.DOTALL)
if paths_match:
    paths_text = paths_match.group(1).strip()
    path_score_pairs = re.findall(r'^(.*?):\s*([\d.]+)$', paths_text, re.MULTILINE)
    
    for path, score in path_score_pairs:
        path = path.strip()
        paths.append(path)
        relevance_scores[path] = float(score)
else:
    # Try numbered list format: 1. path: score
    path_score_pairs = re.findall(r'(?:\d+\.|\-)\s*(.*?):\s*([\d.]+)', content)
    
    if path_score_pairs:
        for path, score in path_score_pairs:
            path = path.strip()
            paths.append(path)
            relevance_scores[path] = float(score)
    else:
        # Last resort: Extract any paths mentioned in the response
        for path in paths_data:
            if path in content:
                paths.append(path)
                # Assign decreasing scores by position
                relevance_scores[path] = max(0.1, 1.0 - 0.1 * len(paths))

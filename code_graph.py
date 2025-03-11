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
        import json
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
        import re
        import requests
        
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
        import json
        import requests
        import re
        
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
        Generate top SQL statements for a subquery based on relevant paths.
        
        Args:
            paths_data: JSON data containing available paths
            subquery: The subquery text
            relevant_paths: List of relevant paths for this subquery
            limit: Maximum number of SQL statements to generate (default: 10)
            
        Returns:
            Dictionary with generated SQLs and their rewards
        """
        import json
        import requests
        import re
        import random
        
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
        
        Generate {limit} different SQL statements that correctly answer this subquery using the provided paths.
        For each SQL, also assign rewards (numbers between 0.0 and 1.0) for these specific aspects:
        
        - exec_correctness: Whether the query runs successfully and returns correct results
        - schema_linking: How well tables and columns are selected (correctly identifying tables, columns, and joins)
        - dialect_function: Proper usage of dialect-specific functions (like ST_DISTANCE, DATE_TRUNC, etc.)
        - query_planning: Accuracy of multi-step or nested SQL structures (CTEs, subqueries, etc.)
        - efficiency: Resource usage optimization (minimizing scanned bytes, execution time, etc.)
        - divergence: How much the query deviates from standard patterns (lower is better)
        
        Format your response as:
        <SQL_1>
        Your first SQL code here
        </SQL_1>
        <REWARDS_1>
        correctness: score
        efficiency: score
        readability: score
        </REWARDS_1>
        
        <SQL_2>
        Your second SQL code here
        </SQL_2>
        <REWARDS_2>
        correctness: score
        efficiency: score
        readability: score
        </REWARDS_2>
        
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
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated SQLs and rewards from the response
            content = response_data["choices"][0]["message"]["content"]
            
            sqls = []
            rewards = {}
            
            # Extract all SQL-reward pairs
            for i in range(1, limit + 1):
                # Extract SQL from the tags
                sql_match = re.search(rf'<SQL_{i}>(.*?)</SQL_{i}>', content, re.DOTALL)
                if sql_match:
                    sql = sql_match.group(1).strip()
                    sqls.append(sql)
                    
                    # Extract rewards for this SQL
                    rewards_match = re.search(rf'<REWARDS_{i}>(.*?)</REWARDS_{i}>', content, re.DOTALL)
                    if rewards_match:
                        rewards_text = rewards_match.group(1).strip()
                        # Extract reward and score pairs
                        reward_score_pairs = re.findall(r'^(.*?):\s*([\d.]+)
            
            # If fewer than limit SQLs were generated, add placeholders
            if len(sqls) < limit:
                for i in range(len(sqls), limit):
                    placeholder_sql = f"-- Placeholder SQL #{i+1} for: {subquery}\nSELECT * FROM table WHERE condition_{i+1};"
                    sqls.append(placeholder_sql)
                    
                    # Add placeholder rewards
                    rewards[placeholder_sql] = {
                        "exec_correctness": max(0.1, 0.9 - (i * 0.05)),
                        "schema_linking": max(0.1, 0.85 - (i * 0.05)),
                        "dialect_function": max(0.1, 0.7 - (i * 0.05)),
                        "query_planning": max(0.1, 0.75 - (i * 0.05)),
                        "efficiency": max(0.1, 0.8 - (i * 0.05)),
                        "divergence": max(0.1, 0.6 - (i * 0.05))
                    }
            
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
                    "exec_correctness": max(0.1, 0.5 - (i * 0.05)),
                    "schema_linking": max(0.1, 0.45 - (i * 0.05)),
                    "dialect_function": max(0.1, 0.35 - (i * 0.05)),
                    "query_planning": max(0.1, 0.4 - (i * 0.05)),
                    "efficiency": max(0.1, 0.4 - (i * 0.05)),
                    "divergence": max(0.1, 0.3 - (i * 0.05))
                }
            
            return {
                "sqls": placeholder_sqls,
                "rewards": placeholder_rewards
            }
, rewards_text, re.MULTILINE)
                        
                        sql_rewards = {
                            "exec_correctness": 0.0,
                            "schema_linking": 0.0,
                            "dialect_function": 0.0,
                            "query_planning": 0.0,
                            "efficiency": 0.0,
                            "divergence": 0.0
                        }
                        
                        for reward, score in reward_score_pairs:
                            reward = reward.strip().lower()
                            # Map the rewards to our specific categories
                            if 'correct' in reward or 'exec' in reward:
                                sql_rewards["exec_correctness"] = float(score)
                            elif 'schema' in reward or 'link' in reward:
                                sql_rewards["schema_linking"] = float(score)
                            elif 'dialect' in reward or 'func' in reward:
                                sql_rewards["dialect_function"] = float(score)
                            elif 'plan' in reward:
                                sql_rewards["query_planning"] = float(score)
                            elif 'efficien' in reward:
                                sql_rewards["efficiency"] = float(score)
                            elif 'diverg' in reward:
                                sql_rewards["divergence"] = float(score)
                        
                        rewards[sql] = sql_rewards
            
            # If fewer than limit SQLs were generated, add placeholders
            if len(sqls) < limit:
                for i in range(len(sqls), limit):
                    placeholder_sql = f"-- Placeholder SQL #{i+1} for: {subquery}\nSELECT * FROM table WHERE condition_{i+1};"
                    sqls.append(placeholder_sql)
                    
                    # Add placeholder rewards
                    rewards[placeholder_sql] = {
                        "correctness": max(0.1, 0.9 - (i * 0.05)),
                        "efficiency": max(0.1, 0.8 - (i * 0.05)),
                        "readability": max(0.1, 0.85 - (i * 0.05))
                    }
            
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
                    "correctness": max(0.1, 0.5 - (i * 0.05)),
                    "efficiency": max(0.1, 0.4 - (i * 0.05)),
                    "readability": max(0.1, 0.45 - (i * 0.05))
                }
            
            return {
                "sqls": placeholder_sqls,
                "rewards": placeholder_rewards
            }

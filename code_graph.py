class SQLQueryAnalyzer:
    """Main class for SQL query analysis with subqueries and path analysis."""
    
    def __init__(self, api_url, api_key, model):
        """
        Initialize the SQL query analyzer.
        
        Args:
            api_url: URL for the LLM API
            api_key: API key for authentication
            model: Model name to use
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        
        # Initialize components
        self.query_segmenter = LLMQuerySegmenter(api_url, api_key, model)
        self.sql_generator = LLMSQLGenerator(api_url, api_key, model)
        self.path_analyzer = None
    
    def analyze_query(self, database_schema, question, hint):
        """
        Analyze a natural language query, generate subqueries, and perform path analysis.
        
        Args:
            database_schema: Database schema in text format
            question: Natural language question
            hint: Additional hints about the question
            
        Returns:
            Dictionary with detailed analysis results
        """
        # Initialize the path analyzer with the database schema
        self.path_analyzer = SQLPathAnalyzer(database_schema)
        
        # Step 1: Segment the original query into subqueries
        subqueries = self.query_segmenter.segment_query(question)
        dependencies = self.query_segmenter.identify_dependencies(subqueries)
        
        # Step 2: Generate SQL for each subquery
        subquery_results = []
        for i, subquery in enumerate(subqueries):
            # Generate SQL and path for this subquery
            result = self.sql_generator.generate_sql_for_subquery(database_schema, subquery)
            
            # Analyze the path for this subquery
            path_analysis = self.path_analyzer.evaluate_path(subquery, result.get("path", []))
            
            subquery_results.append({
                "subquery_index": i,
                "subquery_text": subquery,
                "sql": result.get("sql", ""),
                "path": result.get("path", []),
                "path_str": path_analysis.get("path_str", ""),
                "path_scores": path_analysis.get("scores", {}),
                "dependencies": dependencies.get(i, [])
            })
        
        # Step 3: Generate the final, complete SQL query
        final_sql = self.generate_full_sql(database_schema, question, hint)
        
        # Step 4: Calculate overall scores
        overall_scores = self._calculate_overall_scores(subquery_results)
        
        # Return the complete analysis results
        return {
            "original_query": question,
            "hint": hint,
            "subqueries": subqueries,
            "subquery_results": subquery_results,
            "final_sql": final_sql,
            "overall_scores": overall_scores
        }
    
    def _calculate_overall_scores(self, subquery_results):
        """Calculate overall scores across all subquery paths."""
        if not subquery_results:
            return {
                "schema_linking": 0.0,
                "join_quality": 0.0,
                "dialect_function": 0.0,
                "data_calculation": 0.0,
                "query_planning": 0.0,
                "relevance": 0.0,
                "overall": 0.0
            }
        
        score_categories = ["schema_linking", "join_quality", "dialect_function", 
                           "data_calculation", "query_planning", "relevance", "overall"]
        
        # Sum up scores from all subqueries
        total_scores = {category: 0.0 for category in score_categories}
        for result in subquery_results:
            scores = result.get("path_scores", {})
            for category in score_categories:
                total_scores[category] += scores.get(category, 0.0)
        
        # Calculate average scores
        num_subqueries = len(subquery_results)
        average_scores = {
            category: round(total_scores[category] / num_subqueries, 2) 
            for category in score_categories
        }
        
        return average_scores
    
    def generate_full_sql(self, database_schema, question, hint):
        """
        Generate the final SQL query using the divide and conquer approach.
        
        Args:
            database_schema: Database schema in text format
            question: Natural language question
            hint: Additional hints about the question
            
        Returns:
            Generated SQL query
        """
        prompt = self._create_prompt(database_schema, question, hint)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a database expert who specializes in SQL query generation."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Lower temperature for more deterministic outputs
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated SQL from the response
            sql_response = response_data["choices"][0]["message"]["content"]
            
            # Extract SQL query from <FINAL_ANSWER> tags
            sql_match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', sql_response, re.DOTALL)
            
            if sql_match:
                return sql_match.group(1).strip()
            else:
                # If no tags are found, return the full response
                return sql_response
                
        except Exception as e:
            print(f"Error generating full SQL: {e}")
            return None
    
    
    
    def _create_prompt(self, database_schema, question, hint):
        """Create the prompt for generating the full SQL query."""
        file_path = "sql_subquery_prompt.txt"
        with open(file_path, "r", encoding="utf-8") as file:
            prompt = file.read()
    
        return prompt.format(
            DATABASE_SCHEMA=database_schema,
            QUESTION=question,
            HINT=hint
        )

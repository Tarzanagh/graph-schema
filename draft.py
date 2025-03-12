"""
llm_service.py - Module for interacting with LLM APIs
"""
import requests
import re
import json
import os
from typing import List, Dict, Any, Optional

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
        self.api_key = api_key or os.getenv('LLM_API_KEY')
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
            
            # Flexible payload construction
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add model if specified
            if self.model:
                payload["model"] = self.model
            
            # Make API call
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            # Check response
            if response.status_code == 200:
                response_json = response.json()
                
                # Flexible response parsing
                if isinstance(response_json, dict):
                    return response_json.get("output", 
                           response_json.get("text", 
                           response_json.get("response", "")))
                elif isinstance(response_json, str):
                    return response_json
                
                return str(response_json)
            
            # Raise error for non-200 status
            response.raise_for_status()
            
        except requests.RequestException as e:
            print(f"Error calling LLM API: {e}")
            raise RuntimeError(f"API call failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error in LLM API call: {e}")
            raise RuntimeError(f"Unexpected error: {str(e)}")
    
    def decompose_query(self, 
                        query: str, 
                        schema_details: List[Dict[str, Any]]) -> List[str]:
        """
        Decompose a complex query into subqueries using an LLM.
        
        Args:
            query (str): The original natural language question
            schema_details (List[Dict]): Database schema information
            
        Returns:
            List[str]: List of subqueries
        """
        # Import format_schema_for_prompt here to avoid circular import
        # Note: You'll need to implement this function or import from the correct module
        def format_schema_for_prompt(schema_details):
            """Placeholder for schema formatting function."""
            schema_text = "DATABASE SCHEMA:\n"
            for table in schema_details:
                schema_text += f"Table: {table.get('table_name', 'Unknown')}\n"
            return schema_text
        
        # Create prompt for LLM
        schema_text = format_schema_for_prompt(schema_details)
        
        prompt = f"""
        Decompose the following database query into 2-4 distinct subqueries:
        
        DATABASE SCHEMA:
        {schema_text}
        
        ORIGINAL QUERY: {query}
        
        Provide subqueries formatted as:
        <SUBQUERIES>
        1. First subquery
        2. Second subquery
        ...
        </SUBQUERIES>
        """
        
        # Call LLM
        try:
            llm_response = self.call_llm(prompt, max_tokens=500, temperature=0.3)
            
            # Parse subqueries
            subqueries_match = re.search(r'<SUBQUERIES>(.*?)</SUBQUERIES>', llm_response, re.DOTALL)
            
            if subqueries_match:
                subqueries_text = subqueries_match.group(1).strip()
                subquery_list = re.findall(r'^\d+\.\s*(.*?)$', subqueries_text, re.MULTILINE)
                
                return subquery_list or [query]
            
            # Fallback to original query
            return [query]
        
        except Exception as e:
            print(f"Error in query decomposition: {e}")
            return [query]
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading configuration: {e}")
            return {}

def main():
    """
    Example usage of LLMService.
    """
    try:
        # Load configuration
        config_path = "config.json"
        config = LLMService.load_config(config_path)
        
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
        
        # Example LLM call
        response = llm_service.call_llm("Write a two-sentence explanation of machine learning")
        print(response)
    
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()

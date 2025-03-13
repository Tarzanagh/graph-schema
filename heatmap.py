def create_simple_heatmap(graph, node_importance):
    """
    Create a simple heatmap visualization of node importance using imshow.
    
    Args:
        graph: The graph structure
        node_importance: Dictionary mapping node_id to importance score
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get all tables from the graph
    tables = [node for node in graph.nodes() if graph.nodes[node].get('type') == 'table']
    tables.sort()  # Sort for consistent ordering
    
    # Get all columns from the graph
    columns = [node for node in graph.nodes() if graph.nodes[node].get('type') == 'column']
    # Sort columns by their table
    columns.sort(key=lambda col: (graph.nodes[col].get('table', ''), col))
    
    # Create a matrix for the heatmap
    num_tables = len(tables)
    num_columns = len(columns)
    
    # Initialize the matrix with zeros
    matrix = np.zeros((num_tables, num_columns))
    
    # Fill the matrix with importance scores
    for i, table in enumerate(tables):
        for j, column in enumerate(columns):
            # Check if this column belongs to this table
            if graph.nodes[column].get('table', '') == table:
                # Use a higher base value to show table-column relationship
                base_value = 0.3
            else:
                base_value = 0
            
            # Add importance scores
            table_importance = node_importance.get(table, 0)
            column_importance = node_importance.get(column, 0)
            
            # Combine importance scores (weighted sum)
            matrix[i, j] = base_value + 0.7 * column_importance + 0.3 * table_importance
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Create the heatmap
    im = plt.imshow(matrix, cmap='Blues', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, label='Importance Score')
    
    # Set labels
    plt.title('Table-Column Importance Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Tables')
    
    # Add table names as y-tick labels
    plt.yticks(range(num_tables), tables, fontsize=8)
    
    # For columns, we might need to rotate them for readability
    plt.xticks(range(num_columns), [col.split('.')[-1] for col in columns], 
              rotation=90, fontsize=6)
    
    plt.tight_layout()
    return plt

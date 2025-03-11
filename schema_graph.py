def create_schema_graph(db_file):
    """
    Extract a schema graph from the database with the following relationships:
    1. Same Table Match: Two columns belonging to the same table are linked
    2. Primary-Foreign Key (Column-Column): Link between foreign key and primary key columns
    3. Foreign Key (Column-Table): Link from foreign key column to referenced table
    4. Primary Key (Column-Table): Link from primary key column to its table
    5. Table-Column Match: Link between a table and its columns
    6. Primary-Foreign Key (Table-Table): Link between tables with foreign key relationships
    """
    
    # Create directed graph
    graph = nx.DiGraph()
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    # Store primary keys and foreign keys for later use
    primary_keys = {}  # table -> list of primary key columns
    foreign_keys = {}  # table -> list of (column, ref_table, ref_column)
    all_columns = {}   # table -> list of all columns
    
    # First pass: Add all tables and their columns as nodes
    for table in tables:
        # Add table node
        graph.add_node(table, type='table', node_class='table')
        
        # Get columns for this table
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        
        # Initialize collections for this table
        primary_keys[table] = []
        all_columns[table] = []
        
        # In SQLite, PRAGMA table_info returns columns with structure:
        # (cid, name, type, notnull, dflt_value, pk)
        for col in columns:
            col_name = col[1]
            col_id = f"{table}.{col_name}"
            is_pk = col[5] == 1
            
            # Add column node
            graph.add_node(col_id, 
                          type='column',
                          node_class='column', 
                          data_type=col[2],
                          is_primary_key=is_pk,
                          not_null=col[3]==1)
            
            # Store column
            all_columns[table].append(col_id)
            
            # Store primary key if applicable
            if is_pk:
                primary_keys[table].append(col_id)
            
            # Add table-column relationship (5. Table-Column Match)
            graph.add_edge(table, col_id, 
                          relationship_type='table_column',
                          relationship=f"Column in {table}")
    
    # Second pass: Extract foreign key information
    for table in tables:
        # Initialize foreign keys for this table
        foreign_keys[table] = []
        
        # Extract foreign key relationships
        cursor.execute(f"PRAGMA foreign_key_list({table});")
        fk_list = cursor.fetchall()
        
        for fk in fk_list:
            fk_col = fk[3]          # Column in current table
            ref_table = fk[2]       # Referenced table
            ref_col = fk[4]         # Referenced column
            
            # Store foreign key
            foreign_keys[table].append((f"{table}.{fk_col}", ref_table, f"{ref_table}.{ref_col}"))
    
    # Third pass: Add all the required relationships
    for table in tables:
        # 1. Same Table Match: Connect columns in the same table
        for i, col1 in enumerate(all_columns[table]):
            for col2 in all_columns[table][i+1:]:
                graph.add_edge(col1, col2, 
                              relationship_type='same_table',
                              relationship=f"Columns in same table ({table})")
                graph.add_edge(col2, col1, 
                              relationship_type='same_table',
                              relationship=f"Columns in same table ({table})")
        
        # Handle foreign keys
        for fk_col, ref_table, ref_col in foreign_keys[table]:
            # 2. Primary-Foreign Key (Column-Column)
            graph.add_edge(fk_col, ref_col, 
                          relationship_type='pk_fk_column',
                          relationship=f"Foreign key reference")
            
            # 3. Foreign Key (Column-Table)
            graph.add_edge(fk_col, ref_table, 
                          relationship_type='fk_table',
                          relationship=f"Foreign key to table")
            
            # 4. Primary-Foreign Key (Table-Table)
            graph.add_edge(table, ref_table, 
                          relationship_type='pk_fk_table',
                          relationship=f"Table foreign key relationship")
    
    # 5. Primary Key (Column-Table): Add links from primary keys to their tables
    for table, pk_cols in primary_keys.items():
        for pk_col in pk_cols:
            graph.add_edge(pk_col, table, 
                          relationship_type='pk_table',
                          relationship=f"Primary key of table")
    
    conn.close()
    
    # Print some statistics
    print(f"Schema graph created with:")
    print(f" - {len([n for n, d in graph.nodes(data=True) if d.get('type') == 'table'])} table nodes")
    print(f" - {len([n for n, d in graph.nodes(data=True) if d.get('type') == 'column'])} column nodes")
    
    # Count edges by relationship type
    edge_counts = {}
    for _, _, data in graph.edges(data=True):
        rel_type = data.get('relationship_type', 'unknown')
        edge_counts[rel_type] = edge_counts.get(rel_type, 0) + 1
    
    for rel_type, count in edge_counts.items():
        print(f" - {count} {rel_type} edges")
    
    return graph

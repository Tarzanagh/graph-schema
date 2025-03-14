import networkx as nx

# Save as GEXF (Graph Exchange XML Format)
nx.write_gexf(enhanced_graph, "enhanced_graph.gexf")

# Alternative formats:
# GRAPHML
nx.write_graphml(enhanced_graph, "enhanced_graph.graphml")

# JSON
nx.write_json(enhanced_graph, "enhanced_graph.json")

# GEXF preserves all node and edge attributes and is compatible with visualization tools like Gephi

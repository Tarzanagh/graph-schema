import json
with open("enhanced_graph.json", "w") as f:
    json_graph = nx.node_link_data(enhanced_graph)
    json.dump(json_graph, f)

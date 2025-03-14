import json
json.dump(nx.node_link_data(enhanced_graph), open("enhanced_graph.json", "w"))

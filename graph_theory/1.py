import networkx as nx
from utils import verify_min_cost_flow
G = nx.DiGraph()

G.add_node("S")
G.add_node("T")

G.add_edge("S", "2", capacity = 5, weight = 3)
G.add_edge("S", "3", capacity = 3, weight = 6)
G.add_edge("2", "4", capacity = 2, weight = 8)
G.add_edge("3", "2", capacity = 1, weight = 2)
G.add_edge("3", "5", capacity = 4, weight = 2)
G.add_edge("4", "3", capacity = 1, weight = 1)
G.add_edge("4", "5", capacity = 3, weight = 4)
G.add_edge("4", "T", capacity = 2, weight = 10)
G.add_edge("5", "T", capacity = 5, weight = 2)


flow_dict = nx.max_flow_min_cost(
    G, 
    "S",
    "T",
    capacity = "capacity",
    weight = "weight",
)

total_cost = nx.cost_of_flow(
    G,
    flow_dict,
    weight = "weight",
)

max_flow_val = sum(flow_dict["S"].values())
G.nodes["S"]["demand"] = -max_flow_val 
G.nodes["T"]["demand"] = max_flow_val

print("最小总费用:", total_cost)

print("\n各边流量:")
for u, v, data in G.edges(data = True):
    flow = flow_dict[u][v]

    print(
        f"{u} -> {v}: "
        f"流量={flow}, "
        f"容量={data['capacity']}, "
        f"单位费用={data['weight']}, "
        f"边费用={flow * data['weight']}"
    )

#验证部分
verify_min_cost_flow(G, flow_dict)
total_cost, flow_dict = nx.network_simplex(G)
verify_min_cost_flow(G, flow_dict)
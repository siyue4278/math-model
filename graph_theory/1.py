import networkx as nx

G = nx.DiGraph()

G.add_node("S", demand = -5)
G.add_node("A", demand = 0)
G.add_node("B", demand = 0)
G.add_node("T", demand = 5)

G.add_edge("S", "A", capacity = 4, weight = 2)
G.add_edge("S", "B", capacity = 3, weight = 1)
G.add_edge("A", "B", capacity = 2, weight = 1)
G.add_edge("A", "T", capacity = 3, weight = 3)
G.add_edge("B", "T", capacity = 4, weight = 2)

flow_dict = nx.min_cost_flow(
    G, 
    demand = "demand",
    capacity = "capacity",
    weight = "weight",
)

total_cost = nx.cost_of_flow(
    G,
    flow_dict,
    weight = "weight",
)

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

def verify_min_cost_flow(
    G: nx.DiGraph,
    flow_dict: dict,
) -> None:

    total_demand = sum(
        G.nodes[node]["demand"]
        for node in G.nodes
    )

    print("总需求:", total_demand)

    assert total_demand == 0

    #容量平衡
    for u, v, data in G.edges(data = True):
        flow = flow_dict[u][v]
        capacity = data["capacity"]

        assert 0 <= flow <= capacity

    #节点平衡
    for node in G.nodes:
        inflow = sum(
            flow_dict[u][node]
            for u in G.predecessors(node)
        )

        outflow = sum(
        flow_dict[node][v]
            for v in G.successors(node)
        )
    #总供需平衡
        demand = G.nodes[node]["demand"]

   
        print(
            f"{node}: "
            f"流入={inflow}, "
            f"流出={outflow}, "
            f"流入-流出={inflow - outflow}, "
            f"demand={demand}"
        )

        assert inflow - outflow == demand 
    
total_cost, flow_dict = nx.network_simplex(G)

verify_min_cost_flow(G, flow_dict)


#灵敏度分析

import networkx as nx


def build_network(sb_capacity: int) -> nx.DiGraph:
    """根据 S->B 的容量建立最小费用流网络。"""

    G = nx.DiGraph()

    # NetworkX：供应为负，需求为正
    G.add_node("S", demand=-5)
    G.add_node("A", demand=0)
    G.add_node("B", demand=0)
    G.add_node("T", demand=5)

    G.add_edge("S", "A", capacity=4, weight=2)
    G.add_edge("S", "B", capacity=sb_capacity, weight=1)
    G.add_edge("A", "B", capacity=2, weight=1)
    G.add_edge("A", "T", capacity=3, weight=3)
    G.add_edge("B", "T", capacity=4, weight=2)

    return G


for sb_capacity in range(1, 7):
    G = build_network(sb_capacity)

    total_cost, flow_dict = nx.network_simplex(
        G,
        demand="demand",
        capacity="capacity",
        weight="weight",
    )

    print(f"\nS->B容量 = {sb_capacity}")
    print(f"最小总费用 = {total_cost}")

    for u, v in G.edges:
        flow = flow_dict[u][v]

        if flow > 0:
            print(
                f"  {u}->{v}: "
                f"流量={flow}, "
                f"单位费用={G[u][v]['weight']}"
            )

total_cost, flow_dict = nx.network_simplex(G)

verify_min_cost_flow(G, flow_dict)
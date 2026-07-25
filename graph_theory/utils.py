import networkx as nx

def verify_min_cost_flow(
    G: nx.DiGraph,
    flow_dict: dict,
    demand_key: str = "demand",
    capacity_key: str = "capacity",
    verbose: bool = True,
) -> bool:
    """验证最小费用流 (Min-Cost Flow) 解的合法性与守恒性"""
    
    # 1. 检查全局供需平衡
    total_demand = sum(G.nodes[node].get(demand_key, 0) for node in G.nodes)
    if verbose:
        print(f"================ [ 网络流可行性校验 ] ================")
        print(f"► 全局总需求和: {total_demand}")

    assert total_demand == 0, f"【校验失败】全局供需不平衡！Sum({demand_key}) = {total_demand} ≠ 0"

    # ----------------------------------------------------
    # 2. 检查边容量限制 (加上边容量与流量的显式打印)
    # ----------------------------------------------------
    if verbose:
        print("\n► 边容量限制校验:")

    for u, v, data in G.edges(data=True):
        flow = flow_dict.get(u, {}).get(v, 0)
        capacity = data.get(capacity_key, float("inf"))
        
        if verbose:
            cap_str = f"{capacity}" if capacity != float("inf") else "∞"
            print(f"  └─ 边 {u:<2} -> {v:<2}: 实际流量={flow:<3} | 容量上限={cap_str}")

        assert 0 <= flow <= capacity, (
            f"【校验失败】边 ({u} -> {v}) 违反容量约束！当前流量={flow}, 容量上限={capacity}"
        )

    # ----------------------------------------------------
    # 3. 检查节点流量守恒
    # ----------------------------------------------------
    if verbose:
        print("\n► 节点流量守恒状态:")

    for node in G.nodes:
        inflow = sum(flow_dict.get(u, {}).get(node, 0) for u in G.predecessors(node))
        outflow = sum(flow_dict.get(node, {}).get(v, 0) for v in G.successors(node))
        demand = G.nodes[node].get(demand_key, 0)
        net_flow = inflow - outflow

        if verbose:
            print(
                f"  └─ 节点 {node:<4}: "
                f"流入={inflow:<3} | "
                f"流出={outflow:<3} | "
                f"净流入(in-out)={net_flow:<3} | "
                f"设定demand={demand}"
            )

        assert net_flow == demand, (
            f"【校验失败】节点 '{node}' 流量不守恒！净流入={net_flow}, 但 demand={demand}"
        )

    if verbose:
        print("================ [ ✅ 校验全部通过 ] ================\n")

    return True
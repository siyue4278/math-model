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

import networkx as nx

def analyse_critical_path(G: nx.DiGraph, verbose: bool = True):
    """计算 AOE 网络的关键路径、工期和活动时差，并打印可视化结果。"""

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("网络存在循环，必须先检查任务依赖关系。")

    topo_order = list(nx.topological_sort(G))

    # ----------------------------------------------------
    # 1. 正向计算事件最早发生时间 (ve)
    # ----------------------------------------------------
    ve = {node: 0 for node in G.nodes}

    for node in topo_order:
        for next_node in G.successors(node):
            duration = G[node][next_node]["duration"]
            ve[next_node] = max(
                ve[next_node],
                ve[node] + duration,
            )

    project_duration = max(ve.values()) if ve else 0

    # ----------------------------------------------------
    # 2. 反向计算事件最迟发生时间 (vl)
    # ----------------------------------------------------
    vl = {node: project_duration for node in G.nodes}

    for node in reversed(topo_order):
        successors = list(G.successors(node))
        if successors:
            vl[node] = min(
                vl[next_node] - G[node][next_node]["duration"]
                for next_node in successors
            )

    # ----------------------------------------------------
    # 3. 计算每项活动的时差
    # ----------------------------------------------------
    activity_results = []
    critical_edges = []  # 记录关键边 (start, end)

    for start, end, data in G.edges(data=True):
        duration = data["duration"]
        activity_name = data.get("name", f"{start}->{end}")

        earliest_start = ve[start]
        latest_start = vl[end] - duration
        slack = latest_start - earliest_start
        is_critical = (slack == 0)

        if is_critical:
            critical_edges.append((start, end))

        activity_results.append(
            {
                "活动": activity_name,
                "起点": start,
                "终点": end,
                "持续时间": duration,
                "最早开始": earliest_start,
                "最迟开始": latest_start,
                "时差": slack,
                "关键活动": is_critical,
            }
        )

    critical_activities = [res for res in activity_results if res["关键活动"]]

    # ----------------------------------------------------
    # 🌟 4. 提取关键路径（将关键边串联成节点序列）
    # ----------------------------------------------------
    # 构造仅包含关键边的子图来提取路径
    critical_subgraph = nx.DiGraph(critical_edges)
    
    # 寻找起点（入度为0）和终点（出度为0）
    sources = [n for n in critical_subgraph.nodes if critical_subgraph.in_degree(n) == 0]
    sinks = [n for n in critical_subgraph.nodes if critical_subgraph.out_degree(n) == 0]

    critical_paths = []
    for src in sources:
        for snk in sinks:
            if nx.has_path(critical_subgraph, src, snk):
                paths = list(nx.all_simple_paths(critical_subgraph, src, snk))
                critical_paths.extend(paths)

    # 格式化路径文本 (如: "S -> 2 -> 4 -> T")
    formatted_paths = [" -> ".join(map(str, path)) for path in critical_paths]

    # ----------------------------------------------------
    # 🌟 5. 格式化终端打印 (如果 verbose=True)
    # ----------------------------------------------------
    if verbose:
        print("================ [ AOE 网络关键路径分析 ] ================")
        print(f"► 项目总工期 : {project_duration}")
        
        print("\n► 关键路径 (Critical Path):")
        for idx, p_str in enumerate(formatted_paths, 1):
            print(f"  └─ 路径 {idx}: {p_str}")

        print("\n► 活动明细与时差 (Activity Details):")
        print(f"  {'活动':<10} | {'起点->终点':<10} | {'工期':<6} | {'最早开始':<8} | {'最迟开始':<8} | {'时差(Slack)':<10} | {'关键活动'}")
        print("  " + "-" * 75)
        for act in activity_results:
            is_crit_str = "  是" if act["关键活动"] else "  否"
            edge_str = f"{act['起点']}->{act['终点']}"
            print(
                f"  {act['活动']:<10} | "
                f"{edge_str:<10} | "
                f"{act['持续时间']:<6} | "
                f"{act['最早开始']:<8} | "
                f"{act['最迟开始']:<8} | "
                f"{act['时差']:<10} | "
                f"{is_crit_str}"
            )
        print("=========================================================\n")

    return {
        "项目工期": project_duration,
        "事件最早时间": ve,
        "事件最迟时间": vl,
        "活动结果": activity_results,
        "关键活动": critical_activities,
        "关键路径节点序列": critical_paths,      # 返回列表形式 [['S', '2', '4', 'T']]
        "关键路径文本": formatted_paths,          # 返回文本形式 ['S -> 2 -> 4 -> T']
    }
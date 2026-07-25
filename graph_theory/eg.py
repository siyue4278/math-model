import matplotlib.pyplot as plt
import networkx as nx


# =========================
# 1. 构造所有图
# =========================

graphs = [
    # 经典图
    ("Cubical graph\n立方体图", nx.cubical_graph()),
    ("Cycle graph C5\n5节点环图", nx.cycle_graph(5)),
    ("Complete graph K4\n4节点完全图", nx.complete_graph(4)),
    ("Path graph P5\n5节点路径图", nx.path_graph(5)),
    ("Star graph\n中心点+4个叶子", nx.star_graph(4)),
    ("Wheel graph W5\n5节点轮图", nx.wheel_graph(5)),
    ("Petersen graph\nPetersen图", nx.petersen_graph()),
    ("Tutte graph\nTutte图", nx.tutte_graph()),

    # 随机图
    (
        "Erdos-Renyi graph\nG(100, 0.1)",
        nx.erdos_renyi_graph(100, 0.1, seed=42),
    ),
    (
        "Barabasi-Albert graph\nBA(100, 3)",
        nx.barabasi_albert_graph(100, 3, seed=42),
    ),
    (
        "Watts-Strogatz graph\nWS(100, 4, 0.1)",
        nx.watts_strogatz_graph(100, 4, 0.1, seed=42),
    ),

    # 几何与格点图
    ("Grid graph 5×5\n二维方格图", nx.grid_2d_graph(5, 5)),
    (
        "Hexagonal lattice 3×3\n六边形格点图",
        nx.hexagonal_lattice_graph(3, 3),
    ),
    (
        "Triangular lattice 3×3\n三角形格点图",
        nx.triangular_lattice_graph(3, 3),
    ),
]


# =========================
# 2. 为不同图选择合适布局
# =========================

def choose_layout(name, graph):
    """根据图的种类选择较合适的二维布局。"""

    if "Cubical" in name:
        return nx.spring_layout(graph, seed=42)

    if "Cycle" in name:
        return nx.circular_layout(graph)

    if "Complete" in name:
        return nx.circular_layout(graph)

    if "Path" in name:
        return {
            node: (index, 0)
            for index, node in enumerate(graph.nodes())
        }

    if "Star" in name:
        return nx.shell_layout(
            graph,
            nlist=[[0], list(range(1, 5))]
        )

    if "Wheel" in name:
        return nx.shell_layout(
            graph,
            nlist=[[0], list(range(1, 5))]
        )

    if "Petersen" in name:
        return nx.shell_layout(
            graph,
            nlist=[
                list(range(5)),
                list(range(5, 10)),
            ],
        )

    if "Tutte" in name:
        return nx.spring_layout(graph, seed=42)

    if "Erdos" in name:
        return nx.spring_layout(graph, seed=42)

    if "Barabasi" in name:
        return nx.spring_layout(graph, seed=42)

    if "Watts" in name:
        return nx.circular_layout(graph)

    if "Grid" in name:
        # grid_2d_graph 的节点本身就是二维坐标
        return {node: node for node in graph.nodes()}

    if "Hexagonal" in name:
        return nx.get_node_attributes(graph, "pos")

    if "Triangular" in name:
        return nx.get_node_attributes(graph, "pos")

    return nx.spring_layout(graph, seed=42)


# =========================
# 3. 绘制所有图
# =========================

fig, axes = plt.subplots(
    4,
    4,
    figsize=(18, 18),
)

axes = axes.flatten()

for index, (name, graph) in enumerate(graphs):
    ax = axes[index]
    pos = choose_layout(name, graph)

    # 大图减少节点和标签尺寸
    is_large_graph = graph.number_of_nodes() >= 40

    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        with_labels=not is_large_graph,
        node_size=40 if is_large_graph else 280,
        font_size=7,
        width=0.5 if is_large_graph else 1.0,
    )

    ax.set_title(
        f"{name}\n"
        f"|V|={graph.number_of_nodes()}, "
        f"|E|={graph.number_of_edges()}",
        fontsize=10,
    )

# 一共有14张图，但画布有16个位置，关闭多余坐标轴
for index in range(len(graphs), len(axes)):
    axes[index].axis("off")

plt.tight_layout()
plt.savefig(
    "all_networkx_graphs.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()
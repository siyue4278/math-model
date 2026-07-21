import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()
G.add_weighted_edges_from([(0, 1, 2),
                           (0, 2, 1),
                           (0, 3, 3),
                           (0, 4, 4),
                           (0, 5, 4),
                           (0, 6, 2),
                           (0, 7, 5),
                           (0, 8, 4),
                           (1, 2, 4),
                           (1, 8, 1),
                           (2, 3, 1),
                           (3, 4, 1),
                           (4, 5, 5),
                           (5, 6, 2),
                           (6, 7, 3),
                           (7, 8 ,5)])

if not nx.is_connected(G):
    raise ValueError("原图不连通，不存在生成树")

T_kruskal = nx.minimum_spanning_tree(
    G,
    weight = "weight",
    algorithm = "kruskal",
)

T_prim = nx.minimum_spanning_tree(
    G,
    weight = "weight",
    algorithm = "prim",
)

cost_kruskal = T_kruskal.size(weight = "weight")
cost_prim = T_prim.size(weight = "weight")
print("kruskal结果:")
for u, v, weight in sorted(T_kruskal.edges(data = "weight")):
    print(f"{u} -- {v}, 权重 = {weight}")

print("\nprim结果:")
for u, v, weight in sorted(T_prim.edges(data = "weight")):
    print(f"{u} -- {v}, 权重 = {weight}")



labels = {i: f"v{i}" for i in range(9)}

nx.draw(T_kruskal, labels = labels, with_labels=True, node_color = "lightyellow", node_shape="s")

plt.show()

nx.draw(T_prim, labels = labels, with_labels=True, node_color = "lightyellow", node_shape="s")

plt.show()
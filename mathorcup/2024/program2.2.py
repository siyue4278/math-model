import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ============================================================
# 1. 原模型数据 —— 与之前 MILP 完全一致
# ============================================================

# Python 下标从0开始：
# 挖1=(0), 挖2=(1), ...
E = [
    (0, 0),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2),
    (3, 1), (3, 2)
]

# 标准配车数 m_ij
m = {
    (0, 0): 1,

    (1, 0): 2,
    (1, 1): 1,

    (2, 0): 2,
    (2, 1): 2,
    (2, 2): 1,

    (3, 1): 2,
    (3, 2): 1
}

# 三种矿车库存
N = np.array([7, 7, 3], dtype=int)

# 挖掘机购买价格，万元
purchase = np.array(
    [100, 140, 200, 320],
    dtype=float
)

# 有效生产贡献：
# 销售收入 - 挖掘机有效工作燃油
A = np.array(
    [3095.04, 3830.40, 5473.92, 5792.64],
    dtype=float
)

# 挖掘机固定成本：
# 采购 + 人工 + 维护
G_exc = np.array(
    [148, 194, 263, 392],
    dtype=float
)

# 矿车运营成本：
# 燃油 + 人工 + 维护
C_truck = np.array(
    [168.96, 207.84, 253.44],
    dtype=float
)


# ============================================================
# 2. 枚举全部配置 z_ijk
# ============================================================

configs = []

for i, j in E:

    # 实际投入第j型矿车数量 k = 1,...,N_j
    for k in range(1, N[j] + 1):

        # 根据原模型最优性压缩得到
        x = math.ceil(k / m[(i, j)])
        u = k / m[(i, j)]

        # 完全代入原目标函数：
        # A_i*u - G_i*x - C_j*k
        profit = (
            A[i] * u
            - G_exc[i] * x
            - C_truck[j] * k
        )

        # 该配置占用的启动资金
        budget = purchase[i] * x

        configs.append({
            "i": i,
            "j": j,
            "k": k,
            "x": x,
            "u": u,
            "profit": profit,
            "budget": budget
        })


print("配置变量总数 =", len(configs))


# ============================================================
# 3. 查看全部配置
# ============================================================

print("\n前10个配置：")

for idx, cfg in enumerate(configs[:10]):

    print(
        f"z[{cfg['i']+1},{cfg['j']+1},{cfg['k']}] : "
        f"x={cfg['x']}, "
        f"u={cfg['u']:.2f}, "
        f"profit={cfg['profit']:.2f}, "
        f"budget={cfg['budget']:.2f}"
    )


# ============================================================
# 4. 48个 z 全部是0-1变量
# ============================================================

n = len(configs)

# scipy milp 是最小化，所以利润取负
c = -np.array(
    [cfg["profit"] for cfg in configs],
    dtype=float
)

integrality = np.ones(n, dtype=int)

bounds = Bounds(
    np.zeros(n),
    np.ones(n)
)


# ============================================================
# 5. 约束
# ============================================================

rows = []
lower = []
upper = []


# ------------------------------------------------------------
# 约束A：每一种挖掘机最多选择一个配置
# ------------------------------------------------------------

for i in range(4):

    row = np.array([
        1.0 if cfg["i"] == i else 0.0
        for cfg in configs
    ])

    rows.append(row)
    lower.append(-np.inf)
    upper.append(1.0)


# ------------------------------------------------------------
# 约束B：至少3种挖掘机
# 因为每种最多选一个配置，所以 sum(z) 就是型号数
# ------------------------------------------------------------

row = np.ones(n)

rows.append(row)
lower.append(3.0)
upper.append(np.inf)


# ------------------------------------------------------------
# 约束C：启动资金 <= 2400万元
# ------------------------------------------------------------

row = np.array(
    [cfg["budget"] for cfg in configs],
    dtype=float
)

rows.append(row)
lower.append(-np.inf)
upper.append(2400.0)


# ------------------------------------------------------------
# 约束D：三种矿车库存
# ------------------------------------------------------------

for j in range(3):

    row = np.array([
        cfg["k"] if cfg["j"] == j else 0.0
        for cfg in configs
    ])

    rows.append(row)
    lower.append(-np.inf)
    upper.append(float(N[j]))


# ============================================================
# 6. 精确求解
# ============================================================

A_cons = np.vstack(rows)

constraints = LinearConstraint(
    A_cons,
    np.array(lower),
    np.array(upper)
)

result = milp(
    c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints
)


# ============================================================
# 7. 输出最优配置
# ============================================================

print("\n求解成功：", result.success)
print("配置模型最大利润：", -result.fun, "万元")

selected = []

for idx, z in enumerate(result.x):

    if z > 0.5:
        cfg = configs[idx]
        selected.append(cfg)

        print(
            f"选择 z[{cfg['i']+1},{cfg['j']+1},{cfg['k']}] = 1"
            f" -> x={cfg['x']}, "
            f"u={cfg['u']:.2f}, "
            f"profit={cfg['profit']:.2f}, "
            f"budget={cfg['budget']:.2f}"
        )


# ============================================================
# 8. 恢复原变量
# ============================================================

x_rec = np.zeros(4)
truck_used = np.zeros(3)
u_rec = {}

for cfg in selected:

    i = cfg["i"]
    j = cfg["j"]

    x_rec[i] = cfg["x"]
    truck_used[j] += cfg["k"]
    u_rec[(i, j)] = cfg["u"]


print("\n恢复的挖掘机购买量：")
for i in range(4):
    print(f"挖{i+1}: {x_rec[i]:.0f}")

print("\n矿车总使用量：")
for j in range(3):
    print(f"矿{j+1}: {truck_used[j]:.0f} / {N[j]}")

print("\n启动资金：")
total_budget = sum(cfg["budget"] for cfg in selected)
print(total_budget, "万元")
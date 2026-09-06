import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


# ============================================================
# 2024 MathorCup D 题 —— 问题3
# 经典 MILP 基准模型
#
# 目的：
# 1. 完整继承问题2已经锁定的经济口径与 11 条约束结构；
# 2. 扩展到 10 种挖掘机、10 种矿车；
# 3. 先求出问题3的精确经典基准解；
# 4. 后续再用同一个问题构造完整 QUBO / subQUBO。
# ============================================================


# ============================================================
# 1. 基础数据
# ============================================================

# 矿车库存：
# 矿1~矿5：各 5 辆
# 矿6~矿10：各 3 辆
N = np.array(
    [5, 5, 5, 5, 5, 3, 3, 3, 3, 3],
    dtype=float
)

# 十种挖掘机参数
# 斗容（m^3）
bucket = np.array(
    [0.9, 1.2, 1.8, 2.1, 2.6, 3.5, 5.0, 6.0, 8.0, 10.0],
    dtype=float
)

# 作业效率（斗/h）
efficiency = np.array(
    [190, 175, 165, 150, 140, 130, 120, 110, 105, 100],
    dtype=float
)

# 挖掘机油耗（L/h）
fuel_exc = np.array(
    [28, 30, 34, 38, 42, 50, 60, 75, 90, 100],
    dtype=float
)

# 挖掘机采购价格（万元）
purchase = np.array(
    [100, 140, 200, 320, 440, 500, 640, 760, 860, 1000],
    dtype=float
)

# 挖掘机人工成本（元/月）
labor_exc = np.array(
    [7000, 7500, 8500, 9000, 10000, 12000, 13000, 16000, 18000, 20000],
    dtype=float
)

# 挖掘机维护成本（元/月）
maint_exc = np.array(
    [1000, 1500, 2000, 3000, 5000, 8000, 10000, 13000, 15000, 18000],
    dtype=float
)

# 十种矿车参数
# 油耗（L/h）
fuel_truck = np.array(
    [15, 18, 22, 27, 33, 40, 50, 55, 64, 70],
    dtype=float
)

# 人工成本（元/月）
labor_truck = np.array(
    [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 15000],
    dtype=float
)

# 维护成本（元/月）
maint_truck = np.array(
    [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    dtype=float
)


# ============================================================
# 2. 匹配关系表 E 与标准配车数 m_ij
#
# Python 下标从 0 开始：
# (i, j) = (挖掘机型号-1, 矿车型号-1)
# ============================================================

m = {
    # 挖1
    (0, 0): 3,
    (0, 1): 3,
    (0, 2): 2,

    # 挖2
    (1, 0): 3,
    (1, 1): 3,
    (1, 2): 3,
    (1, 3): 2,

    # 挖3
    (2, 0): 4,
    (2, 1): 3,
    (2, 2): 3,
    (2, 3): 3,
    (2, 4): 2,

    # 挖4
    (3, 0): 5,
    (3, 1): 4,
    (3, 2): 3,
    (3, 3): 3,
    (3, 4): 3,
    (3, 5): 2,

    # 挖5
    (4, 1): 5,
    (4, 2): 4,
    (4, 3): 3,
    (4, 4): 3,
    (4, 5): 3,
    (4, 6): 2,
    (4, 7): 2,

    # 挖6
    (5, 2): 5,
    (5, 3): 4,
    (5, 4): 3,
    (5, 5): 3,
    (5, 6): 3,
    (5, 7): 2,
    (5, 8): 2,

    # 挖7
    (6, 2): 5,
    (6, 3): 5,
    (6, 4): 4,
    (6, 5): 3,
    (6, 6): 3,
    (6, 7): 3,
    (6, 8): 2,
    (6, 9): 2,

    # 挖8
    (7, 3): 5,
    (7, 4): 5,
    (7, 5): 4,
    (7, 6): 3,
    (7, 7): 3,
    (7, 8): 3,
    (7, 9): 3,

    # 挖9
    (8, 4): 5,
    (8, 5): 5,
    (8, 6): 4,
    (8, 7): 3,
    (8, 8): 3,
    (8, 9): 3,

    # 挖10
    (9, 5): 5,
    (9, 6): 5,
    (9, 7): 4,
    (9, 8): 3,
    (9, 9): 3,
}

E = list(m.keys())

print("允许匹配边数 =", len(E))
assert len(E) == 58


# ============================================================
# 3. 五年经济系数
#
# 5 年：
# 5 * 12 * 20 * 8 = 9600 h
#
# 矿石售价：20 元/m^3
# 油价：7 元/L
#
# 全部统一为“万元”
# ============================================================

HOURS_5Y = 5 * 12 * 20 * 8   # 9600 h

# 每型挖掘机满负荷产量（m^3/h）
q = bucket * efficiency

# 5年满负荷销售收入（万元）
R = q * HOURS_5Y * 20 / 10000.0

# 5年挖掘机有效作业燃油成本（万元）
F_exc = fuel_exc * HOURS_5Y * 7 / 10000.0

# 生产收益系数
# A_i = R_i - F_i^exc
A = R - F_exc

# 挖掘机固定成本：
# 采购 + 5年人工 + 5年维护
G_exc = (
    purchase
    + 60 * (labor_exc + maint_exc) / 10000.0
)

# 每辆矿车5年运营成本：
# 燃油 + 人工 + 维护
C_truck = (
    fuel_truck * HOURS_5Y * 7 / 10000.0
    + 60 * (labor_truck + maint_truck) / 10000.0
)


print("\n====== 五年经济系数 ======")

print("A_i =")
print(np.round(A, 2))

print("G_i =")
print(np.round(G_exc, 2))

print("C_j =")
print(np.round(C_truck, 2))


# ============================================================
# 4. 原 MILP 变量
#
# x_i     : 第 i 型挖掘机购买数量，整数
# a_i     : 是否购买第 i 型挖掘机，0/1
# y_ij    : 第 i 型挖掘机是否选择第 j 型矿车匹配，0/1
# k_ij    : 实际配置的第 j 型矿车数量，整数
# u_ij    : 满负荷等效作业能力，连续
#
# 目标：
#
# max
# sum_(i,j) A_i u_ij
# - sum_i G_i x_i
# - sum_(i,j) C_j k_ij
# ============================================================

n_exc = 10
n_edges = len(E)

# 变量下标
idx_x = {
    i: i
    for i in range(n_exc)
}

idx_a = {
    i: 10 + i
    for i in range(n_exc)
}

base_y = 20

idx_y = {
    e: base_y + t
    for t, e in enumerate(E)
}

base_k = base_y + n_edges

idx_k = {
    e: base_k + t
    for t, e in enumerate(E)
}

base_u = base_k + n_edges

idx_u = {
    e: base_u + t
    for t, e in enumerate(E)
}

n_var = base_u + n_edges

print("\nMILP总变量数 =", n_var)


# ============================================================
# 5. 目标函数
#
# scipy.optimize.milp 做最小化，
# 所以把最大化利润改写成最小化 -利润。
# ============================================================

c = np.zeros(
    n_var,
    dtype=float
)

# + G_i x_i
for i in range(n_exc):
    c[idx_x[i]] = G_exc[i]

# + C_j k_ij - A_i u_ij
for e in E:

    i, j = e

    c[idx_k[e]] = C_truck[j]

    c[idx_u[e]] = -A[i]


# ============================================================
# 6. 变量上下界与整数性
# ============================================================

lower = np.zeros(
    n_var,
    dtype=float
)

upper = np.full(
    n_var,
    np.inf,
    dtype=float
)

# 预算决定的挖掘机数量粗上界
M = np.floor(
    4000 / purchase
).astype(int)

# x_i
for i in range(n_exc):
    upper[idx_x[i]] = M[i]

# a_i
for i in range(n_exc):
    upper[idx_a[i]] = 1

# y_ij, k_ij, u_ij
for e in E:

    i, j = e

    upper[idx_y[e]] = 1

    upper[idx_k[e]] = N[j]

    # u_ij <= x_i <= M_i
    upper[idx_u[e]] = M[i]


# integrality:
# 0 = continuous
# 1 = integer
integrality = np.zeros(
    n_var,
    dtype=int
)

for i in range(n_exc):

    integrality[idx_x[i]] = 1

    integrality[idx_a[i]] = 1

for e in E:

    integrality[idx_y[e]] = 1

    integrality[idx_k[e]] = 1


# ============================================================
# 7. 约束矩阵
# ============================================================

row_idx = []
col_idx = []
data = []

constraint_lb = []
constraint_ub = []

row = 0


def add_constraint(
    coeffs,
    lb=-np.inf,
    ub=np.inf
):

    global row

    for var_idx, value in coeffs.items():

        if abs(value) > 0:

            row_idx.append(row)

            col_idx.append(var_idx)

            data.append(value)

    constraint_lb.append(lb)

    constraint_ub.append(ub)

    row += 1


# ------------------------------------------------------------
# ① 启动资金
#
# sum p_i x_i <= 4000
# ------------------------------------------------------------

add_constraint(
    {
        idx_x[i]: purchase[i]
        for i in range(n_exc)
    },
    ub=4000
)


# ------------------------------------------------------------
# ② a_i <= x_i <= M_i a_i
# ------------------------------------------------------------

for i in range(n_exc):

    # a_i - x_i <= 0
    add_constraint(
        {
            idx_a[i]: 1,
            idx_x[i]: -1
        },
        ub=0
    )

    # x_i - M_i a_i <= 0
    add_constraint(
        {
            idx_x[i]: 1,
            idx_a[i]: -M[i]
        },
        ub=0
    )


# ------------------------------------------------------------
# ③ 至少 5 种挖掘机
#
# sum a_i >= 5
# ------------------------------------------------------------

add_constraint(
    {
        idx_a[i]: 1
        for i in range(n_exc)
    },
    lb=5
)


# ------------------------------------------------------------
# ④ 每种已购买挖掘机只选择一种矿车类型
#
# sum_j y_ij = a_i
# ------------------------------------------------------------

for i in range(n_exc):

    coeffs = {
        idx_a[i]: -1
    }

    for e in E:

        if e[0] == i:

            coeffs[idx_y[e]] = 1

    add_constraint(
        coeffs,
        lb=0,
        ub=0
    )


# ------------------------------------------------------------
# ⑤ 若选择匹配，则至少配置 1 辆矿车；
#    且不能超过该类矿车库存
#
# y_ij <= k_ij <= N_j y_ij
# ------------------------------------------------------------

for e in E:

    i, j = e

    # y_ij - k_ij <= 0
    add_constraint(
        {
            idx_y[e]: 1,
            idx_k[e]: -1
        },
        ub=0
    )

    # k_ij - N_j y_ij <= 0
    add_constraint(
        {
            idx_k[e]: 1,
            idx_y[e]: -N[j]
        },
        ub=0
    )


# ------------------------------------------------------------
# ⑥ 每类矿车总使用量不能超过库存
#
# sum_i k_ij <= N_j
# ------------------------------------------------------------

for j in range(10):

    coeffs = {
        idx_k[e]: 1
        for e in E
        if e[1] == j
    }

    add_constraint(
        coeffs,
        ub=N[j]
    )


# ------------------------------------------------------------
# ⑦ 同一种挖掘机的总等效作业量不能超过购买数
#
# sum_j u_ij <= x_i
# ------------------------------------------------------------

for i in range(n_exc):

    coeffs = {
        idx_x[i]: -1
    }

    for e in E:

        if e[0] == i:

            coeffs[idx_u[e]] = 1

    add_constraint(
        coeffs,
        ub=0
    )


# ------------------------------------------------------------
# ⑧ 矿车不足时产能按比例下降
#
# m_ij u_ij <= k_ij
# ------------------------------------------------------------

for e in E:

    add_constraint(
        {
            idx_u[e]: m[e],
            idx_k[e]: -1
        },
        ub=0
    )


# ============================================================
# 8. 构造 scipy LinearConstraint
# ============================================================

A_constraint = coo_matrix(
    (
        data,
        (row_idx, col_idx)
    ),
    shape=(row, n_var)
).tocsr()

constraints = LinearConstraint(
    A_constraint,
    np.array(
        constraint_lb,
        dtype=float
    ),
    np.array(
        constraint_ub,
        dtype=float
    )
)

bounds = Bounds(
    lower,
    upper
)


# ============================================================
# 9. 求解
# ============================================================

result = milp(
    c=c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraints,
    options={
        "disp": False
    }
)


print("\n================ MILP 求解结果 ================")

print("success =", result.success)
print("message =", result.message)

if not result.success:

    raise RuntimeError(
        "MILP 没有成功求解。"
    )


solution = result.x

profit_opt = -result.fun


# ============================================================
# 10. 解码最优方案
# ============================================================

x_opt = np.rint([
    solution[idx_x[i]]
    for i in range(n_exc)
]).astype(int)

a_opt = np.rint([
    solution[idx_a[i]]
    for i in range(n_exc)
]).astype(int)

budget_used = float(
    purchase @ x_opt
)

type_count = int(
    np.sum(a_opt)
)


print("\n五年最大净利润 =", profit_opt, "万元")

print(
    "启动资金 =",
    budget_used,
    "/ 4000 万元"
)

print(
    "购买挖掘机数量 x =",
    x_opt.tolist()
)

print(
    "使用挖掘机型号数 =",
    type_count
)


print("\n====== 最优匹配与实际矿车配置 ======")

truck_total = np.zeros(
    10,
    dtype=float
)

component_profit_sum = 0.0

for e in E:

    i, j = e

    y_val = solution[idx_y[e]]

    k_val = solution[idx_k[e]]

    u_val = solution[idx_u[e]]

    if y_val > 0.5:

        # 修正数值误差，便于显示
        k_display = int(round(k_val))

        u_display = float(u_val)

        truck_total[j] += k_display

        component_profit = (
            A[i] * u_display
            - G_exc[i] * x_opt[i]
            - C_truck[j] * k_display
        )

        component_profit_sum += (
            component_profit
        )

        print(
            f"挖{i+1} -> 矿{j+1}: "
            f"x={x_opt[i]}, "
            f"m={m[e]}, "
            f"k={k_display}, "
            f"u={u_display:.6f}, "
            f"利润贡献={component_profit:.2f}"
        )


print("\n矿车总使用量 =")
print([
    int(round(v))
    for v in truck_total
])

print(
    "矿车库存 =",
    [
        int(v)
        for v in N
    ]
)

print(
    "\n分项利润求和 =",
    component_profit_sum,
    "万元"
)

print(
    "与求解器目标差 =",
    abs(
        component_profit_sum
        - profit_opt
    )
)


# ============================================================
# 11. 为下一步 QUBO / subQUBO 做规模预检查
#
# 若继续采用问题2的配置变量：
#
# z_ijk = 1
# 表示：
# 第 i 型挖掘机匹配第 j 型矿车，
# 实际投入 k 辆该矿车。
#
# 每条允许边 (i,j) 需要 N_j 个配置变量。
# ============================================================

config_count = int(
    sum(
        N[j]
        for i, j in E
    )
)

print(
    "\n====== 下一步 QUBO 规模预检查 ======"
)

print(
    "完整配置变量数 =",
    config_count
)

print(
    "是否已超过 100 bit =",
    config_count > 100
)

assert config_count == 234

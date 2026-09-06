import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# =========================
# 1. 数据
# =========================

I = range(4)

# 允许的匹配，Python使用0开始
E = [
    (0, 0),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2),
    (3, 1), (3, 2)
]

m = {
    (0, 0): 1,
    (1, 0): 2, (1, 1): 1,
    (2, 0): 2, (2, 1): 2, (2, 2): 1,
    (3, 1): 2, (3, 2): 1
}

# 挖掘机采购价格（万元）
purchase = np.array([100, 140, 200, 320], dtype=float)

# 最大购买数量
M = np.floor(2400 / purchase).astype(int)

# 已有矿车数量
N = np.array([7, 7, 3], dtype=int)

# 五年满负荷销售收入
R = np.array([3283.2, 4032.0, 5702.4, 6048.0])

# 五年满负荷挖掘机燃油成本
F_exc = np.array([188.16, 201.60, 228.48, 255.36])

# 单位有效产能：收入 - 挖掘机燃油
A = R - F_exc

# 挖掘机固定成本：采购 + 人工 + 维护
G_exc = np.array([148.0, 194.0, 263.0, 392.0])

# 矿车五年运营成本：燃油 + 人工 + 维护
C_truck = np.array([168.96, 207.84, 253.44])


# =========================
# 2. 建立变量索引
# =========================

idx = {}
names = []

def add_var(name):
    idx[name] = len(names)
    names.append(name)

# x_i
for i in I:
    add_var(("x", i))

# a_i
for i in I:
    add_var(("a", i))

# y_ij
for i, j in E:
    add_var(("y", i, j))

# k_ij
for i, j in E:
    add_var(("k", i, j))

# u_ij
for i, j in E:
    add_var(("u", i, j))

n = len(names)


# =========================
# 3. 目标函数
# scipy milp 默认最小化
# 所以最大利润取负号
# =========================

c = np.zeros(n)

for i in I:
    c[idx[("x", i)]] = G_exc[i]

for i, j in E:
    c[idx[("k", i, j)]] = C_truck[j]
    c[idx[("u", i, j)]] = -A[i]


# =========================
# 4. 变量上下界和整数性
# =========================

lb = np.zeros(n)
ub = np.full(n, np.inf)
integrality = np.zeros(n, dtype=int)

for i in I:
    ub[idx[("x", i)]] = M[i]
    integrality[idx[("x", i)]] = 1

    ub[idx[("a", i)]] = 1
    integrality[idx[("a", i)]] = 1

for i, j in E:
    ub[idx[("y", i, j)]] = 1
    integrality[idx[("y", i, j)]] = 1

    ub[idx[("k", i, j)]] = N[j]
    integrality[idx[("k", i, j)]] = 1

    ub[idx[("u", i, j)]] = M[i]


# =========================
# 5. 添加约束
# =========================

rows = []
lower = []
upper = []

def add_constraint(coeffs, lo=-np.inf, hi=np.inf):
    row = np.zeros(n)

    for key, value in coeffs.items():
        row[idx[key]] = value

    rows.append(row)
    lower.append(lo)
    upper.append(hi)


# 启动资金
add_constraint(
    {("x", i): purchase[i] for i in I},
    hi=2400
)


# x_i 与 a_i 绑定
for i in I:

    # x_i >= a_i
    add_constraint(
        {
            ("x", i): 1,
            ("a", i): -1
        },
        lo=0
    )

    # x_i <= M_i a_i
    add_constraint(
        {
            ("x", i): 1,
            ("a", i): -M[i]
        },
        hi=0
    )


# 至少3种挖掘机
add_constraint(
    {("a", i): 1 for i in I},
    lo=3
)


# 每种挖掘机恰好选择一种矿车
for i in I:

    coeff = {("a", i): -1}

    for ii, j in E:
        if ii == i:
            coeff[("y", ii, j)] = 1

    add_constraint(coeff, lo=0, hi=0)


# k_ij 与 y_ij 绑定
for i, j in E:

    # k_ij >= y_ij
    add_constraint(
        {
            ("k", i, j): 1,
            ("y", i, j): -1
        },
        lo=0
    )

    # k_ij <= N_j y_ij
    add_constraint(
        {
            ("k", i, j): 1,
            ("y", i, j): -N[j]
        },
        hi=0
    )


# 各型号矿车库存
for j in range(3):

    coeff = {}

    for i, jj in E:
        if jj == j:
            coeff[("k", i, j)] = 1

    add_constraint(coeff, hi=N[j])


# 有效挖掘机数量 <= 实际购买量
for i in I:

    coeff = {("x", i): -1}

    for ii, j in E:
        if ii == i:
            coeff[("u", ii, j)] = 1

    add_constraint(coeff, hi=0)


# 没选匹配则 u_ij = 0
for i, j in E:

    add_constraint(
        {
            ("u", i, j): 1,
            ("y", i, j): -M[i]
        },
        hi=0
    )


# 标准矿车配比约束
# m_ij * u_ij <= k_ij
for i, j in E:

    add_constraint(
        {
            ("u", i, j): m[(i, j)],
            ("k", i, j): -1
        },
        hi=0
    )


# =========================
# 6. 求解
# =========================

A_cons = np.vstack(rows)

constraints = LinearConstraint(
    A_cons,
    np.array(lower),
    np.array(upper)
)

result = milp(
    c,
    integrality=integrality,
    bounds=Bounds(lb, ub),
    constraints=constraints
)


# =========================
# 7. 输出
# =========================

print("求解成功：", result.success)
print("五年最大净利润：", -result.fun, "万元")

print("\n挖掘机购买方案：")
for i in I:
    value = result.x[idx[("x", i)]]
    print(f"挖{i+1}: {value:.4f}")

print("\n矿车匹配与分配：")
for i, j in E:

    y = result.x[idx[("y", i, j)]]
    k = result.x[idx[("k", i, j)]]
    u = result.x[idx[("u", i, j)]]

    if y > 0.5:
        print(
            f"挖{i+1} -> 矿{j+1}: "
            f"k={k:.4f}, u={u:.4f}"
        )

print("\n购买总成本：")
cost = sum(
    purchase[i] * result.x[idx[('x', i)]]
    for i in I
)

print(cost, "万元")
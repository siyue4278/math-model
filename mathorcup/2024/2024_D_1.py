import dimod
from dwave.samplers import SimulatedAnnealingSampler


# ============================================================
# 1. 参数
# ============================================================

# 缩放后的利润
profit = {
    1: 2,
    2: 3,
    3: 5,
    4: 6,
}

# 缩放后的购买成本
cost = {
    1: 5,
    2: 7,
    3: 10,
    4: 16,
}

# x_i 的 QUBO 编码
#
# x1 = y1 + z11 + 2z12 + 4z13 + 8z14 + 8z15   <= 24
# x2 = y2 + z21 + 2z22 + 4z23 + 8z24 + z25    <= 17
# x3 = y3 + z31 + 2z32 + 4z33 + 4z34           <= 12
# x4 = y4 + z41 + 2z42 + 3z43                  <= 7

encoding = {
    1: {
        "y1": 1,
        "z11": 1,
        "z12": 2,
        "z13": 4,
        "z14": 8,
        "z15": 8,
    },
    2: {
        "y2": 1,
        "z21": 1,
        "z22": 2,
        "z23": 4,
        "z24": 8,
        "z25": 1,
    },
    3: {
        "y3": 1,
        "z31": 1,
        "z32": 2,
        "z33": 4,
        "z34": 4,
    },
    4: {
        "y4": 1,
        "z41": 1,
        "z42": 2,
        "z43": 3,
    },
}

# 保守地先统一取较大的罚系数
P = 1
P_budget = P
P_type = P
P_link = P


# ============================================================
# 2. 创建空的 Binary Quadratic Model
# ============================================================

bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)


# ============================================================
# 3. 辅助函数：
#    向 BQM 添加 P * (c + sum(a_i*x_i))^2
# ============================================================

def add_squared_penalty(bqm, coeffs, constant, P):
    """
    添加:
        P * (constant + Σ a_i x_i)^2

    因为 x_i ∈ {0,1}，所以 x_i^2 = x_i。
    """

    # 常数平方
    bqm.offset += P * constant**2

    variables = list(coeffs.keys())

    # 一次项：
    # a_i^2 x_i + 2*c*a_i*x_i
    for v in variables:
        a = coeffs[v]
        linear_bias = P * (a**2 + 2 * constant * a)
        bqm.add_variable(v, linear_bias)

    # 二次项：
    # 2*a_i*a_j*x_i*x_j
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            vi = variables[i]
            vj = variables[j]

            ai = coeffs[vi]
            aj = coeffs[vj]

            bqm.add_interaction(
                vi,
                vj,
                P * 2 * ai * aj
            )


# ============================================================
# 4. 利润目标
#
# min  -(2*x1 + 3*x2 + 5*x3 + 6*x4)
# ============================================================

for i in range(1, 5):
    for var, weight in encoding[i].items():
        bqm.add_variable(
            var,
            -profit[i] * weight
        )


# ============================================================
# 5. 预算约束
#
# 5*x1 + 7*x2 + 10*x3 + 16*x4 + s = 120
# ============================================================

budget_coeffs = {}

for i in range(1, 5):
    for var, weight in encoding[i].items():
        budget_coeffs[var] = (
            budget_coeffs.get(var, 0)
            + cost[i] * weight
        )

# slack:
# s = q0 + 2q1 + 4q2 + 8q3 + 16q4 + 32q5 + 57q6
slack_budget = {
    "q0": 1,
    "q1": 2,
    "q2": 4,
    "q3": 8,
    "q4": 16,
    "q5": 32,
    "q6": 57,
}

budget_coeffs.update(slack_budget)

# expression:
# budget + slack - 120 = 0
add_squared_penalty(
    bqm,
    budget_coeffs,
    constant=-120,
    P=P_budget,
)


# ============================================================
# 6. 至少选择 3 种型号
#
# y1 + y2 + y3 + y4 - t = 3
# ============================================================

type_coeffs = {
    "y1": 1,
    "y2": 1,
    "y3": 1,
    "y4": 1,
    "t": -1,
}

add_squared_penalty(
    bqm,
    type_coeffs,
    constant=-3,
    P=P_type,
)


# ============================================================
# 7. Link penalty
#
# z_ik <= y_i
#
# penalty = P * z_ik * (1 - y_i)
# ============================================================

for i in range(1, 5):

    y = f"y{i}"

    for var in encoding[i]:

        # y_i 本身不需要约束自己
        if var == y:
            continue

        z = var

        # P*z
        bqm.add_variable(z, P_link)

        # -P*z*y
        bqm.add_interaction(z, y, -P_link)


# ============================================================
# 8. 解码函数
# ============================================================

def decode_x(sample):

    x = {}

    for i in range(1, 5):
        value = 0

        for var, weight in encoding[i].items():
            value += weight * sample[var]

        x[i] = int(value)

    return x


# ============================================================
# 9. 先验证我们已经知道的 MILP 最优方案
#
# x = (1, 2, 10, 0)
# ============================================================

known = {v: 0 for v in bqm.variables}

# x1 = 1
known["y1"] = 1

# x2 = 2 = y2 + z21
known["y2"] = 1
known["z21"] = 1

# x3 = 10
# 10 = y3 + z31 + z33 + z34
#    = 1 + 1 + 4 + 4
known["y3"] = 1
known["z31"] = 1
known["z33"] = 1
known["z34"] = 1

# x4 = 0
known["y4"] = 0

# 一共3种型号，因此 t=0
known["t"] = 0

# 花费 = 2380万元
# 缩放后 = 119
# 所以剩余预算 = 1
known["q0"] = 1

print("====== MILP已知最优方案检查 ======")
print("x =", decode_x(known))
print("QUBO energy =", bqm.energy(known))


# ============================================================
# 10. 模拟退火求解 QUBO
# ============================================================

sampler = SimulatedAnnealingSampler()

sampleset = sampler.sample(
    bqm,
    num_reads=5000,
    num_sweeps=2000,
)

best = sampleset.first

x_best = decode_x(best.sample)

actual_cost = (
    100 * x_best[1]
    + 140 * x_best[2]
    + 200 * x_best[3]
    + 320 * x_best[4]
)

actual_profit = (
    2000 * x_best[1]
    + 3000 * x_best[2]
    + 5000 * x_best[3]
    + 6000 * x_best[4]
)

selected_types = sum(
    best.sample[f"y{i}"]
    for i in range(1, 5)
)

print()
print("====== 模拟退火结果 ======")
print("x =", x_best)
print("y =", [best.sample[f'y{i}'] for i in range(1, 5)])
print("购买型号数 =", selected_types)
print("采购成本 =", actual_cost, "万元")
print("总利润 =", actual_profit, "万元")
print("QUBO energy =", best.energy)
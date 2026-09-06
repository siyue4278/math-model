import math
import itertools
import dimod
import numpy as np

# ============================================================
# 1. 原模型数据
# ============================================================

E = [
    (0, 0),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2),
    (3, 1), (3, 2)
]

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

# 矿车库存
N = np.array([7, 7, 3], dtype=int)

# 挖掘机购买价格，万元
purchase = np.array([100, 140, 200, 320], dtype=float)

# r_i - F_i^exc
# 即：满负荷销售收入 - 挖掘机有效作业燃油
A = np.array([
    3095.04,
    3830.40,
    5473.92,
    5792.64
])

# 挖掘机固定成本：采购 + 人工 + 维护
G_exc = np.array([
    148.0,
    194.0,
    263.0,
    392.0
])

# 矿车总运营成本：燃油 + 人工 + 维护
C_truck = np.array([
    168.96,
    207.84,
    253.44
])


# ============================================================
# 2. 生成48个配置变量 z_ijk
# ============================================================

configs = []

for i, j in E:
    for k in range(1, N[j] + 1):

        x = math.ceil(k / m[(i, j)])
        u = k / m[(i, j)]

        # 原目标函数的配置贡献
        profit = (
            A[i] * u
            - G_exc[i] * x
            - C_truck[j] * k
        )

        budget = purchase[i] * x

        # 用1开始编号，便于和论文一致
        name = f"z_{i+1}_{j+1}_{k}"

        configs.append({
            "name": name,
            "i": i,
            "j": j,
            "k": k,
            "x": x,
            "u": u,
            "profit": profit,
            "profit_scaled": profit / 1000.0,
            "budget": budget,
            "budget_scaled": int(round(budget / 20))
        })


print("配置变量数 =", len(configs))
assert len(configs) == 48


# ============================================================
# 3. 建立 BQM
# ============================================================

bqm = dimod.BinaryQuadraticModel("BINARY")


# ------------------------------------------------------------
# 目标函数
#
# max profit
# 等价于
# min -profit
#
# 利润统一除以1000
# ------------------------------------------------------------

for cfg in configs:
    bqm.add_linear(
        cfg["name"],
        -cfg["profit_scaled"]
    )


# ============================================================
# 4. 这一轮罚系数只取占位值
#
# 注意：
# 这里暂时不拿它们跑SA！
# 我们只检查QUBO结构和已知可行解能量。
# ============================================================

lambda_B = 1.0
lambda_T = 1.0
lambda_M = 1.0

lambda_1 = 1.0
lambda_2 = 1.0
lambda_3 = 1.0


# ============================================================
# 5. 预算约束
#
# 原：
# sum B_c z_c <= 2400
#
# 除以20：
# sum b_c z_c <= 120
#
# 加 slack:
# sum b_c z_c + s_B = 120
# ============================================================

budget_slack = {
    "sb_0": 1,
    "sb_1": 2,
    "sb_2": 4,
    "sb_3": 8,
    "sb_4": 16,
    "sb_5": 32,
    "sb_6": 57
}

budget_terms = []

for cfg in configs:
    budget_terms.append(
        (cfg["name"], cfg["budget_scaled"])
    )

for var, weight in budget_slack.items():
    budget_terms.append((var, weight))

bqm.add_linear_equality_constraint(
    budget_terms,
    lagrange_multiplier=lambda_B,
    constant=-120
)


# ============================================================
# 6. 至少3种挖掘机
#
# 在“每种型号最多一个配置”成立时：
#
# sum z = 型号数量
#
# 3种：t=0
# 4种：t=1
#
# sum z - t = 3
# ============================================================

type_terms = []

for cfg in configs:
    type_terms.append((cfg["name"], 1))

type_terms.append(("t_type", -1))

bqm.add_linear_equality_constraint(
    type_terms,
    lagrange_multiplier=lambda_T,
    constant=-3
)


# ============================================================
# 7. 同一种挖掘机至多选择一个配置
#
# 对同一个 i 的所有配置两两加入 z_c z_c'
# ============================================================

for i in range(4):

    same_type = [
        cfg["name"]
        for cfg in configs
        if cfg["i"] == i
    ]

    for z1, z2 in itertools.combinations(same_type, 2):
        bqm.add_quadratic(
            z1,
            z2,
            lambda_M
        )


# ============================================================
# 8. 矿1库存
#
# sum k*z_i1k + s1 = 7
# ============================================================

truck1_terms = []

for cfg in configs:
    if cfg["j"] == 0:
        truck1_terms.append(
            (cfg["name"], cfg["k"])
        )

truck1_slack = {
    "s1_0": 1,
    "s1_1": 2,
    "s1_2": 4
}

for var, weight in truck1_slack.items():
    truck1_terms.append((var, weight))

bqm.add_linear_equality_constraint(
    truck1_terms,
    lagrange_multiplier=lambda_1,
    constant=-7
)


# ============================================================
# 9. 矿2库存
#
# sum k*z_i2k + s2 = 7
# ============================================================

truck2_terms = []

for cfg in configs:
    if cfg["j"] == 1:
        truck2_terms.append(
            (cfg["name"], cfg["k"])
        )

truck2_slack = {
    "s2_0": 1,
    "s2_1": 2,
    "s2_2": 4
}

for var, weight in truck2_slack.items():
    truck2_terms.append((var, weight))

bqm.add_linear_equality_constraint(
    truck2_terms,
    lagrange_multiplier=lambda_2,
    constant=-7
)


# ============================================================
# 10. 矿3库存
#
# sum k*z_i3k + s3 = 3
# ============================================================

truck3_terms = []

for cfg in configs:
    if cfg["j"] == 2:
        truck3_terms.append(
            (cfg["name"], cfg["k"])
        )

truck3_slack = {
    "s3_0": 1,
    "s3_1": 2
}

for var, weight in truck3_slack.items():
    truck3_terms.append((var, weight))

bqm.add_linear_equality_constraint(
    truck3_terms,
    lagrange_multiplier=lambda_3,
    constant=-3
)


# ============================================================
# 11. 检查变量数
# ============================================================

print("\n====== QUBO规模 ======")

print("总二进制变量数 =", len(bqm.variables))

print("其中：")
print("配置变量 =", 48)
print("预算slack =", 7)
print("型号辅助 =", 1)
print("矿1 slack =", 3)
print("矿2 slack =", 3)
print("矿3 slack =", 2)

assert len(bqm.variables) == 64


# ============================================================
# 12. 把原MILP已知最优解编码成64-bit sample
# ============================================================

sample_opt = {
    var: 0
    for var in bqm.variables
}

# 已知最优配置
sample_opt["z_1_1_7"] = 1
sample_opt["z_2_2_7"] = 1
sample_opt["z_3_3_2"] = 1
sample_opt["z_4_3_1"] = 1

# 一共用了4种挖掘机：
# sum(z) - t = 3
# 4 - 1 = 3
sample_opt["t_type"] = 1

# 最优解预算正好2400，
# 因此预算 slack = 0
#
# 三种矿车：
# 7, 7, 3
# 也全部用满，因此三个库存 slack 都为0


# ============================================================
# 13. 计算已知最优方案的 QUBO energy
# ============================================================

energy = bqm.energy(sample_opt)

print("\n====== 已知MILP最优方案的QUBO检查 ======")
print("QUBO energy =", energy)
print("理论应为     =", -58508.64 / 1000)

print(
    "误差 =",
    abs(energy - (-58508.64 / 1000))
)


# ============================================================
# 14. 独立检查各约束残差
# ============================================================

selected = [
    cfg for cfg in configs
    if sample_opt[cfg["name"]] == 1
]

budget_used = sum(
    cfg["budget_scaled"]
    for cfg in selected
)

type_count = len(selected)

truck_used = [
    sum(
        cfg["k"]
        for cfg in selected
        if cfg["j"] == j
    )
    for j in range(3)
]

print("\n====== 约束人工检查 ======")

print("预算（20万元单位） =", budget_used, "/ 120")
print("型号数量 =", type_count)
print("矿车使用 =", truck_used, "/ [7, 7, 3]")

print("\n选择的配置：")

for cfg in selected:
    print(
        cfg["name"],
        f"x={cfg['x']}",
        f"u={cfg['u']}",
        f"k={cfg['k']}",
        f"profit={cfg['profit']:.2f}"
    )
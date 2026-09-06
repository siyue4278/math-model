import math
import itertools

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler


# ============================================================
# 2024 MathorCup D - 问题2
# QUBO + SA 多随机种子稳定性检验
# 一次运行50轮，同时统计前30轮和前50轮
# ============================================================

MILP_BENCHMARK = 58508.64

NUM_READS = 5000
NUM_SWEEPS = 30000
MAX_RUNS = 50
CHECKPOINTS = (30, 50)


# -------------------- 原模型数据 --------------------

E = [
    (0, 0),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2),
    (3, 1), (3, 2),
]

m = {
    (0, 0): 1,
    (1, 0): 2, (1, 1): 1,
    (2, 0): 2, (2, 1): 2, (2, 2): 1,
    (3, 1): 2, (3, 2): 1,
}

N = np.array([7, 7, 3], dtype=int)

purchase = np.array([100, 140, 200, 320], dtype=float)

A = np.array([
    3095.04,
    3830.40,
    5473.92,
    5792.64,
])

G_exc = np.array([
    148.0,
    194.0,
    263.0,
    392.0,
])

C_truck = np.array([
    168.96,
    207.84,
    253.44,
])


# -------------------- 48个配置变量 --------------------

configs = []

for i, j in E:
    for k in range(1, N[j] + 1):
        x = math.ceil(k / m[(i, j)])
        u = k / m[(i, j)]

        profit = (
            A[i] * u
            - G_exc[i] * x
            - C_truck[j] * k
        )

        budget = purchase[i] * x

        configs.append({
            "name": f"z_{i+1}_{j+1}_{k}",
            "i": i,
            "j": j,
            "k": k,
            "x": x,
            "u": u,
            "profit": profit,
            "profit_scaled": profit / 1000.0,
            "budget": budget,
            "budget_scaled": int(round(budget / 20)),
        })

assert len(configs) == 48


# ============================================================
# 构造64-bit QUBO
# ============================================================

bqm = dimod.BinaryQuadraticModel("BINARY")

# 目标：max profit <=> min -profit
for cfg in configs:
    bqm.add_linear(
        cfg["name"],
        -cfg["profit_scaled"]
    )

# 已验证的最终罚系数
lambda_B = 1
lambda_T = 1
lambda_M = 30
lambda_truck = [1, 1, 1]


# 预算：sum b_c z_c + s_B = 120
budget_terms = [
    (cfg["name"], cfg["budget_scaled"])
    for cfg in configs
]

for var, weight in {
    "sb_0": 1,
    "sb_1": 2,
    "sb_2": 4,
    "sb_3": 8,
    "sb_4": 16,
    "sb_5": 32,
    "sb_6": 57,
}.items():
    budget_terms.append((var, weight))

bqm.add_linear_equality_constraint(
    budget_terms,
    lagrange_multiplier=lambda_B,
    constant=-120,
)


# 至少3种型号：
# 在每种型号至多一个配置成立时，sum(z)-t=3
type_terms = [
    (cfg["name"], 1)
    for cfg in configs
]
type_terms.append(("t_type", -1))

bqm.add_linear_equality_constraint(
    type_terms,
    lagrange_multiplier=lambda_T,
    constant=-3,
)


# 同型号至多一个配置
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


# 三类矿车库存
truck_slacks = [
    {"s1_0": 1, "s1_1": 2, "s1_2": 4},
    {"s2_0": 1, "s2_1": 2, "s2_2": 4},
    {"s3_0": 1, "s3_1": 2},
]

for j in range(3):
    terms = [
        (cfg["name"], cfg["k"])
        for cfg in configs
        if cfg["j"] == j
    ]

    for var, weight in truck_slacks[j].items():
        terms.append((var, weight))

    bqm.add_linear_equality_constraint(
        terms,
        lagrange_multiplier=lambda_truck[j],
        constant=-int(N[j]),
    )


assert len(bqm.variables) == 64


# ============================================================
# 解码与原问题可行性检查
# ============================================================

def decode(sample):
    selected = [
        cfg
        for cfg in configs
        if sample.get(cfg["name"], 0) == 1
    ]

    profit = sum(
        cfg["profit"]
        for cfg in selected
    )

    budget = sum(
        cfg["budget"]
        for cfg in selected
    )

    truck_used = [
        sum(
            cfg["k"]
            for cfg in selected
            if cfg["j"] == j
        )
        for j in range(3)
    ]

    config_count_by_type = [
        sum(
            1
            for cfg in selected
            if cfg["i"] == i
        )
        for i in range(4)
    ]

    type_count = sum(
        n > 0
        for n in config_count_by_type
    )

    feasible = (
        budget <= 2400 + 1e-8
        and all(
            truck_used[j] <= N[j]
            for j in range(3)
        )
        and all(
            n <= 1
            for n in config_count_by_type
        )
        and type_count >= 3
    )

    return {
        "selected": selected,
        "profit": profit,
        "budget": budget,
        "truck_used": truck_used,
        "type_count": type_count,
        "feasible": feasible,
    }


# ============================================================
# SA：一次跑50轮，同时统计前30轮和前50轮
# ============================================================

sampler = SimulatedAnnealingSampler()

run_records = []

print("====== QUBO规模与参数 ======")
print("总二进制变量数 =", len(bqm.variables))
print("罚系数 =", (lambda_B, lambda_T, lambda_M, *lambda_truck))
print("num_reads =", NUM_READS)
print("num_sweeps =", NUM_SWEEPS)
print("总运行轮数 =", MAX_RUNS)

for run in range(MAX_RUNS):
    kwargs = {
        "num_reads": NUM_READS,
        "num_sweeps": NUM_SWEEPS,
    }

    if "seed" in sampler.parameters:
        kwargs["seed"] = run

    sampleset = sampler.sample(
        bqm,
        **kwargs
    )

    best = sampleset.first
    info = decode(best.sample)

    gap = (
        MILP_BENCHMARK
        - info["profit"]
    )

    run_records.append({
        "run": run + 1,
        "seed": run,
        "energy": float(best.energy),
        "profit": float(info["profit"]),
        "gap": float(gap),
        "feasible": bool(info["feasible"]),
        "sample": best.sample,
        "decoded": info,
    })

    print(
        f"run {run+1:02d}: "
        f"profit={info['profit']:.2f}, "
        f"gap={gap:.2f}, "
        f"feasible={info['feasible']}"
    )


# ============================================================
# 30轮 / 50轮稳定性汇总
# ============================================================

def summarize(n_runs):
    records = run_records[:n_runs]

    profits = np.array(
        [r["profit"] for r in records],
        dtype=float
    )

    feasible_count = sum(
        r["feasible"]
        for r in records
    )

    hit_count = int(
        np.sum(
            np.isclose(
                profits,
                MILP_BENCHMARK,
                atol=1e-5
            )
        )
    )

    best_idx = int(
        np.argmax(profits)
    )

    return {
        "n": n_runs,
        "best": float(np.max(profits)),
        "mean": float(np.mean(profits)),
        "std": float(np.std(profits, ddof=1)) if n_runs > 1 else 0.0,
        "hit_count": hit_count,
        "hit_rate": hit_count / n_runs,
        "feasible_count": feasible_count,
        "feasible_rate": feasible_count / n_runs,
        "best_record": records[best_idx],
    }


summaries = {
    n: summarize(n)
    for n in CHECKPOINTS
}

print("\n================ 稳定性检验汇总 ================")
print(
    f"{'轮数':>6}"
    f"{'Best':>14}"
    f"{'Mean':>14}"
    f"{'Std':>14}"
    f"{'MILP命中':>12}"
    f"{'命中率':>12}"
    f"{'可行率':>12}"
)

for n in CHECKPOINTS:
    s = summaries[n]

    print(
        f"{s['n']:>6d}"
        f"{s['best']:>14.2f}"
        f"{s['mean']:>14.2f}"
        f"{s['std']:>14.2f}"
        f"{s['hit_count']:>8d}/{s['n']:<3d}"
        f"{s['hit_rate']:>11.2%}"
        f"{s['feasible_rate']:>11.2%}"
    )


# ============================================================
# 输出50轮中的全局最好方案
# ============================================================

best_50 = summaries[50]["best_record"]
best_info = best_50["decoded"]

print("\n================ 50轮全局最好方案 ================")
print("run =", best_50["run"])
print("seed =", best_50["seed"])
print("五年净利润 =", best_info["profit"])
print("与MILP利润差 =", MILP_BENCHMARK - best_info["profit"])
print("启动资金 =", best_info["budget"], "/ 2400")
print("矿车使用 =", best_info["truck_used"], "/ [7, 7, 3]")
print("型号数 =", best_info["type_count"])
print("原问题可行 =", best_info["feasible"])

print("\n选择的配置：")

for cfg in best_info["selected"]:
    print(
        cfg["name"],
        f"x={cfg['x']}",
        f"k={cfg['k']}",
        f"u={cfg['u']:.6f}",
        f"profit={cfg['profit']:.2f}"
    )

import math
import itertools
import random

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler


# ============================================================
# 2024 MathorCup D - 问题3
# 动态 pair-subQUBO + SA + Exact旁路诊断 + 约束违规统计
# ============================================================

# -------------------- 数据 --------------------

N = np.array([5, 5, 5, 5, 5, 3, 3, 3, 3, 3], dtype=int)

bucket = np.array([0.9, 1.2, 1.8, 2.1, 2.6, 3.5, 5.0, 6.0, 8.0, 10.0])
efficiency = np.array([190, 175, 165, 150, 140, 130, 120, 110, 105, 100])
fuel_exc = np.array([28, 30, 34, 38, 42, 50, 60, 75, 90, 100])
purchase = np.array([100, 140, 200, 320, 440, 500, 640, 760, 860, 1000])
labor_exc = np.array([7000, 7500, 8500, 9000, 10000, 12000, 13000, 16000, 18000, 20000])
maint_exc = np.array([1000, 1500, 2000, 3000, 5000, 8000, 10000, 13000, 15000, 18000])

fuel_truck = np.array([15, 18, 22, 27, 33, 40, 50, 55, 64, 70])
labor_truck = np.array([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 15000])
maint_truck = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

m = {
    (0, 0): 3, (0, 1): 3, (0, 2): 2,
    (1, 0): 3, (1, 1): 3, (1, 2): 3, (1, 3): 2,
    (2, 0): 4, (2, 1): 3, (2, 2): 3, (2, 3): 3, (2, 4): 2,
    (3, 0): 5, (3, 1): 4, (3, 2): 3, (3, 3): 3, (3, 4): 3, (3, 5): 2,
    (4, 1): 5, (4, 2): 4, (4, 3): 3, (4, 4): 3, (4, 5): 3, (4, 6): 2, (4, 7): 2,
    (5, 2): 5, (5, 3): 4, (5, 4): 3, (5, 5): 3, (5, 6): 3, (5, 7): 2, (5, 8): 2,
    (6, 2): 5, (6, 3): 5, (6, 4): 4, (6, 5): 3, (6, 6): 3, (6, 7): 3, (6, 8): 2, (6, 9): 2,
    (7, 3): 5, (7, 4): 5, (7, 5): 4, (7, 6): 3, (7, 7): 3, (7, 8): 3, (7, 9): 3,
    (8, 4): 5, (8, 5): 5, (8, 6): 4, (8, 7): 3, (8, 8): 3, (8, 9): 3,
    (9, 5): 5, (9, 6): 5, (9, 7): 4, (9, 8): 3, (9, 9): 3,
}
E = list(m.keys())
assert len(E) == 58


# -------------------- 五年经济系数 --------------------

HOURS_5Y = 5 * 12 * 20 * 8

q = bucket * efficiency
R = q * HOURS_5Y * 20 / 10000.0
F_exc = fuel_exc * HOURS_5Y * 7 / 10000.0
A = R - F_exc

G_exc = purchase + 60 * (labor_exc + maint_exc) / 10000.0
C_truck = (
    fuel_truck * HOURS_5Y * 7 / 10000.0
    + 60 * (labor_truck + maint_truck) / 10000.0
)


# -------------------- 配置变量 --------------------

configs = []
configs_by_type = [[] for _ in range(10)]

for i, j in E:
    for k in range(1, N[j] + 1):
        x = math.ceil(k / m[(i, j)])
        u = k / m[(i, j)]
        profit = A[i] * u - G_exc[i] * x - C_truck[j] * k
        budget = purchase[i] * x

        cfg = {
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
        }
        configs.append(cfg)
        configs_by_type[i].append(cfg)

assert len(configs) == 234


# -------------------- 参数 --------------------

MILP_BENCHMARK = 66459.80

LAMBDA_M = 17
LAMBDA_B = 27
LAMBDA_K = 24
LAMBDA_T = 16

PAIR_NUM_READS = 500
PAIR_NUM_SWEEPS = 5000
N_STARTS = 10
MAX_OUTER_SWEEPS = 8
MASTER_SEED = 2024

ALL_PAIRS = list(itertools.combinations(range(10), 2))
sampler = SimulatedAnnealingSampler()


# ============================================================
# 工具函数
# ============================================================

def evaluate_solution(solution):
    profit = 0.0
    budget = 0.0
    truck_used = np.zeros(10, dtype=int)
    type_count = 0

    for cfg in solution:
        if cfg is None:
            continue
        type_count += 1
        profit += cfg["profit"]
        budget += cfg["budget"]
        truck_used[cfg["j"]] += cfg["k"]

    return {
        "profit": profit,
        "budget": budget,
        "truck_used": truck_used,
        "type_count": type_count,
        "feasible": (
            budget <= 4000 + 1e-8
            and np.all(truck_used <= N)
            and type_count >= 5
        ),
    }


def generate_initial_solution(rng):
    for _ in range(1000):
        solution = [None] * 10
        budget_used = 0.0
        truck_used = np.zeros(10, dtype=int)

        for i in rng.sample(range(10), 5):
            candidates = [
                cfg
                for cfg in configs_by_type[i]
                if (
                    cfg["k"] == 1
                    and budget_used + cfg["budget"] <= 4000 + 1e-8
                    and truck_used[cfg["j"]] + cfg["k"] <= N[cfg["j"]]
                )
            ]
            if not candidates:
                break

            candidates.sort(key=lambda cfg: cfg["profit"], reverse=True)
            cfg = rng.choice(candidates[:max(1, (len(candidates) + 1) // 2)])

            solution[i] = cfg
            budget_used += cfg["budget"]
            truck_used[cfg["j"]] += cfg["k"]
        else:
            if evaluate_solution(solution)["feasible"]:
                return solution

    raise RuntimeError("无法生成可行初始解。")


def slack_weights(capacity):
    capacity = int(capacity)
    if capacity <= 0:
        return []

    weights = []
    power = 1

    while sum(weights) + power <= capacity:
        weights.append(power)
        power *= 2

    remainder = capacity - sum(weights)
    if remainder > 0:
        weights.append(remainder)

    return weights


# ============================================================
# Pair-subQUBO
# ============================================================

def build_pair_bqm(current_solution, i1, i2):
    active = {i1, i2}

    fixed_budget = 0.0
    fixed_truck = np.zeros(10, dtype=int)
    fixed_type_count = 0

    for i, cfg in enumerate(current_solution):
        if i in active or cfg is None:
            continue
        fixed_type_count += 1
        fixed_budget += cfg["budget"]
        fixed_truck[cfg["j"]] += cfg["k"]

    residual_budget = 4000.0 - fixed_budget
    residual_budget_scaled = int(round(residual_budget / 20.0))
    residual_truck = (N - fixed_truck).astype(int)

    if residual_budget_scaled < 0 or np.any(residual_truck < 0):
        raise RuntimeError("固定部分已经不可行。")

    bqm = dimod.BinaryQuadraticModel("BINARY")
    active_configs = configs_by_type[i1] + configs_by_type[i2]

    for cfg in active_configs:
        bqm.add_linear(cfg["name"], -cfg["profit_scaled"])

    active_flags = {}

    # 每个 active 型号：不选或恰好选一个配置
    for i in (i1, i2):
        a_name = f"a_sub_{i+1}"
        active_flags[i] = a_name

        terms = [(cfg["name"], 1) for cfg in configs_by_type[i]]
        terms.append((a_name, -1))

        bqm.add_linear_equality_constraint(
            terms,
            lagrange_multiplier=LAMBDA_M,
            constant=0,
        )

    # 至少5种型号
    required_active_types = max(0, 5 - fixed_type_count)

    if required_active_types > 2:
        raise RuntimeError("当前 pair 固定部分无法满足至少5种型号约束。")

    if required_active_types == 1:
        bqm.add_linear_equality_constraint(
            [
                (active_flags[i1], 1),
                (active_flags[i2], 1),
                ("t_sub_type", -1),
            ],
            lagrange_multiplier=LAMBDA_T,
            constant=-1,
        )
    elif required_active_types == 2:
        bqm.add_linear_equality_constraint(
            [
                (active_flags[i1], 1),
                (active_flags[i2], 1),
            ],
            lagrange_multiplier=LAMBDA_T,
            constant=-2,
        )

    # 预算
    budget_terms = [
        (cfg["name"], cfg["budget_scaled"])
        for cfg in active_configs
    ]

    for bit_id, weight in enumerate(slack_weights(residual_budget_scaled)):
        budget_terms.append((f"sb_sub_{bit_id}", weight))

    bqm.add_linear_equality_constraint(
        budget_terms,
        lagrange_multiplier=LAMBDA_B,
        constant=-residual_budget_scaled,
    )

    # 矿车库存
    active_truck_types = sorted({cfg["j"] for cfg in active_configs})

    for j in active_truck_types:
        truck_terms = [
            (cfg["name"], cfg["k"])
            for cfg in active_configs
            if cfg["j"] == j
        ]

        cap = int(residual_truck[j])

        for bit_id, weight in enumerate(slack_weights(cap)):
            truck_terms.append((f"st_sub_{j+1}_{bit_id}", weight))

        bqm.add_linear_equality_constraint(
            truck_terms,
            lagrange_multiplier=LAMBDA_K,
            constant=-cap,
        )

    bit_count = len(bqm.variables)
    if bit_count >= 100:
        raise RuntimeError(
            f"pair ({i1+1},{i2+1}) subQUBO变量数={bit_count}，未满足 <100。"
        )

    return bqm, bit_count


def decode_pair_sample(sample, current_solution, i1, i2):
    candidate = current_solution.copy()

    for i in (i1, i2):
        selected = [
            cfg
            for cfg in configs_by_type[i]
            if sample.get(cfg["name"], 0) == 1
        ]

        if len(selected) > 1:
            return None

        candidate[i] = selected[0] if selected else None

    return candidate



def solve_pair_by_sa(current_solution, i1, i2, seed):
    bqm, bit_count = build_pair_bqm(current_solution, i1, i2)

    kwargs = {
        "num_reads": PAIR_NUM_READS,
        "num_sweeps": PAIR_NUM_SWEEPS,
    }
    if "seed" in sampler.parameters:
        kwargs["seed"] = seed

    sampleset = sampler.sample(bqm, **kwargs)

    best_solution = current_solution.copy()
    best_profit = evaluate_solution(current_solution)["profit"]

    for datum in sampleset.data(fields=["sample"]):
        candidate = decode_pair_sample(datum.sample, current_solution, i1, i2)
        if candidate is None:
            continue

        info = evaluate_solution(candidate)
        if info["feasible"] and info["profit"] > best_profit + 1e-9:
            best_solution = candidate
            best_profit = info["profit"]

    return best_solution, best_profit, bit_count


# ============================================================
# Exact pair 旁路诊断
# ============================================================

def solve_pair_exact(current_solution, i1, i2):
    fixed = current_solution.copy()
    fixed[i1] = None
    fixed[i2] = None

    best_solution = current_solution.copy()
    best_profit = evaluate_solution(current_solution)["profit"]

    choices_1 = [None] + configs_by_type[i1]
    choices_2 = [None] + configs_by_type[i2]

    for cfg1, cfg2 in itertools.product(choices_1, choices_2):
        candidate = fixed.copy()
        candidate[i1] = cfg1
        candidate[i2] = cfg2

        info = evaluate_solution(candidate)
        if info["feasible"] and info["profit"] > best_profit + 1e-9:
            best_solution = candidate
            best_profit = info["profit"]

    return best_solution, best_profit


# ============================================================
# 规模检查
# ============================================================

def pair_bit_upper_bound(i1, i2):
    z_bits = len(configs_by_type[i1]) + len(configs_by_type[i2])
    budget_bits = len(slack_weights(200))

    truck_types = {
        cfg["j"]
        for cfg in configs_by_type[i1] + configs_by_type[i2]
    }
    truck_bits = sum(len(slack_weights(N[j])) for j in truck_types)

    return z_bits + 2 + 1 + budget_bits + truck_bits


max_bits, max_pair = max(
    (pair_bit_upper_bound(i1, i2), (i1 + 1, i2 + 1))
    for i1, i2 in ALL_PAIRS
)

assert max_bits < 100


# ============================================================
# 动态 pair 搜索
# ============================================================

def pair_subqubo_descent(initial_solution, rng, start_id):
    current_solution = initial_solution.copy()
    current_profit = evaluate_solution(current_solution)["profit"]

    diag = {
        "pair_calls": 0,
        "exact_improvable": 0,
        "sa_found_improvement": 0,
        "sa_missed_improvement": 0,
        "sa_hit_exact_improvable": 0,
        "sum_pair_gap": 0.0,
        "sum_pair_gap_improvable": 0.0,
    }

    accepted_moves = 0
    max_bits_seen = 0
    sweeps_used = 0
    moves_per_sweep = []
    stop_reason = None

    for outer in range(MAX_OUTER_SWEEPS):
        pairs = ALL_PAIRS.copy()
        rng.shuffle(pairs)
        improved_this_sweep = 0

        for pair_id, (i1, i2) in enumerate(pairs):
            current_before = current_profit
            _, exact_profit = solve_pair_exact(current_solution, i1, i2)

            seed = start_id * 100000 + outer * 1000 + pair_id
            candidate, sa_profit, bit_count = solve_pair_by_sa(
                current_solution, i1, i2, seed
            )
            max_bits_seen = max(max_bits_seen, bit_count)

            exact_improvement = max(0.0, exact_profit - current_before)
            sa_improvement = max(0.0, sa_profit - current_before)
            pair_gap = max(0.0, exact_profit - sa_profit)

            diag["pair_calls"] += 1
            diag["sum_pair_gap"] += pair_gap

            if exact_improvement > 1e-8:
                diag["exact_improvable"] += 1
                diag["sum_pair_gap_improvable"] += pair_gap
                if sa_improvement > 1e-8:
                    diag["sa_found_improvement"] += 1
                else:
                    diag["sa_missed_improvement"] += 1
                if abs(exact_profit - sa_profit) <= 1e-6:
                    diag["sa_hit_exact_improvable"] += 1

            if sa_profit > current_profit + 1e-8:
                current_solution = candidate
                current_profit = sa_profit
                accepted_moves += 1
                improved_this_sweep += 1

        sweeps_used = outer + 1
        moves_per_sweep.append(improved_this_sweep)

        if improved_this_sweep == 0:
            stop_reason = "自然收敛"
            break

    if stop_reason is None:
        stop_reason = "达到outer上限"

    return {
        "solution": current_solution,
        "profit": current_profit,
        "moves": accepted_moves,
        "max_bits": max_bits_seen,
        "sweeps_used": sweeps_used,
        "moves_per_sweep": moves_per_sweep,
        "stop_reason": stop_reason,
        "diag": diag,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    print("完整配置变量数 =", len(configs))
    print("各型号配置数 =", [len(x) for x in configs_by_type])
    print("45个pair保守最大bit数 =", max_bits, "对应pair =", max_pair)
    print("罚权重 =", (LAMBDA_M, LAMBDA_B, LAMBDA_K, LAMBDA_T))
    print("reads / sweeps / starts / outer上限 =", PAIR_NUM_READS, PAIR_NUM_SWEEPS, N_STARTS, MAX_OUTER_SWEEPS)

    master_rng = random.Random(MASTER_SEED)
    best_solution = None
    best_profit = -np.inf
    final_profits = []

    total = {
        "pair_calls": 0,
        "exact_improvable": 0,
        "sa_found_improvement": 0,
        "sa_missed_improvement": 0,
        "sa_hit_exact_improvable": 0,
        "sum_pair_gap": 0.0,
        "sum_pair_gap_improvable": 0.0,
    }

    natural_stop_count = 0
    limit_stop_count = 0

    for start_id in range(1, N_STARTS + 1):
        rng = random.Random(master_rng.randrange(10**9))
        initial_solution = generate_initial_solution(rng)
        initial_profit = evaluate_solution(initial_solution)["profit"]

        result = pair_subqubo_descent(initial_solution, rng, start_id)
        final_profit = result["profit"]
        final_profits.append(final_profit)

        if final_profit > best_profit + 1e-8:
            best_profit = final_profit
            best_solution = result["solution"].copy()

        diag = result["diag"]
        for key in (
            "pair_calls", "exact_improvable", "sa_found_improvement",
            "sa_missed_improvement", "sa_hit_exact_improvable"
        ):
            total[key] += diag[key]
        total["sum_pair_gap"] += diag["sum_pair_gap"]
        total["sum_pair_gap_improvable"] += diag["sum_pair_gap_improvable"]

        if result["stop_reason"] == "自然收敛":
            natural_stop_count += 1
        else:
            limit_stop_count += 1

        print(
            f"start {start_id:02d}: initial={initial_profit:.2f}, "
            f"final={final_profit:.2f}, gap={MILP_BENCHMARK-final_profit:.2f}, "
            f"moves={result['moves']}, max_bits={result['max_bits']}"
        )
        print(
            f"  outer轮数={result['sweeps_used']}, "
            f"每轮改善数={result['moves_per_sweep']}, "
            f"停止原因={result['stop_reason']}"
        )

    final_profits = np.array(final_profits, dtype=float)

    print("\n================ 实验结果 ================")
    print("最好利润 =", best_profit)
    print("MILP基准 =", MILP_BENCHMARK)
    print("最好方案绝对差 =", MILP_BENCHMARK - best_profit)
    print("最好方案相对gap =", (MILP_BENCHMARK - best_profit) / MILP_BENCHMARK * 100, "%")
    print("命中MILP最优次数 =", np.sum(np.isclose(final_profits, MILP_BENCHMARK, atol=1e-5)), "/", N_STARTS)
    print("最终利润平均值 =", np.mean(final_profits))
    if len(final_profits) > 1:
        print("最终利润标准差 =", np.std(final_profits, ddof=1))

    print("\n================ Outer sweep诊断 ================")
    print("自然收敛start数 =", natural_stop_count, "/", N_STARTS)
    print("达到outer上限start数 =", limit_stop_count, "/", N_STARTS)

    exact_n = total["exact_improvable"]
    pair_n = total["pair_calls"]

    print("\n================ SA-vs-Exact Pair诊断 ================")
    print("总pair调用次数 =", pair_n)
    print("Exact证明可改善的pair次数 =", exact_n)
    print("其中SA也发现改善的次数 =", total["sa_found_improvement"])
    print("SA完全遗漏改善机会次数 =", total["sa_missed_improvement"])
    print("SA命中Exact次数（仅Exact可改善pair） =", total["sa_hit_exact_improvable"])
    print("SA对可改善pair的改善发现率 =", f"{total['sa_found_improvement']/exact_n:.2%}" if exact_n else "0.00%")
    print("SA对可改善pair的Exact命中率 =", f"{total['sa_hit_exact_improvable']/exact_n:.2%}" if exact_n else "0.00%")
    print("全部pair平均 Exact-SA gap =", total["sum_pair_gap"] / pair_n if pair_n else 0.0)
    print("仅Exact可改善pair的平均 Exact-SA gap =", total["sum_pair_gap_improvable"] / exact_n if exact_n else 0.0)

    best_info = evaluate_solution(best_solution)

    print("\n================ 最好方案 ================")
    print("五年净利润 =", best_info["profit"])
    print("启动资金 =", best_info["budget"], "/ 4000")
    print("矿车使用 =", best_info["truck_used"].tolist())
    print("矿车库存 =", N.tolist())
    print("型号数 =", best_info["type_count"])
    print("原问题可行 =", best_info["feasible"])

    print("\n选择的配置：")
    for i, cfg in enumerate(best_solution):
        if cfg is None:
            continue
        print(
            f"挖{i+1}: {cfg['name']}, 矿{cfg['j']+1}, "
            f"x={cfg['x']}, k={cfg['k']}, u={cfg['u']:.6f}, "
            f"profit={cfg['profit']:.2f}"
        )


if __name__ == "__main__":
    main()

import math
import itertools
import random

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler


# ============================================================
# 2024 MathorCup D 题 —— 问题3
# 动态 pair-subQUBO + SA
#
# 核心原则：
# 1. 完整原问题仍然是 234 个配置变量；
# 2. 每次只释放两个挖掘机型号，其余 8 个型号固定；
# 3. 当前 pair 的剩余预算、剩余矿车库存由固定部分决定；
# 4. 当前 pair 建成 <100 bit 的 subQUBO；
# 5. subQUBO 用 SA 求解；
# 6. 只接受使原问题利润严格提高的可行新方案；
# 7. 45 个 pair 动态重叠扫描，不采用永久固定分块。
#
# 这版先验证“动态 subQUBO + SA”能否逼近/命中
# 已知 MILP benchmark = 66459.80 万元。
# ============================================================


# ============================================================
# 1. 问题3数据
# ============================================================

N = np.array(
    [5, 5, 5, 5, 5, 3, 3, 3, 3, 3],
    dtype=int
)

bucket = np.array(
    [0.9, 1.2, 1.8, 2.1, 2.6, 3.5, 5.0, 6.0, 8.0, 10.0],
    dtype=float
)

efficiency = np.array(
    [190, 175, 165, 150, 140, 130, 120, 110, 105, 100],
    dtype=float
)

fuel_exc = np.array(
    [28, 30, 34, 38, 42, 50, 60, 75, 90, 100],
    dtype=float
)

purchase = np.array(
    [100, 140, 200, 320, 440, 500, 640, 760, 860, 1000],
    dtype=float
)

labor_exc = np.array(
    [7000, 7500, 8500, 9000, 10000,
     12000, 13000, 16000, 18000, 20000],
    dtype=float
)

maint_exc = np.array(
    [1000, 1500, 2000, 3000, 5000,
     8000, 10000, 13000, 15000, 18000],
    dtype=float
)

fuel_truck = np.array(
    [15, 18, 22, 27, 33, 40, 50, 55, 64, 70],
    dtype=float
)

labor_truck = np.array(
    [5000, 6000, 7000, 8000, 9000,
     10000, 11000, 12000, 13000, 15000],
    dtype=float
)

maint_truck = np.array(
    [1000, 2000, 3000, 4000, 5000,
     6000, 7000, 8000, 9000, 10000],
    dtype=float
)


# ============================================================
# 2. 表7匹配关系 m_ij
# ============================================================

m = {
    # 挖1
    (0, 0): 3, (0, 1): 3, (0, 2): 2,

    # 挖2
    (1, 0): 3, (1, 1): 3, (1, 2): 3, (1, 3): 2,

    # 挖3
    (2, 0): 4, (2, 1): 3, (2, 2): 3,
    (2, 3): 3, (2, 4): 2,

    # 挖4
    (3, 0): 5, (3, 1): 4, (3, 2): 3,
    (3, 3): 3, (3, 4): 3, (3, 5): 2,

    # 挖5
    (4, 1): 5, (4, 2): 4, (4, 3): 3,
    (4, 4): 3, (4, 5): 3, (4, 6): 2, (4, 7): 2,

    # 挖6
    (5, 2): 5, (5, 3): 4, (5, 4): 3,
    (5, 5): 3, (5, 6): 3, (5, 7): 2, (5, 8): 2,

    # 挖7
    (6, 2): 5, (6, 3): 5, (6, 4): 4,
    (6, 5): 3, (6, 6): 3, (6, 7): 3,
    (6, 8): 2, (6, 9): 2,

    # 挖8
    (7, 3): 5, (7, 4): 5, (7, 5): 4,
    (7, 6): 3, (7, 7): 3, (7, 8): 3, (7, 9): 3,

    # 挖9
    (8, 4): 5, (8, 5): 5, (8, 6): 4,
    (8, 7): 3, (8, 8): 3, (8, 9): 3,

    # 挖10
    (9, 5): 5, (9, 6): 5, (9, 7): 4,
    (9, 8): 3, (9, 9): 3,
}

E = list(m.keys())

assert len(E) == 58


# ============================================================
# 3. 五年经济系数
# ============================================================

HOURS_5Y = 5 * 12 * 20 * 8

q = bucket * efficiency

R = (
    q * HOURS_5Y * 20 / 10000.0
)

F_exc = (
    fuel_exc * HOURS_5Y * 7 / 10000.0
)

A = R - F_exc

G_exc = (
    purchase
    + 60 * (labor_exc + maint_exc) / 10000.0
)

C_truck = (
    fuel_truck * HOURS_5Y * 7 / 10000.0
    + 60 * (labor_truck + maint_truck) / 10000.0
)


# ============================================================
# 4. 生成完整 234 个配置变量
#
# z_ijk = 1：
# 第 i 型挖掘机匹配第 j 型矿车，
# 实际投入 k 辆矿车。
#
# 配置内部恢复：
# x_i = ceil(k/m_ij)
# u_ij = k/m_ij
# ============================================================

configs = []

configs_by_type = [
    []
    for _ in range(10)
]

for i, j in E:

    for k in range(
        1,
        N[j] + 1
    ):

        x = math.ceil(
            k / m[(i, j)]
        )

        u = (
            k / m[(i, j)]
        )

        profit = (
            A[i] * u
            - G_exc[i] * x
            - C_truck[j] * k
        )

        budget = (
            purchase[i] * x
        )

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
            "budget_scaled": int(
                round(budget / 20)
            )
        }

        configs.append(
            cfg
        )

        configs_by_type[i].append(
            cfg
        )


print(
    "完整配置变量数 =",
    len(configs)
)

print(
    "各型号配置数 =",
    [
        len(group)
        for group in configs_by_type
    ]
)

assert len(configs) == 234


# ============================================================
# 5. 评价一个完整方案
#
# solution[i]:
# None -> 不选第 i 型挖掘机
# cfg  -> 选择该型号的一个配置
# ============================================================

def evaluate_solution(
    solution
):

    profit = 0.0
    budget = 0.0

    truck_used = np.zeros(
        10,
        dtype=int
    )

    type_count = 0

    for cfg in solution:

        if cfg is None:
            continue

        type_count += 1

        profit += (
            cfg["profit"]
        )

        budget += (
            cfg["budget"]
        )

        truck_used[
            cfg["j"]
        ] += cfg["k"]

    feasible = (
        budget <= 4000 + 1e-8
        and np.all(
            truck_used <= N
        )
        and type_count >= 5
    )

    return {
        "profit": profit,
        "budget": budget,
        "truck_used": truck_used,
        "type_count": type_count,
        "feasible": feasible
    }


# ============================================================
# 6. 随机生成可行初始解
#
# 不使用 MILP 最优方案。
# ============================================================

def generate_initial_solution(
    rng
):

    for _ in range(1000):

        solution = [
            None
            for _ in range(10)
        ]

        budget_used = 0.0

        truck_used = np.zeros(
            10,
            dtype=int
        )

        # 初始时先随机选 5 个型号，保证型号数约束
        selected_types = rng.sample(
            range(10),
            5
        )

        success = True

        for i in selected_types:

            # 初始解先用 k=1 的轻量配置，
            # 避免一开始耗尽预算和矿车库存。
            candidates = [
                cfg
                for cfg in configs_by_type[i]
                if (
                    cfg["k"] == 1
                    and budget_used
                    + cfg["budget"]
                    <= 4000 + 1e-8
                    and truck_used[
                        cfg["j"]
                    ] + cfg["k"]
                    <= N[
                        cfg["j"]
                    ]
                )
            ]

            if not candidates:

                success = False
                break

            candidates.sort(
                key=lambda cfg: cfg["profit"],
                reverse=True
            )

            # 从较好的一半中随机选，
            # 兼顾质量和 multi-start 多样性。
            top_n = max(
                1,
                (len(candidates) + 1) // 2
            )

            cfg = rng.choice(
                candidates[:top_n]
            )

            solution[i] = cfg

            budget_used += (
                cfg["budget"]
            )

            truck_used[
                cfg["j"]
            ] += cfg["k"]

        if success:

            info = evaluate_solution(
                solution
            )

            if info["feasible"]:
                return solution

    raise RuntimeError(
        "无法生成可行初始解。"
    )


# ============================================================
# 7. 生成 0~capacity 的二进制 slack 权重
#
# 例如：
# capacity=5 -> [1,2,2]
# capacity=3 -> [1,2]
# capacity=200 -> [1,2,4,8,16,32,64,73]
#
# 这样权重和恰好等于 capacity，
# 可以表示 0~capacity 的每个整数。
# ============================================================

def slack_weights(
    capacity
):

    capacity = int(
        capacity
    )

    if capacity <= 0:
        return []

    weights = []

    power = 1

    while (
        sum(weights) + power
        <= capacity
    ):

        weights.append(
            power
        )

        power *= 2

    remainder = (
        capacity
        - sum(weights)
    )

    if remainder > 0:
        weights.append(
            remainder
        )

    return weights


# ============================================================
# 8. subQUBO 罚权重
#
# 不拍脑袋固定一个很大的数字：
#
# 先计算整个问题中“单个配置的最大缩放利润”，
# 再自动取一个明显更大的整数。
#
# 当前数据下最大配置利润 < 16（已除以1000），
# 因此下面自动得到的罚权重会约为 17。
#
# 它主要用于：
# - active 型号配置与 a_i 的链接；
# - 剩余预算；
# - 剩余矿车库存；
# - 至少5种型号。
#
# 最终候选解还会按原问题约束重新检查，
# 不会仅凭 QUBO penalty 判断可行。
# ============================================================

MAX_SINGLE_CONFIG_PROFIT_SCALED = max(
    cfg["profit_scaled"]
    for cfg in configs
)

SUB_PENALTY = (
    math.ceil(
        MAX_SINGLE_CONFIG_PROFIT_SCALED
    )
    + 1
)

print(
    "\n单个配置最大缩放利润 =",
    MAX_SINGLE_CONFIG_PROFIT_SCALED
)

print(
    "subQUBO统一罚权重 =",
    SUB_PENALTY
)


# ============================================================
# 9. 建立某一个动态 pair 的 subQUBO
#
# active pair = (i1, i2)
#
# 其他 8 个型号全部固定。
#
# 为每个 active 型号增加一个 a_i：
#
# sum_c z_ic = a_i
#
# 这样零罚状态天然保证：
# 每种 active 型号“不选或恰好选一个配置”。
# ============================================================

def build_pair_bqm(
    current_solution,
    i1,
    i2
):

    active = (
        i1,
        i2
    )

    # --------------------------------------------------------
    # 9.1 统计固定部分占用的资源
    # --------------------------------------------------------

    fixed_budget = 0.0

    fixed_truck = np.zeros(
        10,
        dtype=int
    )

    fixed_type_count = 0

    for i, cfg in enumerate(
        current_solution
    ):

        if i in active:
            continue

        if cfg is None:
            continue

        fixed_type_count += 1

        fixed_budget += (
            cfg["budget"]
        )

        fixed_truck[
            cfg["j"]
        ] += cfg["k"]

    residual_budget = (
        4000.0
        - fixed_budget
    )

    residual_budget_scaled = int(
        round(
            residual_budget / 20.0
        )
    )

    residual_truck = (
        N - fixed_truck
    ).astype(int)

    if (
        residual_budget_scaled < 0
        or np.any(
            residual_truck < 0
        )
    ):

        raise RuntimeError(
            "固定部分已经不可行，不能建立 pair-subQUBO。"
        )

    # --------------------------------------------------------
    # 9.2 建立 BQM
    # --------------------------------------------------------

    bqm = dimod.BinaryQuadraticModel(
        "BINARY"
    )

    active_configs = (
        configs_by_type[i1]
        + configs_by_type[i2]
    )

    # 子问题只需要优化 active 部分利润，
    # 固定部分利润是常数，不写入 QUBO。
    for cfg in active_configs:

        bqm.add_linear(
            cfg["name"],
            -cfg["profit_scaled"]
        )

    # --------------------------------------------------------
    # 9.3 每个 active 型号：
    #
    # sum z_ic = a_i
    #
    # a_i=0 -> 不选该型号
    # a_i=1 -> 恰好选一个配置
    # --------------------------------------------------------

    active_flags = {}

    for i in active:

        a_name = (
            f"a_sub_{i+1}"
        )

        active_flags[i] = (
            a_name
        )

        terms = [
            (
                cfg["name"],
                1
            )
            for cfg in configs_by_type[i]
        ]

        terms.append(
            (
                a_name,
                -1
            )
        )

        bqm.add_linear_equality_constraint(
            terms,
            lagrange_multiplier=SUB_PENALTY,
            constant=0
        )

    # --------------------------------------------------------
    # 9.4 全局至少 5 种型号
    #
    # fixed_type_count + a_i1 + a_i2 >= 5
    #
    # 因当前完整解始终可行，
    # fixed_type_count 至少为 3。
    #
    # 所以 active pair 至少需要 L=0/1/2 个型号。
    # --------------------------------------------------------

    required_active_types = max(
        0,
        5 - fixed_type_count
    )

    if required_active_types > 2:

        raise RuntimeError(
            "当前 pair 固定部分无法满足至少5种型号约束。"
        )

    if required_active_types == 1:

        # a1 + a2 = 1 或 2
        #
        # 用 t∈{0,1}：
        # a1 + a2 - t = 1
        terms = [
            (
                active_flags[i1],
                1
            ),
            (
                active_flags[i2],
                1
            ),
            (
                "t_sub_type",
                -1
            )
        ]

        bqm.add_linear_equality_constraint(
            terms,
            lagrange_multiplier=SUB_PENALTY,
            constant=-1
        )

    elif required_active_types == 2:

        # a1 + a2 = 2
        terms = [
            (
                active_flags[i1],
                1
            ),
            (
                active_flags[i2],
                1
            )
        ]

        bqm.add_linear_equality_constraint(
            terms,
            lagrange_multiplier=SUB_PENALTY,
            constant=-2
        )

    # L=0 时不需要额外型号约束。

    # --------------------------------------------------------
    # 9.5 剩余预算
    #
    # sum B_c z_c <= residual_budget
    #
    # 按20万元缩放后加入 slack。
    # --------------------------------------------------------

    budget_terms = [
        (
            cfg["name"],
            cfg["budget_scaled"]
        )
        for cfg in active_configs
    ]

    budget_weights = slack_weights(
        residual_budget_scaled
    )

    for bit_id, weight in enumerate(
        budget_weights
    ):

        budget_terms.append(
            (
                f"sb_sub_{bit_id}",
                weight
            )
        )

    bqm.add_linear_equality_constraint(
        budget_terms,
        lagrange_multiplier=SUB_PENALTY,
        constant=-residual_budget_scaled
    )

    # --------------------------------------------------------
    # 9.6 剩余矿车库存
    #
    # 对 active pair 实际会涉及的每一种矿车 j：
    #
    # sum k z_ijk <= residual_truck[j]
    #
    # 再加入对应 slack。
    # --------------------------------------------------------

    active_truck_types = sorted(
        set(
            cfg["j"]
            for cfg in active_configs
        )
    )

    for j in active_truck_types:

        truck_terms = [
            (
                cfg["name"],
                cfg["k"]
            )
            for cfg in active_configs
            if cfg["j"] == j
        ]

        cap = int(
            residual_truck[j]
        )

        truck_weights = slack_weights(
            cap
        )

        for bit_id, weight in enumerate(
            truck_weights
        ):

            truck_terms.append(
                (
                    f"st_sub_{j+1}_{bit_id}",
                    weight
                )
            )

        bqm.add_linear_equality_constraint(
            truck_terms,
            lagrange_multiplier=SUB_PENALTY,
            constant=-cap
        )

    # --------------------------------------------------------
    # 9.7 bit 数检查
    # --------------------------------------------------------

    bit_count = len(
        bqm.variables
    )

    if bit_count >= 100:

        raise RuntimeError(
            f"pair ({i1+1},{i2+1}) "
            f"subQUBO变量数={bit_count}，未满足 <100。"
        )

    meta = {
        "active": active,
        "fixed_type_count": fixed_type_count,
        "required_active_types": required_active_types,
        "residual_budget": residual_budget,
        "residual_budget_scaled": residual_budget_scaled,
        "residual_truck": residual_truck,
        "bit_count": bit_count
    }

    return (
        bqm,
        meta
    )


# ============================================================
# 10. 从一个 subQUBO sample 恢复完整方案
#
# 只根据两个 active 型号的 z 变量恢复；
# a/slack/t 都只是辅助变量。
#
# 若某型号出现 >1 个 z=1，
# 则该 sample 直接视为原问题非法。
# ============================================================

def decode_pair_sample(
    sample,
    current_solution,
    i1,
    i2
):

    candidate = (
        current_solution.copy()
    )

    for i in (
        i1,
        i2
    ):

        selected = [
            cfg
            for cfg in configs_by_type[i]
            if sample.get(
                cfg["name"],
                0
            ) == 1
        ]

        if len(
            selected
        ) > 1:

            return None

        if len(
            selected
        ) == 0:

            candidate[i] = None

        else:

            candidate[i] = (
                selected[0]
            )

    return candidate


# ============================================================
# 11. 用 SA 求一个 pair-subQUBO
#
# 不是只取 sampleset.first：
# 检查全部 samples，
# 从“原问题可行”的候选中取利润最高者。
# ============================================================

sampler = (
    SimulatedAnnealingSampler()
)

# 第一版只做中等规模验证。
# 如果能跑通，再统一增加 reads / sweeps / multi-start。
PAIR_NUM_READS = 100
PAIR_NUM_SWEEPS = 2000


def solve_pair_by_sa(
    current_solution,
    i1,
    i2,
    seed
):

    bqm, meta = build_pair_bqm(
        current_solution,
        i1,
        i2
    )

    kwargs = {
        "num_reads": PAIR_NUM_READS,
        "num_sweeps": PAIR_NUM_SWEEPS
    }

    if (
        "seed"
        in sampler.parameters
    ):

        kwargs["seed"] = seed

    sampleset = sampler.sample(
        bqm,
        **kwargs
    )

    # 当前方案永远作为 fallback，
    # 因而 subQUBO 求解失败也不会把全局解变坏。
    best_solution = (
        current_solution.copy()
    )

    best_info = evaluate_solution(
        best_solution
    )

    best_profit = (
        best_info["profit"]
    )

    feasible_samples = 0

    for datum in sampleset.data(
        fields=[
            "sample",
            "energy",
            "num_occurrences"
        ]
    ):

        sample = (
            datum.sample
        )

        candidate = (
            decode_pair_sample(
                sample,
                current_solution,
                i1,
                i2
            )
        )

        if candidate is None:
            continue

        info = evaluate_solution(
            candidate
        )

        if not info["feasible"]:
            continue

        feasible_samples += (
            datum.num_occurrences
        )

        if (
            info["profit"]
            > best_profit + 1e-9
        ):

            best_profit = (
                info["profit"]
            )

            best_solution = (
                candidate
            )

    return (
        best_solution,
        best_profit,
        meta,
        feasible_samples
    )


# ============================================================
# 12. 检查所有 45 个 pair 的理论最大 bit 数
#
# 用一个“空固定占用”的上界状态计算。
# 实际运行时剩余资源变小，
# slack bit 通常只会更少。
# ============================================================

empty_solution = [
    None
    for _ in range(10)
]

pair_bit_table = []

for i1, i2 in itertools.combinations(
    range(10),
    2
):

    # empty_solution 本身不足5种型号，
    # build_pair_bqm 不适合直接用于 bit 上界。
    #
    # 因此这里只按变量构成直接计算一个保守上界：
    #
    # active z
    # + 2 个 a
    # + 最多 1 个 type辅助
    # + 4000万元预算 slack
    # + active pair 涉及矿车的最大库存 slack

    z_bits = (
        len(
            configs_by_type[i1]
        )
        + len(
            configs_by_type[i2]
        )
    )

    a_bits = 2

    type_aux_bits = 1

    budget_bits = len(
        slack_weights(
            200
        )
    )

    truck_types = sorted(
        set(
            cfg["j"]
            for cfg in (
                configs_by_type[i1]
                + configs_by_type[i2]
            )
        )
    )

    truck_slack_bits = sum(
        len(
            slack_weights(
                N[j]
            )
        )
        for j in truck_types
    )

    upper_bits = (
        z_bits
        + a_bits
        + type_aux_bits
        + budget_bits
        + truck_slack_bits
    )

    pair_bit_table.append(
        (
            upper_bits,
            i1,
            i2
        )
    )


max_pair_bits, max_i1, max_i2 = max(
    pair_bit_table,
    key=lambda x: x[0]
)

print(
    "\n45个pair的保守最大bit数 =",
    max_pair_bits
)

print(
    "对应pair =",
    (
        max_i1 + 1,
        max_i2 + 1
    )
)

assert max_pair_bits < 100


# ============================================================
# 13. 45 个动态 pair
# ============================================================

ALL_PAIRS = list(
    itertools.combinations(
        range(10),
        2
    )
)

assert len(
    ALL_PAIRS
) == 45


# ============================================================
# 14. 动态 pair-subQUBO 邻域搜索
#
# 每轮随机打乱 45 个 pair。
#
# 若 SA 在某个 pair 中找到更优可行方案：
# 立即更新当前全局方案。
#
# 一整轮 45 个 pair 都没有改善：
# 停止。
# ============================================================

def pair_subqubo_descent(
    initial_solution,
    rng,
    start_id,
    max_outer_sweeps=4
):

    current_solution = (
        initial_solution.copy()
    )

    current_profit = (
        evaluate_solution(
            current_solution
        )["profit"]
    )

    accepted_moves = 0

    max_bits_seen = 0

    total_feasible_samples = 0

    for outer in range(
        max_outer_sweeps
    ):

        pairs = (
            ALL_PAIRS.copy()
        )

        rng.shuffle(
            pairs
        )

        improved_this_sweep = 0

        for pair_id, (
            i1,
            i2
        ) in enumerate(
            pairs
        ):

            seed = (
                start_id * 100000
                + outer * 1000
                + pair_id
            )

            (
                candidate,
                candidate_profit,
                meta,
                feasible_samples
            ) = solve_pair_by_sa(
                current_solution,
                i1,
                i2,
                seed
            )

            max_bits_seen = max(
                max_bits_seen,
                meta["bit_count"]
            )

            total_feasible_samples += (
                feasible_samples
            )

            if (
                candidate_profit
                > current_profit + 1e-8
            ):

                current_solution = (
                    candidate
                )

                current_profit = (
                    candidate_profit
                )

                accepted_moves += 1

                improved_this_sweep += 1

        if (
            improved_this_sweep
            == 0
        ):

            break

    return {
        "solution": current_solution,
        "profit": current_profit,
        "accepted_moves": accepted_moves,
        "max_bits_seen": max_bits_seen,
        "feasible_samples": total_feasible_samples
    }


# ============================================================
# 15. 第一轮 multi-start 实验
#
# 先只用 5 个随机初始解，
# 避免第一次测试计算时间过长。
#
# 如果这个版本能正常运行并明显改善初始解，
# 下一步再决定是否提高：
# - N_STARTS
# - PAIR_NUM_READS
# - PAIR_NUM_SWEEPS
# ============================================================

MILP_BENCHMARK = (
    66459.80
)

N_STARTS = 5

MASTER_SEED = (
    2024
)

master_rng = random.Random(
    MASTER_SEED
)

best_solution = None

best_profit = (
    -np.inf
)

hit_count = 0

final_profits = []


print(
    "\n================ "
    "动态 pair-subQUBO + SA "
    "================\n"
)

print(
    "PAIR_NUM_READS =",
    PAIR_NUM_READS
)

print(
    "PAIR_NUM_SWEEPS =",
    PAIR_NUM_SWEEPS
)

print(
    "N_STARTS =",
    N_STARTS
)

print()


for start_id in range(
    1,
    N_STARTS + 1
):

    local_seed = (
        master_rng.randrange(
            10**9
        )
    )

    rng = random.Random(
        local_seed
    )

    initial_solution = (
        generate_initial_solution(
            rng
        )
    )

    initial_info = (
        evaluate_solution(
            initial_solution
        )
    )

    result = (
        pair_subqubo_descent(
            initial_solution,
            rng,
            start_id=start_id,
            max_outer_sweeps=4
        )
    )

    final_solution = (
        result["solution"]
    )

    final_profit = (
        result["profit"]
    )

    final_info = (
        evaluate_solution(
            final_solution
        )
    )

    final_profits.append(
        final_profit
    )

    if (
        final_profit
        > best_profit + 1e-8
    ):

        best_profit = (
            final_profit
        )

        best_solution = (
            final_solution.copy()
        )

    if (
        abs(
            final_profit
            - MILP_BENCHMARK
        )
        < 1e-5
    ):

        hit_count += 1

    print(
        f"start {start_id:02d}: "
        f"initial={initial_info['profit']:.2f}, "
        f"final={final_profit:.2f}, "
        f"gap={MILP_BENCHMARK-final_profit:.2f}, "
        f"moves={result['accepted_moves']}, "
        f"max_bits={result['max_bits_seen']}, "
        f"feasible_samples={result['feasible_samples']}"
    )


# ============================================================
# 16. 结果统计
# ============================================================

final_profits = np.array(
    final_profits,
    dtype=float
)

print(
    "\n================ "
    "实验结果 "
    "================"
)

print(
    "最好利润 =",
    best_profit
)

print(
    "MILP基准 =",
    MILP_BENCHMARK
)

print(
    "最好方案绝对差 =",
    MILP_BENCHMARK
    - best_profit
)

print(
    "最好方案相对gap =",
    (
        MILP_BENCHMARK
        - best_profit
    )
    / MILP_BENCHMARK
    * 100,
    "%"
)

print(
    "命中MILP最优次数 =",
    hit_count,
    "/",
    N_STARTS
)

print(
    "最终利润平均值 =",
    np.mean(
        final_profits
    )
)

if len(
    final_profits
) > 1:

    print(
        "最终利润标准差 =",
        np.std(
            final_profits,
            ddof=1
        )
    )


# ============================================================
# 17. 输出最好方案
# ============================================================

best_info = (
    evaluate_solution(
        best_solution
    )
)

print(
    "\n================ "
    "最好方案 "
    "================"
)

print(
    "五年净利润 =",
    best_info[
        "profit"
    ]
)

print(
    "启动资金 =",
    best_info[
        "budget"
    ],
    "/ 4000"
)

print(
    "矿车使用 =",
    best_info[
        "truck_used"
    ].tolist()
)

print(
    "矿车库存 =",
    N.tolist()
)

print(
    "型号数 =",
    best_info[
        "type_count"
    ]
)

print(
    "原问题可行 =",
    best_info[
        "feasible"
    ]
)

print(
    "\n选择的配置："
)

for i, cfg in enumerate(
    best_solution
):

    if cfg is None:
        continue

    print(
        f"挖{i+1}: "
        f"{cfg['name']}, "
        f"矿{cfg['j']+1}, "
        f"x={cfg['x']}, "
        f"k={cfg['k']}, "
        f"u={cfg['u']:.6f}, "
        f"profit={cfg['profit']:.2f}"
    )

import numpy as np
from scipy.integrate import solve_ivp
import time

# =========================
# 1. 物理参数
# =========================
m1 = 4866.0
m2 = 2433.0
m_add = 1165.992

rho = 1025.0
g = 9.8
A = np.pi

k1 = rho * g * A
c1 = 167.8395
k2 = 80000.0

F0 = 4890.0
w = 2.2143


def F_wave(t):
    return F0 * np.cos(w * t)


# =========================
# 2. 平均功率计算
# =========================
def calculate_power(c_prop, b):

    def system(t, state):
        x1, x2, v1, v2 = state

        v_rel = v1 - v2

        # 非线性PTO阻尼力
        F_pto = c_prop * np.abs(v_rel) ** b * v_rel

        dx1_dt = v1
        dx2_dt = v2

        dv1_dt = (
            F_wave(t)
            - k1 * x1
            - k2 * (x1 - x2)
            - c1 * v1
            - F_pto
        ) / (m1 + m_add)

        dv2_dt = (
            k2 * (x1 - x2)
            + F_pto
        ) / m2

        return [dx1_dt, dx2_dt, dv1_dt, dv2_dt]

    # 论文计算到180秒
    sol = solve_ivp(
        fun=system,
        t_span=(0.0, 180.0),
        y0=[0.0, 0.0, 0.0, 0.0],
        method="BDF",
        dense_output=True,
        rtol=1e-7,
        atol=1e-9
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # 论文采用100～180秒计算平均功率
    t_start = 100.0
    t_end = 180.0

    t_eval = np.linspace(t_start, t_end, 100000)
    state = sol.sol(t_eval)

    v1 = state[2]
    v2 = state[3]
    v_rel = v1 - v2

    # P = C |v_rel|^b * v_rel^2
    P_instant = c_prop * np.abs(v_rel) ** b * v_rel ** 2

    P_average = (
        np.trapezoid(P_instant, t_eval)
        / (t_end - t_start)
    )

    return P_average


# =========================
# 3. 论文参数验证
# =========================
target_c_prop = 81478.0
target_b = 0.3377

start_time = time.time()

result_power = calculate_power(
    target_c_prop,
    target_b
)

end_time = time.time()

print("\n=== 参数验证结果 ===")
print(f"比例系数 C = {target_c_prop}")
print(f"幂指数 b = {target_b}")
print(f"平均输出功率 P = {result_power:.6f} W")
print(f"计算时间 = {end_time - start_time:.2f} s")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import time

# =========================
# 1. 物理参数
# =========================
# 问题4对应参数
m_float = 4866.0
m_vibrator = 2433.0

m_add = 1091.099

k_restoring_force = 1025.0 * 9.8 * np.pi
k_restoring_moment = 8890.7

c_wave_damping_heave = 528.5018
c_wave_damping_pitch = 1655.909

k_spring_heave = 80000.0
k_spring_pitch = 250000.0

F0_heave = 1760.0

# 注意：这是纵摇激励力矩幅值，不是普通力
M0_pitch = 2140.0

I_add = 7142.493
I_float = 8289.434363698517

w = 1.9806
T = 2.0 * np.pi / w


def F_heave_wave(t):
    return F0_heave * np.cos(w * t)


def M_pitch_wave(t):
    return M0_pitch * np.cos(w * t)


# =========================
# 2. 平均功率目标函数
# =========================
def objective(params):
    c_pto_line, c_pto_rotate = params

    def system(t, state):
        (
            z_float,
            z_vibrator,
            v_float,
            v_vibrator,
            theta_float,
            theta_vibrator,
            omega_float,
            omega_vibrator
        ) = state

        I_vibrator = (
            202.75
            + m_vibrator
            * (0.75 + z_vibrator - z_float) ** 2
        )

        dz_float_dt = v_float
        dz_vibrator_dt = v_vibrator

        dv_float_dt = (
            F0_heave * np.cos(w * t)
            - k_restoring_force * z_float
            - k_spring_heave * (z_float - z_vibrator)
            - c_wave_damping_heave * v_float
            - c_pto_line * (v_float - v_vibrator)
        ) / (m_float + m_add)

        dv_vibrator_dt = (
            k_spring_heave * (z_float - z_vibrator)
            + c_pto_line * (v_float - v_vibrator)
        ) / m_vibrator

        dtheta_float_dt = omega_float
        dtheta_vibrator_dt = omega_vibrator

        domega_float_dt = (
            M0_pitch * np.cos(w * t)
            - k_restoring_moment * theta_float
            - k_spring_pitch * (theta_float - theta_vibrator)
            - c_wave_damping_pitch * omega_float
            - c_pto_rotate * (omega_float - omega_vibrator)
        ) / (I_float + I_add)

        domega_vibrator_dt = (
            k_spring_pitch * (theta_float - theta_vibrator)
            + c_pto_rotate * (omega_float - omega_vibrator)
        ) / I_vibrator

        return np.array([
            dz_float_dt,
            dz_vibrator_dt,
            dv_float_dt,
            dv_vibrator_dt,
            dtheta_float_dt,
            dtheta_vibrator_dt,
            domega_float_dt,
            domega_vibrator_dt
        ])

    t_start = 100.0
    t_end = 180.0

    sol = solve_ivp(
        system,
        (0.0, t_end),
        np.zeros(8),
        method="RK45",
        dense_output=True,
        rtol=1e-7,
        atol=1e-9,
        max_step=T / 100.0
    )

    if not sol.success or sol.sol is None:
        return 1e30

    t_eval = np.linspace(t_start, t_end, 8001)
    state = sol.sol(t_eval)

    v_rel = state[2] - state[3]
    omega_rel = state[6] - state[7]

    P_instant = (
        c_pto_line * v_rel**2
        + c_pto_rotate * omega_rel**2
    )

    P_average = (
        np.trapezoid(P_instant, t_eval)
        / (t_end - t_start)
    )

    return -P_average


# =========================
# 3. 差分进化算法
# =========================
print("启动差分进化算法进行二维全局寻优……")

start_time = time.time()

bounds = [
    (0.0, 100000.0),  # 直线阻尼
    (0.0, 100000.0)   # 旋转阻尼
]

result = differential_evolution(
    func=objective,
    bounds=bounds,
    strategy="best1bin",
    maxiter=100,
    popsize=15,
    tol=1e-7,
    mutation=(0.5, 1.0),
    recombination=0.7,
    seed=42,
    polish=True,
    disp=True
)

end_time = time.time()

# =========================
# 4. 输出结果
# =========================
print("\n情况3线性PTO阻尼优化结果")
print(f"最优直线阻尼系数 C_h = {result.x[0]:.4f} N·s/m")
print(f"最优旋转阻尼系数 C_p = {result.x[1]:.4f} N·m·s/rad")
print(f"最大平均输出功率 P_max = {-result.fun:.6f} W")
print(f"优化是否成功：{result.success}")
print(f"终止信息：{result.message}")
print(f"目标函数计算次数：{result.nfev}")
print(f"总计算耗时：{end_time - start_time:.2f} 秒")
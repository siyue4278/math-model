import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, minimize
from scipy.optimize import differential_evolution


# --- 1. 基础参数与问题2设定的波浪频率 ---
m1 = 4866
m2 = 2433
m_add = 1165.992
k1 = 1025 * 9.8 * np.pi
c1 = 167.8395
k2 = 80000

F0 = 4890
w = 2.2143
T = 2 * np.pi / w
def F_wave(t):
    return F0 * np.cos(w * t)

# --- 2. 情况(1) 平均功率计算与单变量优化 ---
def calc_power_linear(c2):
    def system(t, state):
        y1, y2, y3, y4 = state
        v_rel = y3 - y4
        F_pto = c2 * v_rel
        dy1_dt = y3
        dy2_dt = y4
        dy3_dt = (1 / (m1 + m_add)) * (F_wave(t) - k1 * y1 - k2 * (y1 - y2) - c1 * y3 - F_pto)
        dy4_dt = (1 / m2) * (k2 * (y1 - y2) + F_pto)
        return [dy1_dt, dy2_dt, dy3_dt, dy4_dt]

    t_span = (0.0, 40 * T)
    y0 = [0.0, 0.0, 0.0, 0.0]
    
    # ODE 求解并提取 20T ~ 40T 稳态数据
    sol = solve_ivp(system, t_span, y0, method="RK45", dense_output=True, rtol=1e-5, atol=1e-7)
    t_eval = np.linspace(20 * T, 40 * T, 1000)
    sol_eval = sol.sol(t_eval)
    v_rel = sol_eval[2, :] - sol_eval[3, :]
    
    # 数值积分计算平均功率
    P_inst = c2 * (v_rel ** 2)
    P_avg = np.trapezoid(P_inst, t_eval) / (20 * T)
    return P_avg

# 执行单变量一维搜索 (Golden-section / Bounded 算法)
res_1 = minimize_scalar(lambda c: -calc_power_linear(c), bounds=(0, 100000), method='bounded')

# --- 3. 情况(2) 非线性阻尼二维优化 ---
def calc_power_nonlinear(params):
    c_prop, b = params
    def system(t, state):
        y1, y2, y3, y4 = state
        v_rel = y3 - y4
        c2_nonlin = c_prop * (np.abs(v_rel) ** b)
        F_pto = c2_nonlin * v_rel
        dy1_dt = y3
        dy2_dt = y4
        dy3_dt = (1 / (m1 + m_add)) * (F_wave(t) - k1 * y1 - k2 * (y1 - y2) - c1 * y3 - F_pto)
        dy4_dt = (1 / m2) * (k2 * (y1 - y2) + F_pto)
        return [dy1_dt, dy2_dt, dy3_dt, dy4_dt]

    t_span = (0.0, 40 * T)
    y0 = [0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(system, t_span, y0, method="BDF", dense_output=True, rtol=1e-6, atol=1e-6)
    
    t_eval = np.linspace(20 * T, 40 * T, 1000)
    sol_eval = sol.sol(t_eval)
    v_rel = sol_eval[2, :] - sol_eval[3, :]
    
    P_inst = c_prop * (np.abs(v_rel) ** (b + 2))
    P_avg = np.trapezoid(P_inst, t_eval) / (20 * T)
    return P_avg

# 执行二维寻优 (L-BFGS-B 算法)
initial_guess = [50000, 0.5]
bounds = [(0, 100000), (0, 1)]
res_2 = minimize(lambda p: -calc_power_nonlinear(p), initial_guess, method='L-BFGS-B', bounds=bounds)

# --- 4. 结果汇报 ---
print(f"=== 情况(1) 优化结果 ===")
print(f"最优阻尼系数 c2 = {res_1.x:.2f} N·s/m")
print(f"最大平均输出功率 P_max = {-res_1.fun:.2f} W\n")

print(f"=== 情况(2) 优化结果 ===")
print(f"最优比例系数 c_prop = {res_2.x[0]:.2f}")
print(f"最优幂指数 b = {res_2.x[1]:.4f}")
print(f"最大平均输出功率 P_max = {-res_2.fun:.2f} W")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.optimize import minimize_scalar
import time

#1.物理参数
m_float = 4866
m_vibrator = 2433
m_add = 1165.992
k_restoring = 1025 * 9.8 * np.pi
c_daming = 167.8395
k_spring = 80000
F0 = 4890
w = 2.2143
T = 2 * np.pi / w

#2.系统构建
    #1.线性阻尼
def F_wave(t):
    return F0 * np.cos(w * t)

def linear(c_pto_damping):
    def system(t, state):
        y1, y2, y3, y4 = state
        v_rel = y3 - y4
        F_pto_damping = c_pto_damping * v_rel
        dy1_dt = y3
        dy2_dt = y4
        dy3_dt = (1 / (m_float + m_add)) * (F_wave(t) - k_restoring * y1 - k_spring * (y1 - y2) - c_daming * y3 - F_pto_damping)
        dy4_dt = (1 / m_vibrator) * (k_spring * (y1 - y2) + F_pto_damping)
        return [dy1_dt, dy2_dt, dy3_dt, dy4_dt]

    t_span = (0.0, 40 * T)
    y0 = [0.0, 0.0, 0.0, 0.0]
    
        #ODE求解并提取20T~40T稳态数据
    sol = solve_ivp(
        fun=system, 
        t_span=t_span,
        y0=y0,
        method="BDF",
        dense_output=True, 
        rtol=1e-5, 
        atol=1e-7
    )
    t_eval = np.linspace(20 * T, 40 * T, 1000)
    sol_eval = sol.sol(t_eval)
    v_rel = sol_eval[2, :] - sol_eval[3, :]
    
        # 数值积分计算平均功率
    P_inst = c_pto_damping * (v_rel ** 2)
    P_avg = np.trapezoid(P_inst, t_eval) / (20 * T)
    return P_avg

        # 一维边界寻优
res_1 = minimize_scalar(lambda c: -linear(c), bounds=(0, 100000), method='bounded')

    #2.非线性阻尼
def nonlinear(params):
    c_prop, b = params
    
    def system(t, state):
        y1, y2, y3, y4 = state
        v_rel = y3 - y4

        v_safe = np.abs(v_rel) + 1e-6
        
        F_pto = c_prop * (np.abs(v_rel) ** b) * v_rel
        
        dy1_dt = y3
        dy2_dt = y4
        dy3_dt = (1 / (m_float + m_add)) * (F_wave(t) - k_restoring * y1 - k_spring * (y1 - y2) - c_daming * y3 - F_pto)
        dy4_dt = (1 / m_vibrator) * (k_spring * (y1 - y2) + F_pto)
        
        return [dy1_dt, dy2_dt, dy3_dt, dy4_dt]
    
    sol = solve_ivp(
        fun=system,
        t_span=(0.0, 40 * T),
        y0=[0.0, 0.0, 0.0, 0.0],
        method="RK45", 
        dense_output=True,
        rtol=1e-4,  # 在进化算法探索期，稍微放宽容差可以大幅提速
        atol=1e-6
    )

    if not sol.success:
        return 0.0
    
    # 提取 20T 到 40T 的稳态数据进行数值积分
    t_eval = np.linspace(20 * T, 40 * T, 1000)
    sol_eval = sol.sol(t_eval)
    v_rel = sol_eval[2, :] - sol_eval[3, :]
    
    # 瞬时功率 P = F_pto * v_rel
    F_pto_eval = c_prop * (np.abs(v_rel) ** b) * v_rel
    P_instant = F_pto_eval * v_rel 
    P_average = np.trapezoid(P_instant, t_eval) / (20 * T)
    
    #返回负值,differential_evolution默认是求最小值
    return -P_average 

#3.差分进化算法(Differential Evolution)寻优
print("启动差分进化算法进行全局二维寻优...")

start_time = time.time()

bounds_2 = [(0, 100000), (0.0, 1.0)]

res_2 = differential_evolution(
    func=nonlinear, 
    bounds=bounds_2,
    strategy='best1bin',
    maxiter=30,      # 最大进化代数
    popsize=5,       # 种群大小 (5 * 2维 = 10个个体)
    disp=True     
)

end_time = time.time()

#4.结果打印

print(f"\n情况(1)")
print(f"最优比例系数 c = {res_1.x:.2f}")
print(f"线性阻尼最大功率为{-res_1.fun:.2f} w")
print(f"\n情况(2)")
print(f"最优比例系数 c = {res_2.x[0]:.2f}")
print(f"最优幂指数 b = {res_2.x[1]:.4f}")
print(f"最大平均输出功率 P_max = {-res_2.fun:.2f} w")
print(f"总计算耗时: {end_time - start_time:.2f} 秒")
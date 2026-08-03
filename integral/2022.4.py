import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.optimize import minimize_scalar, minimize
import time

#1.物理参数
m_float = 4866
m_vibrator = 2433
m_add = 1028.876

k_restoring_force = 1025 * 9.8 * np.pi
k_restoring_moment = 8890.7

c_wave_daming_heave = 683.4558
c_wave_daming_pitch = 654.3383

k_spring_heave = 80000
k_spring_pitch = 250000

F0_heave = 3640
F0_pitch = 1690

I_add = 7001.914
I_float = 8289.43


w = 1.7152
T = 2 * np.pi / w

#2.系统构建
def F_heave_wave(t):
    return F0_heave * np.cos(w * t)

def F_pitch_wave(t):
    return F0_pitch * np.cos(w * t)

def linear(c_pto_line_daming, c_pto_rotate_daming):
    def wave_energy_heave_systerm(t: float, state: np.ndarray) -> np.ndarray:
    #x_float,x_vibratot,v1,v2,theta1,theta2,w1,w2
        y1, y2, y3, y4, y5, y6, y7, y8 = state

        dy1_dt = y3
        dy2_dt = y4
        dy5_dt = y7
        dy6_dt = y8
        I_vibrator = 202.75 + 2433 * (0.75 + y2 - y1) ** 2

        dy3_dt = (1 / (m_float + m_add)) * (
            F_heave_wave(t) - (k_restoring_force + k_spring_heave) * y1 + k_spring_heave * y2 - (c_wave_daming_heave + c_pto_line_daming) * y3
        + c_pto_line_daming * y4)
        dy4_dt = (1 / m_vibrator) * (
            k_spring_heave * y1  - k_spring_heave * y2 + c_pto_line_daming * y3 - c_pto_line_daming * y4
        )

        dy7_dt = (1 / (I_float + I_add)) * (
            F_pitch_wave(t) - (k_restoring_moment + k_spring_pitch) * y5 + k_spring_pitch * y6 - (c_wave_daming_pitch + c_pto_rotate_daming) * y7
        + c_pto_rotate_daming * y8)
        dy8_dt = (1 / I_vibrator) * (
            k_spring_pitch * y5  - k_spring_pitch * y6 + c_pto_rotate_daming * y7 - c_pto_rotate_daming * y8
        )

        return np.array([dy1_dt, dy2_dt, dy3_dt, dy4_dt, dy5_dt, dy6_dt, dy7_dt, dy8_dt])

    t_span = (0.0, 40 * T)
    y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
        #ODE求解并提取20T~40T稳态数据
    sol = solve_ivp(
        fun=wave_energy_heave_systerm, 
        t_span=t_span,
        y0=y0,
        method="RK45",
        dense_output=True, 
        rtol=1e-4, 
        atol=1e-6
    )
    
    if not sol.success:
            return 0.0
    
    t_eval = np.linspace(20 * T, 40 * T, 1000)
    sol_eval = sol.sol(t_eval)
    v_rel = sol_eval[2, :] - sol_eval[3, :]
    w_rel = sol_eval[6, :] - sol_eval[7, :]
    
    # 数值积分计算平均功率
    P_inst = c_pto_line_daming * (v_rel ** 2) + c_pto_rotate_daming  * (w_rel ** 2)
    P_avg = np.trapezoid(P_inst, t_eval) / (20 * T)
    return -P_avg
    
    # 提取 20T 到 40T 的稳态数据进行数值积分

#3.差分进化算法(Differential Evolution)寻优
print("启动差分进化算法进行全局二维寻优...")

start_time = time.time()

bounds_2 = [(0, 100000), (0, 100000)]

res_2 = differential_evolution(
    func=linear, 
    bounds=bounds_2,
    strategy='best1bin',
    maxiter=30,      # 最大进化代数
    popsize=5,       # 种群大小 (5 * 2维 = 10个个体)
    disp=True     
)

end_time = time.time()

#4.结果打印

print(f"\n情况(2)")
print(f"最优比例系数 c = {res_2.x[0]:.2f}")
print(f"最优比例系数 b = {res_2.x[1]:.4f}")
print(f"最大平均输出功率 P_max = {-res_2.fun:.2f} w")
print(f"总计算耗时: {end_time - start_time:.2f} 秒")
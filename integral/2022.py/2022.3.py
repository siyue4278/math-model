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

c_pto_line_daming = 10000
c_pto_rotate_daming = 1000

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

#3.Runge_Kutta求解

t_span = (0.0, 40 * T)
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

sol_1 = solve_ivp(
    fun=wave_energy_heave_systerm,
    t_span=t_span,
    y0=y0,
    method="RK45",      # 自动使用带有自适应步长的 4(5) 阶龙格库塔法
    dense_output=True,  # 开启稠密输出，在底层自动构建插值函数
    rtol=1e-6,          # 设定相对误差容忍度 (步长自适应的标尺)
    atol=1e-8           # 设定绝对误差容忍度
)

#4.生成0.2秒间隔的精确数据

# 绝对干净的 0.2s 间隔数据，不带任何微小的毫秒偏移
t_target = np.arange(0.0, 40 * T, 0.2)

y_target = sol_1.sol(t_target)

x1 = y_target.T[:, 0]
x2 = y_target.T[:, 1]
v1 = y_target.T[:, 2]
v2 = y_target.T[:, 3]
theta1 = y_target.T[:, 4]
theta2 = y_target.T[:, 5]
w1 = y_target.T[:, 6]
w2 = y_target.T[:, 7]

#5.导出数据

import pandas as pd

# 创建数据框
df_1 = pd.DataFrame({
    '时间 t (s)': t_target,
    '浮子位移 x1 (m)': x1,
    '振子位移 x2 (m)': x2,
    '浮子速度 v1 (m/s)': v1,
    '振子速度 v2 (m/s)': v2,
    '浮子角位移 θ1': theta1,
    '振子角位移 θ2': theta2,
    '浮子角速度 w1': w1,
    '振子角速度 w2': w2
})
# 保存为 Excel 文件
df_1.to_csv("result3_1.csv", index=False, encoding="utf-8-sig")
print("数据已成功导出至csv文件!")

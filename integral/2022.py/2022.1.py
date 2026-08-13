import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#1.物理参数

m_float = 4866
m_vibrator = 2433
m_add = 1335.535
k_restoring_force = 1025 * 9.8 * np.pi
c_daming = 656.3616
k_spring = 80000
c_pto_line_daming = 10000
F0 = 6250
w = 1.4005
T = 2 * np.pi / w


#2.动力学系统构建

def F_wave(t):
    return F0 * np.cos(w * t)

#(1)线性阻尼
def wave_energy_systerm(t: float, state: np.ndarray) -> np.ndarray:
    y1, y2, y3, y4 = state

    dy1_dt = y3
    dy2_dt = y4

    dy3_dt = (1 / (m_float + m_add)) * (
        F_wave(t) - (k_restoring_force + k_spring) * y1 + k_spring * y2 - (c_daming + c_pto_line_daming) * y3
+ c_pto_line_daming * y4)
    dy4_dt = (1 / m_vibrator) * (
        k_spring * y1  - k_spring * y2 + c_pto_line_daming * y3 - c_pto_line_daming * y4
    )
    return np.array([dy1_dt, dy2_dt, dy3_dt, dy4_dt])

#(2)非线性阻尼
def wave_energy_systerm_nonlinear(t: float, state: np.ndarray) -> np.ndarray:
    y1, y2, y3, y4 = state

    dy1_dt = y3
    dy2_dt = y4

    v_rel = y3 - y4
    c2_nonlinear = 10000 * (np.abs(v_rel) ** 0.5)

    F_pto = c2_nonlinear * v_rel

    dy3_dt = (1 / (m_float + m_add)) * (
        F_wave(t) - k_restoring_force * y1 - k_spring * (y1 - y2) - c_daming * y3 - F_pto
    )
    dy4_dt = (1 / m_vibrator) * (
        k_spring * (y1 - y2) + F_pto
    )
    return np.array([dy1_dt, dy2_dt, dy3_dt, dy4_dt])
#3.Runge_Kutta求解

t_span = (0.0, 40 * T)
y0 = [0.0, 0.0, 0.0, 0.0]

sol_1 = solve_ivp(
    fun=wave_energy_systerm,
    t_span=t_span,
    y0=y0,
    method="RK45",      # 自动使用带有自适应步长的 4(5) 阶龙格库塔法
    dense_output=True,  # 开启稠密输出，在底层自动构建插值函数
    rtol=1e-6,          # 设定相对误差容忍度 (步长自适应的标尺)
    atol=1e-8           # 设定绝对误差容忍度
)

sol_2 = solve_ivp(
    fun=wave_energy_systerm_nonlinear,
    t_span=t_span,
    y0=y0,
    method="RK45",      # 自动使用带有自适应步长的 4(5) 阶龙格库塔法
    dense_output=True,  # 开启稠密输出，在底层自动构建插值函数
    rtol=1e-6,          # 设定相对误差容忍度 (步长自适应的标尺)
    atol=1e-8           # 设定绝对误差容忍度
)
#4.生成0.2秒间隔的精确数据

t_target = np.linspace(20 * T, 40 * T, 1000)

y_target_1 = sol_1.sol(t_target)
y_target_2 = sol_2.sol(t_target)

x1_1 = y_target_1.T[:, 0]
x2_1 = y_target_1.T[:, 1]
v1_1 = y_target_1.T[:, 2]
v2_1 = y_target_1.T[:, 3]



x1_2 = y_target_2.T[:, 0]
x2_2 = y_target_2.T[:, 1]
v1_2 = y_target_2.T[:, 2]
v2_2 = y_target_2.T[:, 3]


#5.导出数据

import pandas as pd

#创建数据框
df_1 = pd.DataFrame({
    '时间 t (s)': t_target,
    '浮子位移 x1 (m)': x1_1,
    '振子位移 x2 (m)': x2_1,
    '浮子速度 v1 (m/s)': v1_1,
    '振子速度 v2 (m/s)': v2_1
})

df_2 = pd.DataFrame({
    '时间 t (s)': t_target,
    '浮子位移 x1 (m)': x1_2,
    '振子位移 x2 (m)': x2_2,
    '浮子速度 v1 (m/s)': v1_2,
    '振子速度 v2 (m/s)': v2_2
})


# 保存为 Excel 文件
df_1.to_csv("result1_12.csv", index=False, encoding="utf-8-sig")
print("数据已成功导出至csv文件")

df_2.to_csv("result1_22.csv", index=False, encoding="utf-8-sig")
print("数据已成功导出至csv文件")


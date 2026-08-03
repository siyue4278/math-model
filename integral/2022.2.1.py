import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. 物理参数设置 ---
m1, m2, m_add = 4866, 2433, 1335.535
k1 = 1025 * 9.8 * np.pi
c1 = 656.3616
k2 = 80000

w = 2.2143
T = 2 * np.pi / w
F0 = 6250

def F_wave(t):
    return F0 * np.cos(w * t)

# --- 2. 定义统一的求解函数 ---
def simulate_system(c2, num_periods=35):
    """根据给定的 PTO 阻尼系数 c2 求解系统相对速度"""
    def system(t, state):
        y1, y2, y3, y4 = state
        v_rel = y3 - y4
        F_pto = c2 * v_rel
        dy3_dt = (1 / (m1 + m_add)) * (F_wave(t) - k1 * y1 - k2 * (y1 - y2) - c1 * y3 - F_pto)
        dy4_dt = (1 / m2) * (k2 * (y1 - y2) + F_pto)
        return [y3, y4, dy3_dt, dy4_dt]
    
    t_span = (0.0, num_periods * T)
    t_eval = np.linspace(0, num_periods * T, 4000) # 高密度采样
    sol = solve_ivp(system, t_span, [0, 0, 0, 0], method="RK45", t_eval=t_eval, rtol=1e-6, atol=1e-8)
    
    v_rel = sol.y[2] - sol.y[3]
    return sol.t / T, v_rel

# --- 3. 分别计算“最恶劣工况”和“最优工况” ---
# 最恶劣工况：无 PTO 阻尼 (c2 = 0)，暂态衰减最慢
periods_worst, v_rel_worst = simulate_system(c2=0)

# 最优工况：加入最佳 PTO 阻尼 (c2 = 36518.07)，暂态衰减极快
periods_opt, v_rel_opt = simulate_system(c2=36518.07)

# --- 4. 论文级高质量绘图 ---
# 设置中文字体（Windows 用 SimHei，Mac 若报错可删掉这两行或换成 Arial Unicode MS）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 5.5), dpi=150)

# 绘制两条曲线
plt.plot(periods_worst, v_rel_worst, color='#FF7F0E', alpha=0.8, linewidth=1.5, 
         label='极端边界工况 (阻尼 $c_2=0$): 衰减极慢')
plt.plot(periods_opt, v_rel_opt, color='#1F77B4', alpha=0.9, linewidth=1.5, 
         label='最优阻尼工况 (阻尼 $c_2=36518$): 快速收敛')

# 添加 20T 稳态截断线与区域划分
plt.axvline(x=20, color='red', linestyle='--', linewidth=2, label='统一定义的稳态截断阈值 ($t = 20T$)')
plt.axvspan(0, 20, facecolor='grey', alpha=0.1, label='暂态区 (包含所有衰减过程)')
plt.axvspan(20, 35, facecolor='green', alpha=0.1, label='稳态区 (绝对等幅振荡)')

# 细节美化
plt.title('不同阻尼工况下相对速度的时域衰减对比 (验证 $20T$ 截断阈值的普适性)', fontsize=14, pad=15)
plt.xlabel('时间 (以波浪周期 $T$ 为单位)', fontsize=12)
plt.ylabel('浮子与振子相对速度 (m/s)', fontsize=12)
plt.xlim(0, 35)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.show()
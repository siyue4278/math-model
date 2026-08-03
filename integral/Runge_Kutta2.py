from __future__ import annotations
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray

def RK4_system(
        f: Callable[[float, Array], Array],
        t_span: tuple[float, float],
        y0: Array | list[float],
        h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    通用型四阶龙格-库塔法 (向量化优化版)
    """
    t0, t_end = map(float, t_span)

    if h <= 0:
        raise ValueError("步长h必须大于0")
    if t_end < t0:
        raise ValueError("t_end必须大于t0")

    y = np.asarray(y0, dtype=float)

    if y.ndim != 1:
        raise ValueError("y0必须是一维向量(例如 [0.5])")

    # 【性能优化核心】：预先计算总步数，并预分配内存，杜绝使用 append
    n_steps = int(np.ceil((t_end - t0) / h))
    
    t_values = np.zeros(n_steps + 1)
    y_values = np.zeros((n_steps + 1, len(y))) # 创建 (步数, 变量数) 的矩阵

    # 写入初始值
    t_values[0] = t0
    y_values[0] = y

    t = t0
    
    # 将 enumerate 用于索引填充，比 append 快很多
    for i in range(1, n_steps + 1):
        step = min(h, t_end - t)
        
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + step / 2, y + step * k1 / 2), dtype=float)
        k3 = np.asarray(f(t + step / 2, y + step * k2 / 2), dtype=float)
        k4 = np.asarray(f(t + step, y + step * k3), dtype=float)

        y = y + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + step
        
        # 直接按索引填入预分配好的数组
        t_values[i] = t
        y_values[i] = y

    return t_values, y_values

# ----------------- 测试部分 -----------------

def f(t: float, y: Array) -> Array:
    # 保证返回值也是数组，支持向量化
    return y - t ** 2 + 1.0

def exact_solution(t: np.ndarray) -> np.ndarray:
    return (t + 1) ** 2 - 0.5 * np.exp(t)

# 调用时注意：传入 t_span 和 列表形式的 y0
t, y_rk4_matrix = RK4_system(
    f=f,
    t_span=(0.0, 2.0),
    y0=[0.5], 
    h=0.2,
)

# 【核心修正】：将 (N, 1) 的矩阵拍扁成 (N,) 的一维数组，以匹配精确解
y_rk4 = y_rk4_matrix.flatten()

y_exact = exact_solution(t)
absolute_error = np.abs(y_rk4 - y_exact)

for ti, yi, ye, error in zip(t, y_rk4, y_exact, absolute_error):
    print(f"{ti:8.2f}{yi:15.8f}{ye:15.8f}{error:15.3e}")

print("\n最大绝对误差:", absolute_error.max())

# 画图
plt.figure(figsize=(8, 5))
plt.plot(t, y_rk4, "o--", label="RK4")
plt.plot(t, y_exact, label="Exact solution")
plt.xlabel("t")
plt.ylabel("y")
plt.title("RK4 numerical solution (Vectorized)")
plt.legend()
plt.grid(True)
plt.show()
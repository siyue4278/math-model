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

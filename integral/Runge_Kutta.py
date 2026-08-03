from __future__ import annotations
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np

def RK4_scalar(
        f: Callable[[float, float], float],
        t0: float,
        y0: float,
        t_end: float,
        h: float,
) -> tuple[np.ndarray, np.ndarray]:

    if h <= 0:
        raise ValueError("步长h必须大于0")

    if t_end < t0:
        raise ValueError("t_end必须大于0")

    t_values = [float(t0)]
    y_values = [float(y0)]

    t = float(t0)
    y = float(y0)

    while t < t_end:
        step = min(h, t_end - t)
        k1 = f(t, y)
        k2 = f(
            t + step / 2,
            y + step * k1 / 2,
        )
        k3 = f(
            t + step / 2,
            y + step * k2 / 2,
        )
        k4 = f(
            t + step,
            y + step * k3,
        )

        y = y + step / 6 * (
            k1 + 2 * k2 + 2 * k3 + k4
        )
        t = t + step
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

def f(t: float, y:float) -> float:
    return y - t ** 2 + 1

def exact_solution(t: np.ndarray) -> np.ndarray:
    return (t + 1) ** 2 - 0.5 * np.exp(t)

t, y_rk4 = RK4_scalar(
    f=f,
    t0=0.0,
    y0=0.5,
    t_end=2.0,
    h=0.2,
)

y_exact = exact_solution(t)
absolute_error = np.abs(y_rk4 - y_exact)

for ti, yi, ye, error in zip(
    t,
    y_rk4,
    y_exact,
    absolute_error,
):
    print(
        f"{ti:8.2f}"
        f"{yi:15.8f}"
        f"{ye:15.8f}"
        f"{error:15.3e}"
    )

print("\n最大绝对误差:", absolute_error.max())

plt.figure(figsize=(8, 5))
plt.plot(t, y_rk4, "o--", label = "RK4")
plt.plot(t, y_exact, label = "Exact solution")
plt.xlabel("t")
plt.ylabel("y")
plt.title("RK4 numerical solution")
plt.legend()
plt.grid(True)
plt.show()
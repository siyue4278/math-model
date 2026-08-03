import matplotlib.pyplot as plt
import numpy as np
from Runge_Kutta_vector import RK4_system

mass = 1.0
spring_constant = 4.0

def oscillator(
        t: float,
        state: np.ndarray,
) -> np.ndarray:
    y1 = x
    y2 = v
    x, v = state
    dx_dt = v
    dv_dt = -(spring_constant / mass) * x
    return np.array([dx_dt, dv_dt])

t, states = RK4_system(
    f=oscillator,
    t_span=(0.0, 10.0),
    y0=[1.0, 0.0],
    h=0.01,
)

x = states[:, 0]
v = states[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(t, x, label="Displacement x")
plt.plot(t, v, label="Velocity v")
plt.xlabel("t")
plt.ylabel("State")
plt.title("Harmonic oscillator solved by RK4")
plt.legend()
plt.grid(True)
plt.show()

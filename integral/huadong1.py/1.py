from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize


@dataclass
class Params:

    L1: float = 56.0
    theta1_deg: float = 35.0

    R: float = 100.0
    L2: float = 42.0

    L3: float = 7.0
    theta3_deg: float = 11.0

    rho: float = 1.225          
    g: float = 9.80665          


    mass: float = 70.0         

    mu: float = 0.04

    include_lift: bool = True
    V_ref: float = 25.0       
    rho_ref: float = 1.225    

    s0: float = 0.0
    v0: float = 0.0

    @property
    def total_length(self) -> float:
        return self.L1 + self.L2 + self.L3

P = Params()

def track_geometry(s: float, p: Params = P) -> Tuple[float, float]:


    if s < -1e-10 or s > p.total_length + 1e-10:
        raise ValueError(f"s={s} 超出赛道范围")

    s = min(max(s, 0.0), p.total_length)

    theta1 = np.deg2rad(p.theta1_deg)
    theta3 = np.deg2rad(p.theta3_deg)

    # 第一段：直线
    if s <= p.L1:
        return theta1, 0.0

    # 第二段：圆弧
    if s <= p.L1 + p.L2:
        ds = s - p.L1
        theta = theta1 - ds / p.R
        kappa = 1.0 / p.R
        return theta, kappa

    # 第三段：直线
    return theta3, 0.0


def track_report(p: Params = P):
    theta_after_arc = p.theta1_deg - np.rad2deg(p.L2 / p.R)

    print("=" * 68)
    print("赛道几何检查")
    print("=" * 68)
    print(f"第一段:={p.L1:.2f} m, theta={p.theta1_deg:.2f}°")
    print(f"第二段:L={p.L2:.2f} m, R={p.R:.2f} m")
    print(f"圆弧转角：{np.rad2deg(p.L2 / p.R):.6f}°")
    print(f"圆弧末坡角：{theta_after_arc:.6f}°")
    print(f"第三段:L={p.L3:.2f} m, theta={p.theta3_deg:.2f}°")
    print(f"坡角衔接误差：{theta_after_arc - p.theta3_deg:.6f}°")
    print(f"总助滑距离：{p.total_length:.2f} m")
    print(f"圆弧曲率：{1/p.R:.6f} 1/m")
    print()


RAW_DATA = np.array([
    [0, 22, 45,  0, 0.23494, 30.54704, 0.92468, 6.03852],
    [0, 22, 45,  2, 0.23816, 31.09581, 0.92467, 7.90824],
    [0, 22, 45, -2, 0.23707, 30.45540, 0.92467, 5.55640],
    [0, 22, 45, -4, 0.23144, 33.56088, 0.92467, 5.79355],
    [0, 22, 47,  0, 0.23731, 32.33494, 0.92574, 6.79802],
    [0, 22, 49,  0, 0.24035, 32.48504, 0.92470, 8.92354],
    [0, 24, 45,  0, 0.24047, 30.11432, 0.92579, 7.59590],
    [0, 26, 45,  0, 0.24589, 31.54897, 0.92465, 4.44324],
    [2, 22, 45,  0, 0.23996, 30.42188, 0.92471, 5.73027],
    [4, 22, 45,  0, 0.24495, 32.19710, 0.92360, 9.96331],
], dtype=float)


def convert_table_to_aero_data(raw: np.ndarray, p: Params = P) -> np.ndarray:

    D = raw[:, 5]
    L = raw[:, 7]

    qdyn = 0.5 * p.rho_ref * p.V_ref ** 2
    SD = D / qdyn
    SL = L / qdyn

    return np.column_stack([raw[:, :4], SD, SL])


AERO_DATA = convert_table_to_aero_data(RAW_DATA)




class AeroModel:
  

    VAR_NAMES = ("alpha", "beta", "gamma", "epsilon")
    Q0 = np.array([0.0, 22.0, 45.0, 0.0])

    def __init__(self, data: np.ndarray):
        data = np.asarray(data, dtype=float)

        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError(
                "AERO_DATA 必须是: alpha,beta,gamma,epsilon,SD,SL"
            )

        self.data = data

        base_mask = np.all(
            np.isclose(data[:, :4], self.Q0[None, :], atol=1e-10),
            axis=1
        )

        if not np.any(base_mask):
            raise ValueError("未找到基准姿势 (0,22,45,0)")

        base_row = data[np.flatnonzero(base_mask)[0]]
        self.SD0 = float(base_row[4])
        self.SL0 = float(base_row[5])

        self.sd_interp: Dict[str, interp1d] = {}
        self.sl_interp: Dict[str, interp1d] = {}
        self.bounds: Dict[str, Tuple[float, float]] = {}

        self._build()

    def _build(self):
        for j, name in enumerate(self.VAR_NAMES):
            mask = np.ones(len(self.data), dtype=bool)

            for k in range(4):
                if k != j:
                    mask &= np.isclose(
                        self.data[:, k],
                        self.Q0[k],
                        atol=1e-10,
                    )

            subset = self.data[mask]

            if len(subset) < 2:
                raise ValueError(
                    f"{name} 切片数据不足，无法建立插值函数。"
                )

            x = subset[:, j]
            sd = subset[:, 4]
            sl = subset[:, 5]

            order = np.argsort(x)
            x = x[order]
            sd = sd[order]
            sl = sl[order]

            x_unique, idx = np.unique(x, return_index=True)
            sd_unique = sd[idx]
            sl_unique = sl[idx]

            if len(x_unique) < 2:
                raise ValueError(f"{name} 的插值节点少于2个。")

            self.sd_interp[name] = interp1d(
                x_unique,
                sd_unique - self.SD0,
                kind="linear",
                bounds_error=True,
                assume_sorted=True,
            )

            self.sl_interp[name] = interp1d(
                x_unique,
                sl_unique - self.SL0,
                kind="linear",
                bounds_error=True,
                assume_sorted=True,
            )

            self.bounds[name] = (
                float(x_unique.min()),
                float(x_unique.max()),
            )

    def validate_q(self, q: np.ndarray):
        q = np.asarray(q, dtype=float)

        if q.shape != (4,):
            raise ValueError(
                "q 必须是 [alpha,beta,gamma,epsilon] 四个角度。"
            )

        for i, name in enumerate(self.VAR_NAMES):
            lo, hi = self.bounds[name]
            if q[i] < lo - 1e-12 or q[i] > hi + 1e-12:
                raise ValueError(
                    f"{name}={q[i]:.6f} 超出插值范围 [{lo}, {hi}]"
                )

    def SD(self, q: np.ndarray) -> float:
        self.validate_q(q)

        q = np.asarray(q, dtype=float)

        return float(
            self.SD0
            + sum(
                float(self.sd_interp[name](q[i]))
                for i, name in enumerate(self.VAR_NAMES)
            )
        )

    def SL(self, q: np.ndarray) -> float:
        self.validate_q(q)

        q = np.asarray(q, dtype=float)

        return float(
            self.SL0
            + sum(
                float(self.sl_interp[name](q[i]))
                for i, name in enumerate(self.VAR_NAMES)
            )
        )

    def report(self):
        print("=" * 68)
        print("气动模型")
        print("=" * 68)
        print(f"参考速度 V_ref = {P.V_ref:.4f} m/s")
        print(f"参考空气密度 rho_ref = {P.rho_ref:.6f} kg/m^3")
        print(f"基准姿势 q0 = {self.Q0}")
        print(f"SD0 = {self.SD0:.8f} m^2")
        print(f"SL0 = {self.SL0:.8f} m^2")
        for name in self.VAR_NAMES:
            print(f"{name:8s}: {self.bounds[name]}")
        print()


AERO = AeroModel(AERO_DATA)


def aerodynamic_forces(v: float, q: np.ndarray, p: Params, aero: AeroModel):

    v = max(float(v), 0.0)

    SD = aero.SD(q)
    SL = aero.SL(q)

    D = 0.5 * p.rho * v ** 2 * SD
    L = 0.5 * p.rho * v ** 2 * SL

    if not p.include_lift:
        L = 0.0

    return D, L


def normal_force(v: float, s: float, L: float, p: Params):
    theta, kappa = track_geometry(s, p)

    N = (
        p.mass * p.g * np.cos(theta)
        + p.mass * v ** 2 * kappa
        - L
    )

    return N


def rhs(t: float, y: np.ndarray, q: np.ndarray,
        p: Params, aero: AeroModel):
    
    s, v = float(y[0]), float(y[1])

    # 防止 ODE 数值误差把 s 推出终点
    s_eff = min(max(s, 0.0), p.total_length)

    theta, _ = track_geometry(s_eff, p)

    D, L = aerodynamic_forces(v, q, p, aero)
    N = normal_force(v, s_eff, L, p)

    N_eff = max(N, 0.0)

    friction = p.mu * N_eff

    dvdt = (
        p.g * np.sin(theta)
        - D / p.mass
        - friction / p.mass
    )

    return np.array([v, dvdt])


def end_event(t, y, q, p, aero):
    return y[0] - p.total_length


end_event.terminal = True
end_event.direction = 1


def make_end_event(q, p, aero):
    def event(t, y):
        return end_event(t, y, q, p, aero)
    event.terminal = True
    event.direction = 1
    return event


def simulate(q: np.ndarray, p: Params = P, aero: AeroModel = AERO):

    q = np.asarray(q, dtype=float)
    aero.validate_q(q)


    t_max = 60.0

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, q, p, aero),
        t_span=(0.0, t_max),
        y0=np.array([p.s0, p.v0], dtype=float),
        events=make_end_event(q, p, aero),
        rtol=1e-8,
        atol=1e-10,
        max_step=0.08,
    )

    if sol.status != 1:
        raise RuntimeError(
            "积分未在规定时间内到达终点。"
            f" 最后 s={sol.y[0, -1]:.6f} m, "
            f"v={sol.y[1, -1]:.6f} m/s, "
            f"t={sol.t[-1]:.6f} s。"
        )

    return sol


def terminal_speed(q: np.ndarray, p: Params = P, aero: AeroModel = AERO):
    sol = simulate(q, p, aero)
    return float(sol.y[1, -1])


def minimum_normal_force(q: np.ndarray, p: Params, aero: AeroModel):
    sol = simulate(q, p, aero)

    N_values = []

    for s, v in zip(sol.y[0], sol.y[1]):
        theta, kappa = track_geometry(float(s), p)
        _, L = aerodynamic_forces(float(v), q, p, aero)
        N = (
            p.mass * p.g * np.cos(theta)
            + p.mass * v ** 2 * kappa
            - L
        )
        N_values.append(N)

    return float(np.min(N_values))


def optimize_posture(p: Params, aero: AeroModel):

    bounds = [
        aero.bounds["alpha"],
        aero.bounds["beta"],
        aero.bounds["gamma"],
        aero.bounds["epsilon"],
    ]

    def objective(q):
        try:
            q = np.asarray(q, dtype=float)
            vT = terminal_speed(q, p, aero)
            Nmin = minimum_normal_force(q, p, aero)

            if not np.isfinite(vT):
                return 1e6

            if Nmin < -1e-6:
                return 1e5 + abs(Nmin)

            return -vT

        except Exception:
            return 1e6

    print("\n开始全局姿势优化（Differential Evolution）...")
    print(f"搜索维度: {len(bounds)}，种群规模约: {12 * len(bounds)}")

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=2026,
        popsize=6,
        maxiter=25,
        tol=1e-5,
        polish=False,
        workers=1,
        updating="immediate",
        disp=True,
    )

    print("\n开始局部精修...")
    local_result = minimize(
        objective,
        global_result.x,
        method="Powell",
        bounds=bounds,
        options={
            "xtol": 1e-7,
            "ftol": 1e-9,
            "maxiter": 300,
        },
    )

    return global_result, local_result


def main():
    track_report(P)
    AERO.report()

    q0 = np.array([0.0, 22.0, 45.0, 0.0])

    print("=" * 68)
    print("基准姿势")
    print("=" * 68)
    print("q0 =", q0)
    print(f"SD(q0) = {AERO.SD(q0):.8f} m^2")
    print(f"SL(q0) = {AERO.SL(q0):.8f} m^2")
    print(f"vT(q0) = {terminal_speed(q0, P, AERO):.6f} m/s")
    print()

    import time

    P.include_lift = True
    t_opt0 = time.perf_counter()
    global_full, local_full = optimize_posture(P, AERO)

    q_full = local_full.x
    v_full = terminal_speed(q_full, P, AERO)
    Nmin_full = minimum_normal_force(q_full, P, AERO)
    print(f"含升力优化耗时：{time.perf_counter() - t_opt0:.2f} s")

    print("=" * 68)
    print("最优姿势（含升力）")
    print("=" * 68)
    print(f"alpha   = {q_full[0]:.6f}°")
    print(f"beta    = {q_full[1]:.6f}°")
    print(f"gamma   = {q_full[2]:.6f}°")
    print(f"epsilon = {q_full[3]:.6f}°")
    print(f"v_takeoff = {v_full:.8f} m/s")
    print(f"min(N) = {Nmin_full:.8f} N")
    print()

if __name__ == "__main__":
    main()

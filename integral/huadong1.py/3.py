from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np
from scipy.optimize import differential_evolution


@dataclass
class Anthropometry:
    height: float = 1.76
    mass: float = 70.0

    trunk_length: float = 0.53
    thigh_length: float = 0.42
    shank_length: float = 0.38
    foot_length: float = 0.12

    trunk_mass: float = 35.0
    both_upper_arm_mass: float = 4.2
    both_forearm_hand_mass: float = 4.2
    both_thigh_mass: float = 14.0
    both_shank_foot_mass: float = 7.0
    head_neck_mass: float = 5.6

    trunk_com_from_hip: float = 0.56
    thigh_com_from_hip: float = 0.44
    shank_com_from_knee: float = 0.42

    @property
    def hat_mass(self):
        return (self.trunk_mass + self.both_upper_arm_mass
                + self.both_forearm_hand_mass + self.head_neck_mass)

    @property
    def one_thigh_mass(self):
        return self.both_thigh_mass / 2.0

    @property
    def one_shank_foot_mass(self):
        return self.both_shank_foot_mass / 2.0

A = Anthropometry()


@dataclass
class Params:
    g: float = 9.80665
    rho: float = 1.225
    slope_deg: float = 34.0

    hip_flex_max_deg: float = 145.0
    hip_extension_max_deg: float = 40.0
    knee_flex_max_deg: float = 145.0
    ankle_dorsiflex_max_deg: float = 30.0
    ankle_plantarflex_max_deg: float = 50.0

    balance_epsilon: float = 0.005

    landing_time_mean: float = 0.19
    landing_time_sd: float = 0.05

    seed: int = 2026

P = Params()

@dataclass
class LandingInput:
    vx: float
    vy: float
    speed: float
    v_t: float
    v_n: float
    u_n: float
    aero_t: float
    aero_n: float
    drag: float
    lift: float
    posture: Tuple[float, float, float]


def slope_basis(beta_deg):
    beta = math.radians(beta_deg)
    es = np.array([math.cos(beta), -math.sin(beta)], dtype=float)
    en = np.array([math.sin(beta),  math.cos(beta)], dtype=float)
    return es, en


def load_q2_terminal_state(p=P):

    vx = 26.672522600000001
    vy = -23.384925840000001
    speed = 35.472217559999997

    alpha = 32.875620854401184
    theta = 17.595794246963838
    lam = 13.221199628637869

    D = 516.583182936779394
    L = 502.402525869297619

    ux, uy = vx/speed, vy/speed
    Fa = np.array([
        -D*ux - L*uy,
        -D*uy + L*ux
    ], dtype=float)

    es, en = slope_basis(p.slope_deg)
    vv = np.array([vx, vy], dtype=float)

    vt = float(np.dot(vv, es))
    vn = float(np.dot(vv, en))

    return LandingInput(
        vx=vx, vy=vy, speed=speed,
        v_t=vt, v_n=vn, u_n=-vn,
        aero_t=float(np.dot(Fa, es)),
        aero_n=float(np.dot(Fa, en)),
        drag=float(D), lift=float(L),
        posture=(alpha, theta, lam),
    )


@dataclass
class Pose:
    d: float
    eta: float
    z_hip: float
    trunk_lean_deg: float
    rear_foot_angle_deg: float = 0.0


@dataclass
class PoseGeometry:
    pose: Pose
    rear_support: np.ndarray
    front_support: np.ndarray
    ankle_rear: np.ndarray
    ankle_front: np.ndarray
    knee_rear: np.ndarray
    knee_front: np.ndarray
    hip: np.ndarray
    com_total: np.ndarray
    angles_deg: Dict[str, float]


def angle_between_deg(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    c = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(float(c)))


def signed_angle_deg(v_from, v_to):
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    cross = a[0]*b[1] - a[1]*b[0]
    return math.degrees(math.atan2(cross, float(np.dot(a, b))))


def trunk_unit_in_slope_frame(phi_deg, beta_deg):
    phi = math.radians(phi_deg)
    beta = math.radians(beta_deg)
    return np.array([math.sin(phi-beta), math.cos(phi-beta)], dtype=float)


def solve_knee(ankle, hip, shank_len, thigh_len):

    ankle = np.asarray(ankle, dtype=float)
    hip = np.asarray(hip, dtype=float)

    v = hip - ankle
    r = float(np.linalg.norm(v))
    rmin = abs(thigh_len-shank_len)
    rmax = thigh_len+shank_len
    if not (rmin < r < rmax):
        return None

    e = v/r

    per = np.array([e[1], -e[0]], dtype=float)

    aa = (shank_len**2 - thigh_len**2 + r**2)/(2*r)
    b2 = shank_len**2 - aa**2
    if b2 < -1e-12:
        return None

    return ankle + aa*e + math.sqrt(max(0.0, b2))*per


def rear_ankle_from_psi(psi_deg, a=A):

    psi = math.radians(float(psi_deg))
    lf = a.foot_length
    return np.array([
        lf - lf*math.cos(psi),
        lf*math.sin(psi)
    ], dtype=float)


def build_pose_geometry(pose: Pose, a=A, p=P):
    d = float(pose.d)
    eta = float(pose.eta)
    zH = float(pose.z_hip)
    psi = float(pose.rear_foot_angle_deg)

    if d <= 1e-6 or not (0.0 <= eta <= 1.0) or zH <= 0.0:
        return None
    if psi < 0.0 or psi >= 90.0:
        return None

    SR = np.array([0.0, 0.0], dtype=float)
    SF = np.array([d,   0.0], dtype=float)

    H = np.array([eta*d, zH], dtype=float)


    AR = rear_ankle_from_psi(psi, a)
    AF = SF.copy()

    KR = solve_knee(AR, H, a.shank_length, a.thigh_length)
    KF = solve_knee(AF, H, a.shank_length, a.thigh_length)
    if KR is None or KF is None:
        return None
    if KR[1] < 0.0 or KF[1] < 0.0:
        return None


    uT = trunk_unit_in_slope_frame(pose.trunk_lean_deg, p.slope_deg)

    r_hat = H + a.trunk_com_from_hip*a.trunk_length*uT
    r_tR = H + a.thigh_com_from_hip*(KR-H)
    r_tF = H + a.thigh_com_from_hip*(KF-H)
    r_sR = KR + a.shank_com_from_knee*(AR-KR)
    r_sF = KF + a.shank_com_from_knee*(AF-KF)

    rG = (
        a.hat_mass*r_hat
        + a.one_thigh_mass*r_tR + a.one_thigh_mass*r_tF
        + a.one_shank_foot_mass*r_sR + a.one_shank_foot_mass*r_sF
    ) / a.mass

    # Joint angles
    knee_R = 180.0 - angle_between_deg(H-KR, AR-KR)
    knee_F = 180.0 - angle_between_deg(H-KF, AF-KF)

    neutral = -uT
    hip_R = signed_angle_deg(neutral, KR-H)
    hip_F = signed_angle_deg(neutral, KF-H)


    shR = KR-AR
    shF = KF-AF
    shank_R_from_normal = math.degrees(math.atan2(shR[0], shR[1]))
    shank_F_from_normal = math.degrees(math.atan2(shF[0], shF[1]))

    ankle_R = shank_R_from_normal - psi
    ankle_F = shank_F_from_normal

    angles = {
        "knee_rear_flex": knee_R,
        "knee_front_flex": knee_F,
        "hip_rear_signed": hip_R,
        "hip_front_signed": hip_F,
        "ankle_rear_signed": ankle_R,
        "ankle_front_signed": ankle_F,
        "rear_foot_ground_angle": psi,
    }

    return PoseGeometry(
        pose=pose,
        rear_support=SR,
        front_support=SF,
        ankle_rear=AR,
        ankle_front=AF,
        knee_rear=KR,
        knee_front=KF,
        hip=H,
        com_total=rG,
        angles_deg=angles,
    )


def joint_range_violations(G: PoseGeometry, p=P):
    ang = G.angles_deg

    def viol(x, lo, hi):
        return max(lo-x, 0.0, x-hi)

    return {
        "hip_rear":  viol(ang["hip_rear_signed"],
                          -p.hip_extension_max_deg, p.hip_flex_max_deg),
        "hip_front": viol(ang["hip_front_signed"],
                          -p.hip_extension_max_deg, p.hip_flex_max_deg),
        "knee_rear": viol(ang["knee_rear_flex"], 0.0, p.knee_flex_max_deg),
        "knee_front":viol(ang["knee_front_flex"], 0.0, p.knee_flex_max_deg),
        "ankle_rear":viol(ang["ankle_rear_signed"],
                          -p.ankle_plantarflex_max_deg, p.ankle_dorsiflex_max_deg),
        "ankle_front":viol(ang["ankle_front_signed"],
                           -p.ankle_plantarflex_max_deg, p.ankle_dorsiflex_max_deg),
    }


def pose_valid(G, p=P):
    if G is None:
        return False
    return max(joint_range_violations(G, p).values()) <= 1e-8



def smoothstep(tau):
    tau = float(np.clip(tau, 0.0, 1.0))
    return tau*tau*(3.0-2.0*tau)


def interp_pose(G0: PoseGeometry, G1: PoseGeometry, tau: float):
    h = smoothstep(tau)
    q0 = G0.pose
    q1 = G1.pose
    return Pose(
        d=q0.d + (q1.d-q0.d)*h,
        eta=q0.eta + (q1.eta-q0.eta)*h,
        z_hip=q0.z_hip + (q1.z_hip-q0.z_hip)*h,
        trunk_lean_deg=q0.trunk_lean_deg + (q1.trunk_lean_deg-q0.trunk_lean_deg)*h,
        rear_foot_angle_deg=q0.rear_foot_angle_deg
                            + (q1.rear_foot_angle_deg-q0.rear_foot_angle_deg)*h,
    )


LEG_MIN = abs(A.thigh_length-A.shank_length)+1e-4
LEG_MAX = A.thigh_length+A.shank_length-1e-4

# x = d0, eta0, z0, phi0, eta1, z1, phi1, psi1
BOUNDS = [
    (1e-3, A.foot_length),  # d0
    (0.0, 1.0),             # eta0
    (LEG_MIN, LEG_MAX),     # zH0
    (0.0, 90.0),            # phi0
    (0.0, 1.0),             # eta1
    (LEG_MIN, LEG_MAX),     # zH1
    (0.0, 90.0),            # phi1
    (0.0, 89.0),            # psi_R1
]


def decode_x(x, a=A):
    d0, e0, z0, p0, e1, z1, p1, psi1 = map(float, x)

    G0 = Pose(
        d=d0, eta=e0, z_hip=z0, trunk_lean_deg=p0,
        rear_foot_angle_deg=0.0,
    )
    G1 = Pose(
        d=a.foot_length, eta=e1, z_hip=z1, trunk_lean_deg=p1,
        rear_foot_angle_deg=psi1,
    )
    return G0, G1


def path_metrics(x, n=41, a=A, p=P):
    P0, P1 = decode_x(x, a)
    G0 = build_pose_geometry(P0, a, p)
    G1 = build_pose_geometry(P1, a, p)

    if not pose_valid(G0, p) or not pose_valid(G1, p):
        return None

    if P1.z_hip >= P0.z_hip:
        return None

    minF = 1.0
    minR = 1.0
    max_asym = 0.0

    for tau in np.linspace(0.0, 1.0, n):
        G = build_pose_geometry(interp_pose(G0, G1, float(tau)), a, p)
        if not pose_valid(G, p):
            return None

        sR = G.rear_support[0]
        sF = G.front_support[0]
        sG = G.com_total[0]
        span = sF-sR
        if span <= 1e-9:
            return None

        rF = (sG-sR)/span
        rR = (sF-sG)/span

        if rF < 0.0 or rR < 0.0:
            return None

        minF = min(minF, rF)
        minR = min(minR, rR)
        max_asym = max(max_asym, abs(rF-rR))

    h = float(G0.com_total[1] - G1.com_total[1])

    return {
        "score": min(minF, minR),
        "min_front": minF,
        "min_rear": minR,
        "max_asym": max_asym,
        "h": h,
        "G0": G0,
        "G1": G1,
    }


def stage1_obj(x):
    m = path_metrics(x, n=25)
    if m is None:
        return 1e4
    return -m["score"]


def optimize_layered(a=A, p=P):

    seed = np.array([
        0.116812,
        0.242392,
        0.760573,
        36.524602,
        0.483610,
        0.735421,
        28.262363,
        10.0
    ], dtype=float)

    de = differential_evolution(
        stage1_obj,
        BOUNDS,
        seed=p.seed,
        popsize=12,
        maxiter=110,
        tol=2e-5,
        polish=True,
        workers=1,
        updating="immediate",
        x0=seed,
    )

    rng = np.random.default_rng(p.seed)
    lo = np.array([b[0] for b in BOUNDS], dtype=float)
    hi = np.array([b[1] for b in BOUNDS], dtype=float)
    width = hi-lo

    candidates = [de.x.copy(), seed.copy()]

    for _ in range(1600):
        z = de.x + rng.normal(0.0, 0.05, size=8)*width
        candidates.append(np.clip(z, lo, hi))

    for sig in (0.02, 0.01):
        for _ in range(900):
            z = de.x + rng.normal(0.0, sig, size=8)*width
            candidates.append(np.clip(z, lo, hi))

    for _ in range(500):
        candidates.append(rng.uniform(lo, hi))

    screened = []
    for x in candidates:
        met = path_metrics(x, n=81)
        if met is not None:
            screened.append((np.asarray(x, dtype=float), met))

    if not screened:
        raise RuntimeError("没有高精度可行候选")

    Mstar = max(m["score"] for _, m in screened)
    floor = Mstar - p.balance_epsilon

    elite = [(x, m) for x, m in screened if m["score"] >= floor]
    if not elite:
        raise RuntimeError("第一层近优集合为空")

    best_x, best_m = max(elite, key=lambda item: item[1]["h"])
    stage1_x, stage1_m = max(screened, key=lambda item: item[1]["score"])

    return {
        "stage1_x": stage1_x,
        "stage1": stage1_m,
        "Mstar": Mstar,
        "floor": floor,
        "screened_count": len(screened),
        "elite_count": len(elite),
        "best_x": best_x,
        "best": best_m,
        "de": de,
    }


def impact_validation(landing: LandingInput, p=P, a=A):
    beta = math.radians(p.slope_deg)
    T = p.landing_time_mean

    J_ground = (
        a.mass*landing.u_n
        + (a.mass*p.g*math.cos(beta)-landing.aero_n)*T
    )
    N_avg = J_ground/T

    tau = np.linspace(0.0, 1.0, 20001)
    g1 = np.exp(-0.5*((tau-0.28)/0.13)**2)
    g2 = 0.90*np.exp(-0.5*((tau-0.68)/0.13)**2)
    envelope = tau*(1.0-tau)
    raw = (g1+g2)*(0.35+0.65*envelope)
    shape = raw/np.trapezoid(raw, tau)

    N = N_avg*shape
    BW = a.mass*p.g

    return {
        "T": T,
        "J_ground": J_ground,
        "N_avg": N_avg,
        "N_avg_BW": N_avg/BW,
        "N_peak": float(np.max(N)),
        "N_peak_BW": float(np.max(N))/BW,
        "mean_normal_decel": landing.u_n/T,
    }


def print_pose(name, G):
    print("\n"+name)
    print("-"*76)
    print(f"d                         = {G.pose.d:.8f} m")
    print(f"eta                       = {G.pose.eta:.8f}")
    print(f"z_hip                     = {G.pose.z_hip:.8f} m")
    print(f"trunk_lean                = {G.pose.trunk_lean_deg:.8f} deg")
    print(f"rear foot angle psi_R     = {G.pose.rear_foot_angle_deg:.8f} deg")
    print(f"rear ankle                = ({G.ankle_rear[0]:.8f}, {G.ankle_rear[1]:.8f}) m")
    print(f"front ankle               = ({G.ankle_front[0]:.8f}, {G.ankle_front[1]:.8f}) m")
    print(f"COM                       = ({G.com_total[0]:.8f}, {G.com_total[1]:.8f}) m")
    for k, v in G.angles_deg.items():
        print(f"{k:28s}= {v:10.6f} deg")


def print_result(opt, landing):
    print("="*76)
    print("Q3 V12: 8D Telemark optimization with latest Q2 V4 boundary")
    print("="*76)
    print("\nQ2 -> Q3 terminal state")
    print(f"vx={landing.vx:.8f}, vy={landing.vy:.8f}, speed={landing.speed:.8f} m/s")
    print(f"v_t={landing.v_t:.8f}, v_n={landing.v_n:.8f}, u_n={landing.u_n:.8f} m/s")

    print("\nLayered optimization")
    print(f"Mstar               = {opt['Mstar']:.8f}")
    print(f"balance floor       = {opt['floor']:.8f}")
    print(f"screened / elite    = {opt['screened_count']} / {opt['elite_count']}")
    print(f"selected balance    = {opt['best']['score']:.8f}")
    print(f"selected h          = {opt['best']['h']:.8f} m")
    print(f"selected psi_R1     = {opt['best']['G1'].pose.rear_foot_angle_deg:.8f} deg")
    print(f"min front / rear    = {opt['best']['min_front']:.8f} / {opt['best']['min_rear']:.8f}")
    print(f"max asymmetry       = {opt['best']['max_asym']:.8f}")

    print_pose("G0 touchdown preparation", opt["best"]["G0"])
    print_pose("G1 buffered Telemark", opt["best"]["G1"])

    val = impact_validation(landing)
    print("\nPost-optimization support-force validation only")
    print("-"*76)
    print(f"T                     = {val['T']:.6f} s")
    print(f"J_ground              = {val['J_ground']:.6f} N s")
    print(f"N_avg                 = {val['N_avg']:.6f} N = {val['N_avg_BW']:.6f} BW")
    print(f"N_peak (equiv. pulse) = {val['N_peak']:.6f} N = {val['N_peak_BW']:.6f} BW")


if __name__ == "__main__":
    landing = load_q2_terminal_state()
    opt = optimize_layered()
    print_result(opt, landing)

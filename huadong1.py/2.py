from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

MASS = 70.0
G = 9.80665
RHO = 1.225
V_IN = 28.45513063
TABLE_DEG = 11.0
PUSH_N = 2.75
SLOPE_DEG = 34.0
X_K = 120.0
X_BOTTOM = 140.0
Y_BOTTOM = -60.0
DT = 0.01
T_MAX = 9.0

CONTROL_X = np.array([0.0,35.0,70.0,105.0,140.0], dtype=np.float64)

@dataclass
class Params:
    m: float = MASS
    g: float = G
    rho: float = RHO
    v_in: float = V_IN
    table_deg: float = TABLE_DEG
    takeoff_push_normal: float = PUSH_N
    slope_deg: float = SLOPE_DEG
    x_K: float = X_K
    x_bottom: float = X_BOTTOM
    y_bottom: float = Y_BOTTOM
    dt: float = DT
    t_max: float = T_MAX
    seed: int = 2026
    de_popsize: int = 8
    de_maxiter: int = 65

P = Params()

A_SD = np.zeros((5,3,3), dtype=np.float64)
B_SL = np.zeros((5,3,3), dtype=np.float64)
C_QM = np.zeros((5,3,3), dtype=np.float64)

_A = {
(0,0,0):8.54e-2,(0,0,1):5.33e-4,(0,0,2):-1.24e-4,
(0,1,0):7.90e-5,(0,1,1):7.05e-5,(0,1,2):9.07e-6,
(0,2,0):1.50e-4,(0,2,1):-2.37e-6,(0,2,2):-1.67e-7,
(1,0,0):3.74e-4,(1,0,1):-3.82e-4,(1,0,2):2.38e-5,
(1,1,0):2.71e-4,(1,1,1):-3.63e-5,(1,1,2):-6.02e-8,
(1,2,0):3.74e-6,(1,2,1):9.61e-7,(1,2,2):-9.33e-9,
(2,0,0):4.53e-5,(2,0,1):4.19e-5,(2,0,2):-1.38e-6,
(2,1,0):4.25e-5,(2,1,1):6.17e-7,(2,1,2):3.64e-8,
(2,2,0):-1.49e-6,(2,2,1):-2.05e-8,(2,2,2):-1.54e-10,
(3,0,0):1.20e-5,(3,0,1):-1.56e-6,(3,0,2):4.91e-8,
(3,1,0):-1.71e-6,(3,1,1):6.70e-8,(3,1,2):-3.62e-9,
(3,2,0):4.73e-8,(3,2,1):-7.22e-10,(3,2,2):4.09e-11,
(4,0,0):-1.63e-7,(4,0,1):1.56e-8,(4,0,2):-4.75e-10,
(4,1,0):1.67e-8,(4,1,1):-9.27e-10,(4,1,2):4.15e-11,
(4,2,0):-4.05e-10,(4,2,1):8.76e-12,(4,2,2):-3.92e-13,
}
_B = {
(0,0,0):4.79e-2,(0,0,1):-5.78e-3,(0,0,2):2.63e-4,
(0,1,0):5.91e-3,(0,1,1):2.57e-4,(0,1,2):-9.12e-6,
(0,2,0):2.79e-5,(0,2,1):-3.78e-6,(0,2,2):-1.62e-8,
(1,0,0):4.78e-3,(1,0,1):-1.02e-3,(1,0,2):7.87e-5,
(1,1,0):6.50e-4,(1,1,1):-1.41e-5,(1,1,2):-3.41e-6,
(1,2,0):-1.51e-5,(1,2,1):1.15e-6,(1,2,2):5.33e-8,
(2,0,0):5.20e-4,(2,0,1):9.20e-5,(2,0,2):-5.02e-6,
(2,1,0):4.39e-6,(2,1,1):1.59e-6,(2,1,2):2.02e-7,
(2,2,0):-6.47e-7,(2,2,1):-9.23e-8,(2,2,2):-2.79e-9,
(3,0,0):-8.85e-6,(3,0,1):-2.33e-6,(3,0,2):1.13e-7,
(3,1,0):-1.12e-6,(3,1,1):-7.78e-9,(3,1,2):-5.30e-9,
(3,2,0):3.64e-8,(3,2,1):1.84e-9,(3,2,2):5.67e-11,
(4,0,0):9.96e-9,(4,0,1):1.99e-8,(4,0,2):-9.03e-10,
(4,1,0):1.36e-8,(4,1,1):-3.46e-10,(4,1,2):4.95e-11,
(4,2,0):-3.58e-10,(4,2,1):-8.53e-12,(4,2,2):-4.38e-13,
}
_C = {
(0,0,0):7.87e-3,(0,0,1):-2.17e-3,(0,0,2):1.64e-4,
(0,1,0):2.52e-3,(0,1,1):4.41e-4,(0,1,2):-1.58e-5,
(0,2,0):1.36e-5,(0,2,1):-9.50e-6,(0,2,2):3.21e-7,
(1,0,0):5.77e-5,(1,0,1):5.39e-4,(1,0,2):-2.63e-5,
(1,1,0):6.90e-4,(1,1,1):-1.99e-4,(1,1,2):7.54e-6,
(1,2,0):-1.41e-5,(1,2,1):4.74e-6,(1,2,2):-1.88e-7,
(2,0,0):5.16e-5,(2,0,1):-3.86e-5,(2,0,2):1.41e-6,
(2,1,0):-3.19e-5,(2,1,1):1.35e-5,(2,1,2):-5.61e-7,
(2,2,0):3.82e-7,(2,2,1):-3.33e-7,(2,2,2):1.48e-8,
(3,0,0):-4.00e-6,(3,0,1):1.04e-6,(3,0,2):-3.20e-8,
(3,1,0):4.26e-7,(3,1,1):-3.22e-7,(3,1,2):1.39e-8,
(3,2,0):8.00e-10,(3,2,1):7.90e-9,(3,2,2):-3.73e-10,
(4,0,0):4.03e-8,(4,0,1):-7.89e-9,(4,0,2):2.08e-10,
(4,1,0):-1.46e-9,(4,1,1):2.45e-9,(4,1,2):-1.07e-10,
(4,2,0):-5.47e-11,(4,2,1):-6.05e-11,(4,2,2):2.95e-12,
}
for k,v in _A.items(): A_SD[k]=v
for k,v in _B.items(): B_SL[k]=v
for k,v in _C.items(): C_QM[k]=v

@njit(cache=True)
def _poly3(coeff, alpha, theta, lam):
    out = 0.0
    for i in range(5):
        ai = alpha ** i
        for j in range(3):
            tj = theta ** j
            for k in range(3):
                out += coeff[i,j,k] * ai * tj * (lam ** k)
    return out

@njit(cache=True)
def _seo_aero_numba(alpha, theta, lam):
    return _poly3(A_SD,alpha,theta,lam), _poly3(B_SL,alpha,theta,lam), _poly3(C_QM,alpha,theta,lam)

def seo_aero(alpha, theta, lam):
    s = _seo_aero_numba(float(alpha),float(theta),float(lam))
    return float(s[0]),float(s[1]),float(s[2])

@njit(cache=True)
def _posture_at_x(x, q):
    if x <= 0.0:
        return q[0],q[1],q[2]
    if x >= 140.0:
        return q[12],q[13],q[14]
    if x < 35.0:
        i=0; x0=0.0; x1=35.0
    elif x < 70.0:
        i=1; x0=35.0; x1=70.0
    elif x < 105.0:
        i=2; x0=70.0; x1=105.0
    else:
        i=3; x0=105.0; x1=140.0
    w=(x-x0)/(x1-x0)
    b0=3*i; b1=3*(i+1)
    return ((1-w)*q[b0]+w*q[b1],
            (1-w)*q[b0+1]+w*q[b1+1],
            (1-w)*q[b0+2]+w*q[b1+2])

def posture_at_x(x,q):
    q=np.asarray(q,dtype=np.float64)
    a,t,l=_posture_at_x(float(x),q)
    return float(a),float(t),float(l)

@njit(cache=True)
def _hill_y(x):
    return -60.0 + (140.0-x)*math.tan(math.radians(34.0))

@njit(cache=True)
def _rhs(x,y,vx,vy,q):
    alpha,theta,lam=_posture_at_x(x,q)
    SD,SL,QM=_seo_aero_numba(alpha,theta,lam)
    V2=vx*vx+vy*vy
    if V2<1e-12:
        return vx,vy,0.0,-G
    V=math.sqrt(V2)
    ux=vx/V; uy=vy/V
    D=0.5*RHO*V2*SD
    L=0.5*RHO*V2*SL
    ax=(-D*ux-L*uy)/MASS
    ay=(-D*uy+L*ux)/MASS-G
    return vx,vy,ax,ay

@njit(cache=True)
def _simulate_numba(q):
    ang=math.radians(TABLE_DEG)
    x=0.0; y=0.0
    vx=V_IN*math.cos(ang)-PUSH_N*math.sin(ang)
    vy=V_IN*math.sin(ang)+PUSH_N*math.cos(ang)
    t=0.0

    while t<T_MAX:
        k1=_rhs(x,y,vx,vy,q)

        x2=x+0.5*DT*k1[0]; y2=y+0.5*DT*k1[1]
        vx2=vx+0.5*DT*k1[2]; vy2=vy+0.5*DT*k1[3]
        k2=_rhs(x2,y2,vx2,vy2,q)

        x3=x+0.5*DT*k2[0]; y3=y+0.5*DT*k2[1]
        vx3=vx+0.5*DT*k2[2]; vy3=vy+0.5*DT*k2[3]
        k3=_rhs(x3,y3,vx3,vy3,q)

        x4=x+DT*k3[0]; y4=y+DT*k3[1]
        vx4=vx+DT*k3[2]; vy4=vy+DT*k3[3]
        k4=_rhs(x4,y4,vx4,vy4,q)

        xn=x+DT*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6.0
        yn=y+DT*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6.0
        vxn=vx+DT*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6.0
        vyn=vy+DT*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6.0

        gap0=1e9 if x<X_K else y-_hill_y(x)
        gap1=1e9 if xn<X_K else yn-_hill_y(xn)

        if xn>=X_K and gap0>0.0 and gap1<=0.0:
            frac=gap0/(gap0-gap1)
            xl=x+frac*(xn-x); yl=y+frac*(yn-y)
            vxl=vx+frac*(vxn-vx); vyl=vy+frac*(vyn-vy)
            tl=t+frac*DT
            a,th,la=_posture_at_x(xl,q)
            code=1 if xl<=X_BOTTOM+1e-8 else 2
            return code,xl,yl,vxl,vyl,tl,a,th,la,0.0

        if xn>=X_BOTTOM:
            den=xn-x
            frac=1.0 if abs(den)<1e-12 else (X_BOTTOM-x)/den
            if frac<0: frac=0.0
            if frac>1: frac=1.0
            y140=y+frac*(yn-y)
            vx140=vx+frac*(vxn-vx); vy140=vy+frac*(vyn-vy)
            t140=t+frac*DT
            a,th,la=_posture_at_x(X_BOTTOM,q)
            clearance=y140-Y_BOTTOM
            return 3,X_BOTTOM,y140,vx140,vy140,t140,a,th,la,clearance

        x,y,vx,vy=xn,yn,vxn,vyn
        t+=DT
        if y<-120.0 or vx<=0.0:
            break

    return 0,x,y,vx,vy,t,0.0,0.0,0.0,0.0

def simulate(q):
    q=np.asarray(q,dtype=np.float64).reshape(-1)
    if q.size!=15:
        raise ValueError("需要15个控制变量")
    z=_simulate_numba(q)
    code=int(z[0]); vx=float(z[3]); vy=float(z[4])
    return {
        "valid": code==1,
        "code": code,
        "reason": {0:"integration_failed",1:"landed_in_zone",2:"landed_outside_zone",3:"overflew_bottom"}.get(code,"unknown"),
        "x_land": float(z[1]),
        "y_land": float(z[2]),
        "vx_land": vx,
        "vy_land": vy,
        "landing_speed": math.hypot(vx,vy),
        "flight_time": float(z[5]),
        "landing_posture": (float(z[6]),float(z[7]),float(z[8])),
        "clearance_at_140": float(z[9]),
    }

def old_smoothstep_seed():
    q0=np.array([2.508859,14.912725,5.909795])
    q1=np.array([25.649698,8.462237,18.084116])
    xc=62.392192
    vals=[]
    for x in CONTROL_X:
        z=min(1.0,max(0.0,x/xc))
        h=z*z*(3-2*z)
        vals.extend((q0+(q1-q0)*h).tolist())
    return np.asarray(vals,dtype=float)

BOUNDS=[]
for i in range(5):
    BOUNDS += [(0.0,50.0),(0.0,40.0),(0.0,25.0)]


LD_REF = 3.7148865392585275
QM_ABS_REF = 0.16567999999999983

@njit(cache=True)
def _flight_style_components(q, x_land):
    
    n = 81
    sum_eff = 0.0
    sum_moment = 0.0

    xmax = x_land
    if xmax < 1e-8:
        xmax = 1e-8
    if xmax > X_BOTTOM:
        xmax = X_BOTTOM

    for i in range(n):
        x = xmax * i / (n - 1)
        a, th, la = _posture_at_x(x, q)
        SD, SL, QM = _seo_aero_numba(a, th, la)

        if SD > 1e-10:
            eff = (SL / SD) / LD_REF
        else:
            eff = 0.0
        if eff < 0.0:
            eff = 0.0
        if eff > 1.0:
            eff = 1.0

        mr = abs(QM) / QM_ABS_REF
        if mr > 1.0:
            mr = 1.0

        sum_eff += eff
        sum_moment += mr

    mean_eff = sum_eff / n
    mean_moment = sum_moment / n

    d_flight_position = 3.0 * (1.0 - mean_eff)
    d_balance = 1.0 * mean_moment

    d_arms_legs = 0.0
    d_skis = 0.0

    d_flight = d_flight_position + d_balance + d_arms_legs + d_skis
    if d_flight > 5.0:
        d_flight = 5.0

    return d_flight_position, d_balance, d_arms_legs, d_skis, d_flight


def style_score_60(q, x_land):

    q = np.asarray(q, dtype=np.float64)
    dpos, dbal, darm, dski, dflight = _flight_style_components(q, float(x_land))

    judge_score = 20.0 - dflight
    judge_score = max(0.0, min(20.0, judge_score))

    three_scores = np.array([judge_score, judge_score, judge_score], dtype=float)
    total = float(np.sum(three_scores))

    return {
        "judge_score": judge_score,
        "judge_scores_retained": three_scores,
        "style_total": total,
        "flight_deduction": float(dflight),
        "flight_position_deduction": float(dpos),
        "balance_deduction": float(dbal),
        "arms_legs_deduction": float(darm),
        "skis_deduction": float(dski),
    }


def distance_points_problem(x_land):

    return 120.0 + 1.8 * (float(x_land) - 120.0)


def score_breakdown(q, r):
    if not r["valid"]:
        return None
    s = style_score_60(q, r["x_land"])
    pd = distance_points_problem(r["x_land"])
    return {
        **s,
        "distance_points": pd,
        "total_points": pd + s["style_total"],
    }

def objective(q):
    r = simulate(q)

    if r["valid"]:
        score = score_breakdown(q, r)
        return -score["total_points"]

    if r["code"] == 3:
        return 1000.0 + 10.0 * max(0.0, r["clearance_at_140"])
    if r["code"] == 2:
        return 1200.0 + abs(r["x_land"] - 130.0)
    return 2000.0

def optimize():

    seed = BEST_Q.copy()
    simulate(seed)  
    de=differential_evolution(
        objective, BOUNDS,
        seed=P.seed, popsize=P.de_popsize, maxiter=P.de_maxiter,
        tol=1e-5, polish=False, workers=1, updating="immediate",
        x0=seed,
    )
    loc=minimize(
        objective,de.x,method="Powell",bounds=BOUNDS,
        options={"maxiter":100,"xtol":1e-4,"ftol":1e-8},
    )
    best=loc.x if loc.fun<de.fun else de.x
    return best,simulate(best),de,loc

BEST_Q = np.array([
    5.29540202739,
    9.10061098629,
    21.8557349543,
    19.2958567167,
    3.89661345107,
    22.2170135708,
    9.94960309276,
    2.40339682333,
    22.237301207,
    16.4510988031,
    7.94971671444,
    23.8711695455,
    42.7016046181,
    1.96892203239,
    18.1697795556,
], dtype=float)

def print_result(q,r):
    print("="*76)
    print("Q2 V4: 5-node continuous posture + competition-format 60-point style score")
    print("="*76)
    print(f"NUMBA_AVAILABLE = {NUMBA_AVAILABLE}")
    print(f"valid           = {r['valid']} ({r['reason']})")
    print(f"x_land          = {r['x_land']:.8f} m")
    print(f"y_land          = {r['y_land']:.8f} m")
    print(f"flight_time     = {r['flight_time']:.8f} s")
    print(f"vx_land         = {r['vx_land']:.8f} m/s")
    print(f"vy_land         = {r['vy_land']:.8f} m/s")
    print(f"landing_speed   = {r['landing_speed']:.8f} m/s")
    print(f"landing_posture = {r['landing_posture']}")
    if r["valid"]:
        sc = score_breakdown(q, r)
        print("\nScoring:")
        print(f"distance_points             = {sc['distance_points']:.8f}")
        print(f"judge_score (same model)    = {sc['judge_score']:.8f} / 20")
        print(f"flight_position_deduction   = {sc['flight_position_deduction']:.8f}")
        print(f"balance_deduction           = {sc['balance_deduction']:.8f}")
        print(f"flight_deduction            = {sc['flight_deduction']:.8f}")
        print(f"style_total                 = {sc['style_total']:.8f} / 60")
        print(f"TOTAL                       = {sc['total_points']:.8f}")
    print("\nControl nodes:")
    for x,row in zip(CONTROL_X,np.asarray(q).reshape(5,3)):
        print(f"x={x:6.1f} m : alpha={row[0]:9.5f}, theta={row[1]:9.5f}, lambda={row[2]:9.5f}")

if __name__=="__main__":
    q,r,de,loc=optimize()
    print_result(q,r)

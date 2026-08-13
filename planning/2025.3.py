import time
import math
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution


#1.物理参数


M1 = np.array([20000.0, 0.0, 2000.0])
FY1 = np.array([17800.0, 0.0, 1800.0])

M1_speed = 300.0
FY1_speed_min = 70.0
FY1_speed_max = 140.0

g = 9.8
smoke_r = 10.0
smoke_sink_speed = 3.0
smoke_valid_time = 20.0

target_r = 7.0
target_h = 10.0
target_center = np.array([0.0, 200.0])

# 导弹从初始位置匀速飞向原点
M1_velocity_vector = (-M1_speed * M1 / np.linalg.norm(M1))
missile_arrival_time = np.linalg.norm(M1) / M1_speed
max_delay = np.sqrt(2.0 * FY1[2] / g)
min_drop_gap = 1.0


#2.优化变量

bounds = [
    (0.0, 2.0 * np.pi),   # theta
    (70.0, 140.0),        # speed
    (0.0, 15.0),          # drop1
    (0.0, 20.0),          # extra12
    (0.0, 20.0),          # extra23
    (0.0, max_delay),     # delay1
    (0.0, max_delay),     # delay2
    (0.0, max_delay),     # delay3
]


#3.圆柱上下表面离散化


def sample_cylinder(num_points: int) -> np.ndarray:
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        num_points,
        endpoint=False
    )

    x = target_center[0] + target_r * np.cos(theta)
    y = target_center[1] + target_r * np.sin(theta)

    bottom_points = np.column_stack([
        x,
        y,
        np.zeros(num_points)
    ])

    top_points = np.column_stack([
        x,
        y,
        np.full(num_points, target_h)
    ])

    target_points = np.vstack([top_points, bottom_points])
    return target_points


#4.在njit中将变量表示


@njit(cache=True, fastmath=True)
def decode_variables(x):
    theta = x[0]
    speed = x[1]

    drop_times = np.zeros(3)

    drop_times[0] = x[2]
    drop_times[1] = drop_times[0] + min_drop_gap + x[3]
    drop_times[2] = drop_times[1] + min_drop_gap + x[4]

    delays = np.zeros(3)

    delays[0] = x[5]
    delays[1] = x[6]
    delays[2] = x[7]

    burst_times = drop_times + delays

    vx = speed * math.cos(theta)
    vy = speed * math.sin(theta)

    burst_pos = np.zeros(3, 3)

    for i in range(3):
        burst_pos[i, 0] = 17800.0 + vx * burst_times[i]
        burst_pos[i, 1] = vy * burst_times[i]

        burst_pos[i, 2] = 1800.0 - 0.5 * g * delays[i] ** 2

    return (
        drop_times,
        delays,
        burst_times,
        vx,
        vy,
        burst_pos
    )


#5.优化缓存


def build_cache(num_points, dt):
    """
        输入:   圆柱圆周采集点数目
                时间精度

        输出:   离散时间切片
                导弹位置                 时间×三维坐标
                导弹指向目标的向量        时间×目标点下标×三维坐标
                导弹指向目标向量的平方  

    """
    target_points = sample_cylinder(num_points)
    times = np.arange(
        0.0,

    )

    missile_pos = (
        M1[None, :]
        + times[:, None] * M1_velocity_vector[None, :]
    )

    directions = (
        target_points[None, :, :]
        - missile_pos[:, None, :]
    )

    directions_squared = np.sum(
        directions ** 2,
        axis=2,
    )

    return (
        np.ascontiguousarray(times),
        np.ascontiguousarray(missile_pos),
        np.ascontiguousarray(directions),
        np.ascontiguousarray(directions_squared),
    )


#6.优化阶段


@njit(cache=True, fastmath=True)


def fast_score(
    x,
    times,
    missile_pos,
    directions,
    directions_squared,
):
    (
        drop_times,
        delays,
        burst_times,
        vx,
        vy,
        burst_pos
    ) = decode_variables(x)

    if drop_times[2] >= missile_arrival_time:
        return -1e8

    for i in range(3):
        if burst_pos[i, 2] <= 0.0:
            return -1e8

        if burst_times[i] >= missile_arrival_time:
            return -1e8

    dt = times[1] - times[0]

    union_hard = 0.0
    union_guide = 0.0

    individual_hard = np.zeros(3)
    individual_guide = np.zeros(3)

    for it in range(times.shape[0]):
        t = times[it]
        now_missile_x = missile_pos[it, 0]
        now_missile_y = missile_pos[it, 1]
        now_missile_z = missile_pos[it, 2]

        active = np.zeros(3, dtype=np.bool_)

        for i in range(3):
            tau = t - burst_times[i]
            if 0.0 <= tau <= smoke_valid_time:
                active[i] = True

        if not (active[0] or active[1] or active[2]):
            continue

        worst_uinon_d2 = 0.0

        #每个烟雾弹遮蔽所有圆柱点的最差距离平方 
        worst_short_k_squared = np.zeros(3)

        #某个时刻烟雾弹遮挡某个目标点
        for ip in range(directions.shape[1]):
            #某个时刻下导弹指向目标点的方向向量.未归一化
            dx = directions[it, ip, 0]
            dy = directions[it, ip, 1]
            dz = directions[it, ip, 2]

            best_short_squared = 1e30

            for k in range(3):
                if not active[k]:
                    continue

                tau = t - burst_times[k]

                #某个烟雾弹爆炸后的坐标

                k_smoke_x = burst_pos[k, 0]
                k_smoke_y = burst_pos[k, 1]
                k_smoke_z = burst_pos[k, 2] - smoke_sink_speed * tau

                #计算某个烟雾弹遮蔽情况
                #导弹指向烟雾弹中心
                m_to_smoke_x = k_smoke_x - now_missile_x
                m_to_smoke_y = k_smoke_y - now_missile_y
                m_to_smoke_z = k_smoke_z - now_missile_z

                s = (
                    dx * m_to_smoke_x
                    + dy * m_to_smoke_y
                    + dz * m_to_smoke_z
                ) / directions_squared[it, ip]

                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0

                #烟雾弹球心指向最短距离的向量
                smoke_to_short_x = now_missile_x - k_smoke_x + s * dx
                smoke_to_short_y = now_missile_y - k_smoke_y + s * dy
                smoke_to_short_z = now_missile_z - k_smoke_z + s * dz


                #该向量长度的平方
                short_squared = (smoke_to_short_x ** 2 + smoke_to_short_y ** 2 + smoke_to_short_z ** 2)

                #
                if short_squared < best_short_squared:
                    best_short_squared = short_squared

                if short_squared > worst_short_k_squared[k]:
                    worst_short_k_squared = short_squared

            if 

            #联合遮蔽的布尔值
            if 

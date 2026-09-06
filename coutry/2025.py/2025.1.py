import numpy as np

#1.物理参数
M1 = np.array([20000.0, 0.0, 2000.0])
FY1 = np.array([17800.0, 0.0, 1800.0])

M1_speed = 300.0
FY1_speed = 120.0

smoke_drop_time = 1.5
smoke_delay_time = 3.6
g = 9.8
smoke_r = 10.0
smoke_sink_speed = 3.0
smoke_valid_time = 20.0
target_r = 7.0
target_h = 10.0
target_center = np.array([0.0, 200.0])

#2.动力系统属性
theta_M1 = np.atan2(M1[2], M1[0])
v_M1 = np.array([
    -M1_speed * np.cos(theta_M1),
    0.0,
    -M1_speed * np.sin(theta_M1)
])

v_FY1 = np.array([
    -FY1_speed,
    0.0,
    0.0
])

#3.导弹和无人机运动轨迹
def motion(t):
    pos_M1 = M1 + v_M1 * t
    pos_FY1 = FY1 + v_FY1 * t
    return pos_M1, pos_FY1

#4.烟雾弹爆炸前运动

#投放时烟雾弹的位置
smoke_drop_pos = motion(smoke_drop_time)[1]

smoke_delay_pos = (
    smoke_drop_pos 
    + v_FY1 * smoke_delay_time
    + np.array([
        0.0,
        0.0,
        -0.5 * g * smoke_delay_time ** 2
    ])
)

#5.烟雾弹爆炸后运动
def smoke_explosion_pos(t):
    smoke_center = smoke_delay_pos + np.array([
        0.0,
        0.0,
        -smoke_sink_speed * t
    ])

    return smoke_center

#6.圆柱上下表面离散化
def sample_cylinder(num_points=300):
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

    return top_points, bottom_points

#7.遮蔽判断

def segment_intersects_sphere(
    M,
    sample_points,
    smoke_center,
    smoke_r=10
):

    #导弹指向目标
    M_to_sample = sample_points - M

    M_to_sample_squared = np.sum(
        M_to_sample ** 2,
        axis=1
    )

    #导弹指向烟雾弹
    M_to_smoke = smoke_center - M

    projection_parameter = np.sum(
        M_to_sample * M_to_smoke,
        axis=1
    ) / M_to_sample_squared

    #将投影参数限制在[0,1]
    projection_parameter = np.clip(
        projection_parameter,
        0.0,
        1.0
    )
    #烟雾弹球心到M_to_sample线段上的最短距离

    #将[t1,t2,t3]变成列向量做广播与M_to_sample相乘
    closest_points = (
        M
        + projection_parameter[:, None] * M_to_sample
    )

    distance_closest_squared = np.sum(
        (closest_points - smoke_center) ** 2,
        axis=1
    )
    #遮蔽返回1
    return distance_closest_squared <= smoke_r ** 2

#8.判断某一时刻是否遮蔽

def check_occlusion(
        t_after_explosion,
        top_points,
        bottom_points
):
    if t_after_explosion < 0.0:
        return False
    if t_after_explosion > smoke_valid_time:
        return False

    #导弹位置
    absolute_time = smoke_drop_time + smoke_delay_time + t_after_explosion
    pos_M1 = motion(absolute_time)[0]

    #烟雾弹爆炸后位置
    smoke_center = smoke_explosion_pos(t_after_explosion)

    #判断导弹到上圆周
    top_check = segment_intersects_sphere(
    M=pos_M1,
    sample_points=top_points,
    smoke_center=smoke_center,
    smoke_r=10
    )

    bottom_check = segment_intersects_sphere(
    M=pos_M1,
    sample_points=bottom_points,
    smoke_center=smoke_center,
    smoke_r=10
    )

    #当上下所有点都被遮蔽
    return bool(
        np.all(top_check)
        and 
        np.all(bottom_check)
    )

#9.二分搜素状态改变时刻

def binary_search(
        left,
        right,
        left_state,
        top_points,
        bottom_points,
        tolerance=1e-8
):
    while right - left > tolerance:
        middle = (left + right) / 2.0
        middle_state = check_occlusion(
            middle,
            top_points,
            bottom_points
        )

        if middle_state == left_state:
            left = middle
        else:
            right = middle

    return (left + right) / 2.0

#10.搜素有效时间区间
def find_effective_time(
    num_points=300,
    coarse_step=0.01,
    tolerance=1e-8
):
    top_points, bottom_points = sample_cylinder(num_points)

    time_grid = np.arange(
        0.0,
        smoke_valid_time + coarse_step,
        coarse_step
    )

    #计算每个粗网格时刻的遮蔽状态[0][1][1][0]
    states = np.array([
        check_occlusion(
            t,
            top_points,
            bottom_points
        )
        for t in time_grid
    ])

    #找出遮蔽状态下标
    blocked_indices = np.flatnonzero(states)

    if blocked_indices.size == 0:
        raise RuntimeWarning(
            "烟雾有效期内没有有效遮蔽"
        )

    first_blocked_index = blocked_indices[0]
    last_blocked_index = blocked_indices[-1]

    if first_blocked_index == 0:
        raise RuntimeError(
            "遮蔽开始时间位于搜索区间左端。"
        )
    
    if last_blocked_index == len(time_grid) - 1:
        raise RuntimeError(
            "遮蔽结束时间位于搜索区间右端。"
        )

    #寻找遮蔽开始时间
    t_start_block = binary_search(
        left=time_grid[first_blocked_index - 1],
        right=time_grid[first_blocked_index],
        left_state=False,
        top_points=top_points,
        bottom_points=bottom_points,
        tolerance=tolerance
    )

    #寻找遮蔽结束时间
    t_end_block = binary_search(  
        left=time_grid[last_blocked_index],
        right=time_grid[last_blocked_index + 1],
        left_state=True,
        top_points=top_points,
        bottom_points=bottom_points,
        tolerance=tolerance
    )

    return t_start_block, t_end_block

#11.主程序

if __name__ == "__main__":
    t_start_block, t_end_block = find_effective_time(
        num_points=300,
        coarse_step=0.01,
        tolerance=1e-8
    )

    print("导弹速度向量:")
    print(v_M1)

    print("\n无人机速度向量:")
    print(v_FY1)

    print("\n烟幕弹投放位置:")
    print(smoke_drop_pos)

    print("\n烟幕弹起爆位置:")
    print(smoke_explosion_pos(0.0))

    print(
        f"\n烟幕弹起爆绝对时刻:"
        f"5.1s"
    )

    print(
        f"\n起爆后遮蔽开始时刻:"
        f"{t_start_block:.9f} s"
    )

    print(
        f"起爆后遮蔽结束时刻："
        f"{t_end_block:.9f} s"
    )

    print(
        f"\n任务开始后遮蔽开始时刻:"
        f"{5.1 + t_start_block:.9f} s"
    )

    print(
        f"任务开始后遮蔽结束时刻："
        f"{5.1 + t_end_block:.9f} s"
    )

    print(
        f"\n有效遮蔽时长:"
        f"{t_end_block - t_start_block:.9f} s"
    )
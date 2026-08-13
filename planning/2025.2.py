import numpy as np

from dataclasses import dataclass
from typing import Callable

from scipy.optimize import brentq, differential_evolution


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
M1_velocity = (
    -M1_speed
    * M1
    / np.linalg.norm(M1)
)



#2.数值算法参数


burst_time_min = 0.05
burst_time_max = 20.0

#阶段1:全局搜素
global_num_points = 50
global_time_step = 0.08
global_popsize = 10
global_maxiter = 40
global_seed = 42

#阶段2:在阶段1最优点附近精细搜素
refine_num_points = 100
refine_time_step = 0.03
refine_popsize = 8
refine_maxiter = 23
refine_seed = 7

#阶段3:最终验证
final_num_points = 300
final_time_step = 0.01
root_tolerance = 1e-10


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


#4.保存一组策略的解码结果


@dataclass(frozen=True)
class strategy:
    theta: float
    speed: float
    drop_time: float
    burst_time: float
    delay_time: float
    rho: float
    velocity: np.ndarray
    drop_pos: np.ndarray
    burst_pos: np.ndarray


#5:决策变量重参数化


def decode_variables(x: np.ndarray) -> strategy:
    """
    优化变量:
        x = (theta, speed, burst_time, rho)

    其中:
        theta       无人机航向角，单位 rad
        speed       无人机速度，单位 m/s
        burst_time  从任务开始到烟幕弹起爆的绝对时间
        rho         起爆延迟占 burst_time 的比例,0 <= rho <= 1

        
    定义:
        delay_time                          投放后到爆炸
        delay_time = rho * burst_time       
        drop_time = burst_time - delay_time 任务开始到投放

    """
    theta, speed, burst_time, rho = np.asarray(x, dtype=float)
    delay_time = rho * burst_time
    drop_time = burst_time - delay_time

    direction = np.array([
        np.cos(theta),
        np.sin(theta),
        0.0
    ])

    velocity = speed * direction

    #无人机在投放时刻的位置
    drop_pos = (
        FY1
        + velocity * drop_time 
    )

    #投放后做平抛，爆炸时的位置
    burst_pos = (
        drop_pos
        + velocity * delay_time
        + np.array([
            0.0,
            0.0,
            -0.5 * g * delay_time ** 2
        ])
    )

    return strategy(
        theta=float(theta),
        speed=float(speed),
        burst_time=float(burst_time),
        rho=float(rho),

        drop_time=float(drop_time),
        delay_time=float(delay_time),
        velocity=velocity,
        drop_pos=drop_pos,
        burst_pos=burst_pos,
    )


#6.遮蔽裕量


def occlusion_margin(
    x: np.ndarray,
    times_after_burst: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """
    对每个时刻t,定义:
        margin(t) = 烟幕半径
                    - 所有目标采样点视线到球心距离的最大值
    判断：
        margin(t) > 0  完全遮蔽且有余量
        margin(t) = 0  位于遮蔽边界
        margin(t) < 0  未完全遮蔽
    参数：
        x
            四个优化变量。

        times_after_burst
            起爆后时间数组，形状为 (T,)。

        target_points
            目标圆柱采样点，形状为 (N, 3)。

    返回：
        形状为 (T,) 的遮蔽裕量数组。
    """
    strategy = decode_variables(x)

    times_after_burst = np.asarray(
        times_after_burst,
        dtype=float,
    )

    # 导弹所使用的是任务开始后的绝对时刻
    absolute_time = (
        strategy.burst_time
        + times_after_burst
    )

    missile_pos = (
        M1[None, :]                #三维坐标，广播后为时间×三维坐标
        + absolute_time[:, None] * M1_velocity[None, :]
    )

    # 烟幕云团起爆后只沿 z 轴负方向下沉
    cloud_positions = (
        strategy.burst_pos[None, :] #三维坐标
        + np.column_stack(          #本来是时间,stack后为时间×三维坐标
            (
                np.zeros_like(times_after_burst),#三维坐标里的x轴
                np.zeros_like(times_after_burst),
                -smoke_sink_speed * times_after_burst,
            )
        )
    )                               #广播后为时间×三维坐标

    #directions[t, i]：
    #时刻t从导弹位置指向第i个目标采样点的向量
    directions = (
        target_points[None, :, :]   #点数×三维坐标
        - missile_pos[:, None, :]   #时间×三维坐标
    )                               #广播后为时间×点数×三维坐标
                                
    direction_length_squared = np.sum(
        directions**2,
        axis=2,                     #三维坐标求和
    )

    # 时刻 t 从导弹位置指向烟幕球心的向量
    start_to_cloud = (
        cloud_positions[:, None, :] #时间×三维坐标,广播后为时间×点数×三维坐标
        - missile_pos[:, None, :]   #时间×三维坐标,广播后为时间×点数×三维坐标
    )

    # 球心在线段所在直线上的投影参数
    projection = np.sum(            #由时间×点数×三维坐标，再对三维坐标轴平方求和，变成时间×点数
        directions * start_to_cloud,
        axis=2,
    ) / direction_length_squared

    #投影参数必须限制在[0, 1]：
    #0是导弹位置,1是目标点。
    projection = np.clip(
        projection,
        0.0,
        1.0,
    )
    #每条导弹—目标视线线段上，距离球心最近的点
    closest_points = (                          #广播后为时间×点数×三维坐标
        missile_pos[:, None, :]                 #时间×三维坐标
        + projection[:, :, None] * directions   #时间×点数
    )
    #距离最短线段
    offsets = (                                 #广播后为时间×点数×三维坐标
        closest_points
        - cloud_positions[:, None, :]
    )

    distance_squared = np.sum(                  
        offsets**2,
        axis=2,
    )

    # 每个时刻取最难遮蔽的那个目标点
    maximum_distance_squared = np.max(distance_squared, axis=1)


    return smoke_r ** 2 - maximum_distance_squared #每个时间下的遮蔽裕量,时间


def occlusion_margin_scalar(
    t_after_burst: float,
    x: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """单个时刻的遮蔽裕量，供 Brent 求根使用。"""
    value = occlusion_margin(
        x=x,
        times_after_burst=np.array([t_after_burst]),
        target_points=target_points,
    )[0]            #把输出数组输出为标量

    return float(value)


#7.从离散裕量估计遮蔽区间

#一次性线性插值
def interpolate_root(
        t_left: float,
        t_right: float,
        margin_left: float,
        margin_right: float,
) -> float:

    denominator = margin_right - margin_left
    if denominator == 0.0:
        return (t_left + t_right) / 2.0
    return (
        t_left
        - margin_left * (t_right - t_left) / denominator
    )

def approximate_intervals(
        times: np.ndarray,
        margins: np.ndarray,
) -> list[tuple[float, float]]:
    """
        根据离散时间点上的裕量正负，
        用线性插值估计所有有效遮蔽区间。
    """
    blocked = margins >= 0.0
    intervals: list[tuple[float, float]] = []

    start: float | None
    start = float(times[0]) if blocked[0] else None

    for index in range(len(times) - 1):
        if blocked[index] == blocked[index + 1]:
            continue

        root = interpolate_root(
            t_left=float(times[index]),
            t_right=float(times[index + 1]),
            margin_left=float(margins[index]),
            margin_right=float(margins[index + 1]),
        )

        if not blocked[index] and blocked[index + 1]:
            # False -> True：进入遮蔽区间
            start = root
        else:
            # True -> False：离开遮蔽区间
            if start is None:
                start = float(times[0])

            intervals.append((start, root))
            start = None

    if blocked[-1] and start is not None:
        intervals.append(
            (start, float(times[-1]))
        )

    return intervals    #返回时间段数×时间区间

#8.快速评价策略
def evaluate_candidate_approximately(
        x: np.ndarray,
        target_points: np.ndarray,
        time_step: float,
) -> tuple[float, float, list[tuple[float, float]]]:
    """
        返回：
            total_duration
                有效遮蔽总时长。
    
            maximum_margin
                整个有效期内最大的遮蔽裕量。
                若始终未完全遮蔽，它仍可衡量离成功还有多远。
    
            intervals
                近似有效遮蔽区间。
    """
    strategy = decode_variables(x)

    #基本约束，惩罚函数
    if strategy.burst_time <= 0.0:
        return 0.0, -1e9, []
    if strategy.drop_time <= 0.0:
        return 0.0, -1e9, []
    if strategy.delay_time <= 0.0:
        return 0.0, -1e9, []
    if strategy.burst_pos[2] <= 0.0:
        return 0.0, -1e9, []

    times = np.arange(
        0.0,
        smoke_valid_time + 0.5 * time_step,
        time_step,
    )

    margins = occlusion_margin(
        x=x,
        times_after_burst=times,
        target_points=target_points,
    )

    intervals = approximate_intervals(
        times=times,
        margins=margins,
    )

    total_duration = sum(
        end - start
        for start, end in intervals
    )

    max_margin = float(np.max(margins))

    return (
        float(total_duration),
        max_margin,
        intervals,
    )

#9.构造优化函数
def make_objective(
        target_points:np.ndarray,
        time_step: float,
) -> Callable[[np.ndarray], float]:
    """
        scipy 的 differential_evolution 默认求最小值。
    
        若已经产生有效遮蔽：
            主要最小化 -duration。
    
        若尚未产生有效遮蔽：
            使用 -maximum_margin 引导算法接近可行区域，
            避免所有差方案的目标函数都等于零。
    """
    def objective(
            x: np.ndarray
    ) -> float:
        duration, max_margin, interval = evaluate_candidate_approximately(
            x=x,
            target_points=target_points,
            time_step=time_step,
        )

        if max_margin <= -1e8:
            return 1e9

        if duration > 0.0:
                    # 不改变“最大化遮蔽时长”的主要目标。
            return (
                    -duration
                   - 1e-6 * max_margin
            )
                    # 没有形成完整遮蔽时，
            # maximum_margin 越接近 0 越好。
        return -max_margin
    return objective

#10.全局搜素

def run_global_search() -> np.ndarray:
    #阶段1:全局差分进化
    target_points = sample_cylinder(global_num_points)
    objective = make_objective(
        target_points=target_points,
        time_step=global_time_step,
    )

    bounds = [
        (0.0, 2.0 * np.pi),
        (FY1_speed_min, FY1_speed_max),
        (burst_time_min, burst_time_max),
        (0.0, 1.0),
    ]

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy="best1bin",
        popsize=global_popsize,
        maxiter=global_maxiter,
        mutation=(0.5, 1.0),
        recombination=0.9,
        seed=global_seed,
        polish=False,
        workers=1,
        updating="immediate",
        tol=1e-6
    )

    return np.asarray(result.x, dtype=float)

#11.局部精化

def build_refine_bounds(
        x_best: np.ndarray,
) -> list[tuple[float, float]]:
    theta, speed, burst_time, rho = x_best

    theta_width = 0.08
    speed_width = 5.0
    burst_width = 0.30
    rho_width = 0.20

    if (
        theta - theta_width < 0.0
        or theta + theta_width > 2.0 * np.pi
    ):
        theta_bounds = (0.0, 2.0 * np.pi)
    else:
        theta_bounds = (
            theta - theta_width,
            theta + theta_width,
        )

    return [
        theta_bounds,

        (
            #无人机下限的最大值，减边界
            max(FY1_speed_min, speed - speed_width),
            #无人机上限的最小值，加边界
            min(FY1_speed_max, speed + speed_width),
        ),

        (
            max(
                burst_time_min, burst_time - burst_width,
            ),
            min(
                burst_time_max, burst_time + burst_width,
            ),
        ),

        (
            max(0.0, rho - rho_width),
            min(1.0, rho + rho_width),
        )
    ]

def run_refine(
        x_global: np.ndarray,
) -> np.ndarray:
    #阶段2:全局最优解附近提高精度
    target_points = sample_cylinder(refine_num_points)

    objective = make_objective(
        target_points=target_points,
        time_step=refine_time_step,
    )

    result = differential_evolution(
        func=objective,
        bounds=build_refine_bounds(x_global),
        strategy="best1bin",
        popsize=refine_popsize,
        maxiter=refine_maxiter,
        mutation=(0.4, 0.9),
        recombination=0.9,
        seed=refine_seed,
        polish=False,
        workers=1,
        updating="immediate",
        tol=1.e-7, 
    )

    return np.asarray(result.x, dtype=float)

#12.Brent边界求根


def exact_occlusion_intervals(
    x: np.ndarray,
    num_points: int = final_num_points,
    scan_step: float = final_time_step,
) -> list[tuple[float, float]]:
    """
    先用 scan_step 找到裕量变号区间，
    再用 brentq 求解 margin(t) = 0。

    支持一段或多段有效遮蔽区间。
    """
    target_points = sample_cylinder(
        num_points
    )

    times = np.arange(
        0.0,
        smoke_valid_time + 0.5 * scan_step,
        scan_step,
    )

    margins = occlusion_margin(
        x=x,
        times_after_burst=times,
        target_points=target_points,
    )

    blocked = margins >= 0.0
    intervals: list[tuple[float, float]] = []

    start: float | None
    start = 0.0 if blocked[0] else None

    for index in range(len(times) - 1):
        if blocked[index] == blocked[index + 1]:
            continue

        left = float(times[index])
        right = float(times[index + 1])

        margin_left = float(margins[index])
        margin_right = float(margins[index + 1])

        if margin_left == 0.0:
            root = left
        elif margin_right == 0.0:
            root = right
        else:
            root = brentq(
                f=lambda t: occlusion_margin_scalar(
                    t_after_burst=t,
                    x=x,
                    target_points=target_points,
                ),
                a=left,
                b=right,
                xtol=root_tolerance,
                rtol=1.e-13,
            )

        if not blocked[index] and blocked[index + 1]:
            start = float(root)
        else:
            if start is None:
                start = 0.0

            intervals.append(
                (start, float(root))
            )
            start = None

    if blocked[-1] and start is not None:
        intervals.append(
            (start, smoke_valid_time)
        )

    return intervals

def exact_total_duration(
    x: np.ndarray,
    num_points: int = final_num_points,
) -> tuple[float, list[tuple[float, float]]]:
    """计算精确有效遮蔽总时长。"""
    intervals = exact_occlusion_intervals(
        x=x,
        num_points=num_points,
    )

    duration = sum(
        end - start
        for start, end in intervals
    )

    return float(duration), intervals


#13.敏感性分析


def sensitivity_analysis(
    x: np.ndarray,
) -> list[tuple[int, float]]:
    """比较不同圆周取点数下的最终遮蔽时长。"""
    results: list[tuple[int, float]] = []

    for num_points in [100, 200, 300, 500, 1000]:
        duration, _ = exact_total_duration(
            x=x,
            num_points=num_points,
        )

        results.append(
            (num_points, duration)
        )

    return results


#14.结果输出


def print_strategy(
    title: str,
    x: np.ndarray,
    num_points: int,
) -> None:
    """输出策略和精确遮蔽结果。"""
    strategy = decode_variables(x)
    print("=" * 60)
    print(title)
    print("=" * 60)
    duration, intervals = exact_total_duration(
        x=x,
        num_points=num_points,
    )

    print(
        f"航向角theta:"
        f"{strategy.theta:.9f} rad"
    )
    print(
        f"航向角："
        f"{np.degrees(strategy.theta):.6f}°"
    )
    print(
        f"无人机速度："
        f"{strategy.speed:.9f} m/s"
    )

    print(
        f"投放时刻："
        f"{strategy.drop_time:.9f} s"
    )
    print(
        f"起爆时刻："
        f"{strategy.burst_time:.9f} s"
    )
    print(
        f"投放至起爆延迟："
        f"{strategy.delay_time:.9f} s"
    )

    print(
        "无人机速度向量：",
        np.array2string(
            strategy.velocity,
            precision=9,
        ),
    )
    print(
        "烟幕弹投放位置：",
        np.array2string(
            strategy.drop_pos,
            precision=9,
        ),
    )
    print(
        "烟幕弹起爆位置：",
        np.array2string(
            strategy.burst_pos,
            precision=9,
        ),
    )

    print("\n起爆后的有效遮蔽区间:")

    if not intervals:
        print("无有效遮蔽区间")
    else:
        for index, (start, end) in enumerate(
            intervals,
            start=1,
        ):
            print(
                f"区间 {index}:"
                f"[{start:.9f}, {end:.9f}] s:"
                f"时长 {end - start:.9f} s"
            )

    print(
        f"\n有效遮蔽总时长:"
        f"{duration:.9f} s"
    )



def main() -> None:
    print("第一阶段:全局差分进化搜索……")
    x_global = run_global_search()

    print_strategy(
        title="第一阶段结果",
        x=x_global,
        num_points=final_num_points,
    )

    print("\n第二阶段:局部精化搜索……")
    x_final = run_refine(x_global)

    print_strategy(
        title="第二阶段最终结果",
        x=x_final,
        num_points=final_num_points,
    )

if __name__ == "__main__":
    main()

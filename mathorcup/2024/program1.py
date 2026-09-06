import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# 变量顺序：
# [x1, x2, x3, x4, y1, y2, y3, y4]

# 最大利润 -> 最小化负利润
c = np.array([
    -2000, -3000, -5000, -6000,
    0, 0, 0, 0
], dtype=float)

M = np.array([24, 17, 12, 7])

A = []
lb = []
ub = []

# 1. 预算约束
A.append([100, 140, 200, 320, 0, 0, 0, 0])
lb.append(-np.inf)
ub.append(2400)

# 2. x_i <= M_i y_i
for i in range(4):
    row = np.zeros(8)
    row[i] = 1
    row[4 + i] = -M[i]

    A.append(row)
    lb.append(-np.inf)
    ub.append(0)

# 3. x_i >= y_i
for i in range(4):
    row = np.zeros(8)
    row[i] = 1
    row[4 + i] = -1

    A.append(row)
    lb.append(0)
    ub.append(np.inf)

# 4. 至少购买 3 种型号
A.append([0, 0, 0, 0, 1, 1, 1, 1])
lb.append(3)
ub.append(np.inf)

A = np.array(A)
constraint = LinearConstraint(A, lb, ub)

# x_i: 0 <= x_i <= M_i
# y_i: 0 <= y_i <= 1
lower = np.zeros(8)
upper = np.concatenate([M, np.ones(4)])

bounds = Bounds(lower, upper)

# 全部设为整数
# 前4个是非负整数，后4个由于上下界[0,1]所以就是0-1变量
integrality = np.ones(8)

result = milp(
    c=c,
    integrality=integrality,
    bounds=bounds,
    constraints=constraint
)

x = np.rint(result.x[:4]).astype(int)
y = np.rint(result.x[4:]).astype(int)

print("购买数量 x =", x)
print("型号选择 y =", y)
print("采购成本 =", np.dot([100, 140, 200, 320], x), "万元")
print("最大利润 =", -result.fun, "万元")
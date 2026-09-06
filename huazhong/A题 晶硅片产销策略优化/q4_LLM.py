import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 一、从Excel读取数据
# ============================================================

file_path = r'C:\Users\Haru\OneDrive\Desktop\math model\huazhong/A题 晶硅片产销策略优化\附件\附件2：2024.1-8生产系统数据 4.14.xlsx'

df_sales = pd.read_excel(file_path, sheet_name='销售收入', header=None)
df_cost_price = pd.read_excel(file_path, sheet_name='耗材价格', header=None)

products = ['N型182.2*183.75*130', 'N型210*210*130', 'N型210*182*130', 'P型210*210*150']

# ============================================================
# 二、9月预测值（来自第二题）
# ============================================================

PRED_9 = {
    '电价': 0.6069,
    '电镀金刚线价格': 9.884,
    '方棒价格': {
        'N型182.2*183.75*130': 55.7,
        'N型210*210*130': 55.7,
        'N型210*182*130': 55.7,
        'P型210*210*150': 45.4
    }
}


# ============================================================
# Q4：外部信息对方棒价格的修正
# ============================================================

S_R = -0.2673 

sigma_rod_N = 16.0868
sigma_rod_P = 14.5055

rod_N_adjusted = 55.7 + S_R * sigma_rod_N
rod_P_adjusted = 45.4 + S_R * sigma_rod_P

PRED_9['方棒价格']['N型182.2*183.75*130'] = rod_N_adjusted
PRED_9['方棒价格']['N型210*210*130'] = rod_N_adjusted
PRED_9['方棒价格']['N型210*182*130'] = rod_N_adjusted
PRED_9['方棒价格']['P型210*210*150'] = rod_P_adjusted

print("\n【Qwen外部信息修正后的方棒价格】")
print(f"N型方棒：{rod_N_adjusted:.4f} 元/kg")
print(f"P型方棒：{rod_P_adjusted:.4f} 元/kg")


# ============================================================
# 三、使用你给的四组单耗数据（全部手动输入）
# ============================================================

# 1. N型182.2*183.75*130
N182_consumption = {
    '单晶方棒': 144.07,
    '电': 400.00,
    '底板胶': 0.16,
    '金刚线粘棒胶水': 0.14,
    '塑料板': 2.25,
    '电镀金刚线': 34.97,
    '水性切割液': 8.11,
    '一体轮': 0.27,
    '过滤袋': 0.45,
    '过氧化氢': 2.13,
    '硅片清洗剂': 6.07,
    '氢氧化钾': 0.04,
    '乳酸': 0.85,
    'POF包装袋': 57.14,
    'PP瓦楞板': 28.57,
    '硅片泡沫箱': 3.57,
    '硫酸纸': 114.00,
    '木托盘': 0.15,
    '珍珠棉垫片': 86.00
}

# 2. N型210*210*130
N210_130_consumption = {
    '单晶方棒': 189.82,
    '电': 536.68,
    '底板胶': 0.18,
    '金刚线粘棒胶水': 0.20,
    '塑料板': 2.56,
    '电镀金刚线': 50.00,
    '水性切割液': 13.94,
    '一体轮': 0.31,
    '过滤袋': 0.51,
    '过氧化氢': 2.26,
    '硅片清洗剂': 7.84,
    '氢氧化钾': 0.05,
    '乳酸': 0.96,
    'POF包装袋': 100.00,
    'PP瓦楞板': 33.33,
    '硅片泡沫箱': 4.17,
    '硫酸纸': 200.00,
    '木托盘': 0.17,
    '珍珠棉垫片': 101.00
}

# 3. N型210*182*130
N210_182_consumption = {
    '单晶方棒': 164.32,
    '电': 520.40,
    '底板胶': 0.16,
    '金刚线粘棒胶水': 0.14,
    '塑料板': 2.25,
    '电镀金刚线': 43.00,
    '水性切割液': 8.11,
    '一体轮': 0.27,
    '过滤袋': 0.45,
    '过氧化氢': 2.13,
    '硅片清洗剂': 6.07,
    '氢氧化钾': 0.04,
    '乳酸': 0.85,
    'POF包装袋': 57.14,
    'PP瓦楞板': 28.57,
    '硅片泡沫箱': 3.57,
    '硫酸纸': 114.00,
    '木托盘': 0.15,
    '珍珠棉垫片': 86.00
}

# 4. P型210*210*150
P210_150_consumption = {
    '单晶方棒': 212.89,
    '电': 520.40,
    '底板胶': 0.16,
    '金刚线粘棒胶水': 0.14,
    '塑料板': 2.25,
    '电镀金刚线': 47.74,
    '水性切割液': 8.11,
    '一体轮': 0.27,
    '过滤袋': 0.45,
    '过氧化氢': 2.26,
    '硅片清洗剂': 6.07,
    '氢氧化钾': 0.05,
    '乳酸': 0.85,
    'POF包装袋': 57.14,
    'PP瓦楞板': 33.33,
    '硅片泡沫箱': 3.57,
    '硫酸纸': 114.00,
    '木托盘': 0.15,
    '珍珠棉垫片': 101.00
}

all_consumption = {
    'N型182.2*183.75*130': N182_consumption,
    'N型210*210*130': N210_130_consumption,
    'N型210*182*130': N210_182_consumption,
    'P型210*210*150': P210_150_consumption
}


# ============================================================
# 四、获取耗材价格（从耗材价格表读取8个月均价）
# ============================================================

def get_material_price(name):
    for row in range(2, 25):
        if df_cost_price.iloc[row, 0] == name:
            values = []
            for col in range(2, 10):
                val = df_cost_price.iloc[row, col]
                if pd.notna(val):
                    values.append(val)
            return np.mean(values) if values else 0
    return 0


# ============================================================
# 五、计算各型号单位变动成本（全部使用你给的数据）
# ============================================================

def calculate_unit_vc_with_given_data():
    unit_vc = []

    for prod_idx, prod in enumerate(products):
        vc = 0
        print(f"\n{'=' * 60}")
        print(f"【{prod}】")
        print(f"{'=' * 60}")

        cons = all_consumption[prod]

        for name, val in cons.items():
            # 跳过在公用成本中核算的
            if name in ['美纹胶带', '无水乙醇', '主辊-涂布', '主辊-开槽']:
                continue

            # 方棒：使用你给的单耗 × 方棒价格
            if name == '单晶方棒':
                price = PRED_9['方棒价格'][prod]
                cost = val * price
                vc += cost
                print(f"   {name}: {val:.4f} kg/万片 × {price:.1f} 元/kg = {cost:.2f} 元/万片")
            else:
                price = get_material_price(name)
                cost = val * price
                vc += cost
                print(f"   {name}: {val:.4f} × {price:.4f} = {cost:.2f}")

        print(f"   合计: {vc:.2f} 元/万片 = {vc / 10000:.4f} 元/片")
        unit_vc.append(vc)

    return unit_vc


unit_vc = calculate_unit_vc_with_given_data()

# ============================================================
# 六、汇总
# ============================================================

print("\n" + "=" * 80)
print("【9月单位变动成本汇总】")
print("=" * 80)
for i, prod in enumerate(products):
    print(f"{prod}: {unit_vc[i]:.0f} 元/万片 = {unit_vc[i] / 10000:.4f} 元/片")


# ============================================================
# 七、提取销量和售价历史数据
# ============================================================

def extract_data_direct():
    months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月']
    sales_start_rows = [3, 10, 17, 24, 31, 38, 45, 52]
    volume_data, price_data = [], []
    for row in sales_start_rows:
        vol_row, price_row = [], []
        for i in range(4):
            vol_row.append(df_sales.iloc[row + i - 1, 4])
            price_row.append(df_sales.iloc[row + i - 1, 3])
        volume_data.append(vol_row)
        price_data.append(price_row)
    df_volume = pd.DataFrame(volume_data, index=months, columns=products)
    df_price = pd.DataFrame(price_data, index=months, columns=products)
    return df_volume, df_price


df_volume, df_price = extract_data_direct()


# ============================================================
# 八、幂函数需求曲线拟合
# ============================================================

def fit_power_demand(df_volume, df_price):
    demand_params = {}
    print("\n" + "=" * 80)
    print("【幂函数需求曲线拟合 Q = a × P^(-b)】")
    print("=" * 80)
    for prod in products:
        P = df_price[prod].values
        Q = df_volume[prod].values
        lnP = np.log(P)
        lnQ = np.log(Q)
        lnP_reshape = lnP.reshape(-1, 1)
        model = LinearRegression()
        model.fit(lnP_reshape, lnQ)
        ln_a = model.intercept_
        b = -model.coef_[0]
        a = np.exp(ln_a)
        demand_params[prod] = {'a': a, 'b': b}

        lnQ_pred = model.predict(lnP_reshape)
        Q_pred = np.exp(lnQ_pred)
        ss_res = np.sum((Q - Q_pred) ** 2)
        ss_tot = np.sum((Q - np.mean(Q)) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"\n{prod}:")
        print(f"  Q = {a:.2f} × P^(-{b:.4f})")
        print(f"  R² = {r2:.4f}")
    return demand_params


demand_params = fit_power_demand(df_volume, df_price)

# ============================================================
# Q4：外部信息对需求水平的修正
# ============================================================

S_D = 0.1944 

sigma_demand = {
    'N型182.2*183.75*130': 0.256660,
    'N型210*210*130': 0.165569,
    'N型210*182*130': 0.262689,
    'P型210*210*150': 0.200973
}

print("\n" + "=" * 80)
print("【Qwen外部信息修正后的需求参数】")
print("=" * 80)

for prod in products:
    old_a = demand_params[prod]['a']

    multiplier = np.exp(
        S_D * sigma_demand[prod]
    )

    new_a = old_a * multiplier

    demand_params[prod]['a'] = new_a

    print(
        f"{prod}: "
        f"a {old_a:.2f} → {new_a:.2f} "
        f"(×{multiplier:.4f})"
    )


# ============================================================
# 九、固定成本
# ============================================================

fixed_costs = {
    '公用成本': 250000,
    '固定人工': 500000,
    '折旧': 1200000,
    '营业税费': 138000,
    '销售费用': 100000,
    '管理费用': 568351,
    '财务费用': 49500,
}
fixed_total = sum(fixed_costs.values())
print(f"\n固定成本合计: {fixed_total:,.0f} 元 ({fixed_total / 10000:.2f} 万元)")

# ============================================================
# 十、68%销量约束 → 反推售价区间
# ============================================================

sales_constraints = {
    'N型182.2*183.75*130': {'lower': 7131987, 'upper': 7736431},
    'N型210*210*130': {'lower': 4201560, 'upper': 4639083},
    'N型210*182*130': {'lower': 5926766, 'upper': 7587525},
    'P型210*210*150': {'lower': 8453811, 'upper': 9149589}
}

print("\n" + "=" * 80)
print("【68%销量约束 → 售价可行区间】")
print("=" * 80)

bounds = []
for i, prod in enumerate(products):
    a_i = demand_params[prod]['a']
    b_i = demand_params[prod]['b']
    L_i = sales_constraints[prod]['lower']
    U_i = sales_constraints[prod]['upper']

    P_lower = (a_i / U_i) ** (1 / b_i)
    P_upper = (a_i / L_i) ** (1 / b_i)

    if P_lower > P_upper:
        P_lower, P_upper = P_upper, P_lower

    bounds.append((P_lower, P_upper))
    print(f"{prod}: [{P_lower:.4f}, {P_upper:.4f}]")


# ============================================================
# 十一、利润函数
# ============================================================

def calculate_profit(Q, P):
    revenue = np.sum(Q * P)
    Q_wan = Q / 10000
    vc_total = np.sum(np.array(unit_vc) * Q_wan)
    silicon_mud = np.sum(Q_wan) * 200
    vc_final = vc_total - silicon_mud
    profit_before_tax = revenue - vc_final - fixed_total
    tax = max(profit_before_tax, 0) * 0.15
    return profit_before_tax - tax


def objective(P):
    Q = np.array([demand_params[prod]['a'] * (P[i] ** (-demand_params[prod]['b']))
                  for i, prod in enumerate(products)])

    for i, prod in enumerate(products):
        if Q[i] < sales_constraints[prod]['lower'] or Q[i] > sales_constraints[prod]['upper']:
            return 1e12

    profit = calculate_profit(Q, P)
    return -profit


# ============================================================
# 十二、优化求解
# ============================================================

print("\n" + "=" * 80)
print("【优化求解：差分进化】")
print("=" * 80)

result = differential_evolution(
    objective,
    bounds=bounds,
    strategy='best1bin',
    maxiter=3000,
    popsize=30,
    tol=1e-8,
    seed=42,
    disp=False
)

P_opt = result.x
Q_opt = np.array([demand_params[prod]['a'] * (P_opt[i] ** (-demand_params[prod]['b']))
                  for i, prod in enumerate(products)])
best_profit = -result.fun

print(f"\n最优利润: {best_profit / 10000:.2f} 万元")
print(f"最优售价: {[f'{p:.4f}' for p in P_opt]}")
print(f"对应销量: {[f'{Q_opt[i] / 10000:.1f}万片' for i in range(4)]}")

# ============================================================
# 十三、约束检查
# ============================================================

print("\n" + "=" * 80)
print("【约束检查】")
print("=" * 80)

all_ok = True
for i, prod in enumerate(products):
    q = Q_opt[i]
    lower = sales_constraints[prod]['lower']
    upper = sales_constraints[prod]['upper']
    q_wan = q / 10000
    if lower <= q <= upper:
        status = "✓"
    else:
        status = f"✗ 超出 [{lower / 10000:.1f}, {upper / 10000:.1f}]"
        all_ok = False
    print(f"{prod}: {q_wan:.1f}万片 {status}")

if all_ok:
    print("\n✅ 所有型号销量均在68%区间内！")

# ============================================================
# 十四、输出结果
# ============================================================

print("\n" + "=" * 80)
print("【最优生产计划与销售策略（9月）】")
print("=" * 80)

print("\n" + "-" * 120)
print(
    f"{'型号':<22} {'销量(万片)':>12} {'销量下限':>12} {'销量上限':>12} {'售价(元/片)':>12} {'单位VC(元/片)':>14} {'销售收入(万元)':>14}")
print("-" * 120)

total_revenue = 0
total_vc = 0

for i, prod in enumerate(products):
    revenue = Q_opt[i] * P_opt[i] / 10000
    vc = unit_vc[i] * (Q_opt[i] / 10000) / 10000
    total_revenue += revenue
    total_vc += vc
    lower = sales_constraints[prod]['lower'] / 10000
    upper = sales_constraints[prod]['upper'] / 10000
    print(
        f"{prod:<22} {Q_opt[i] / 10000:>12.1f} {lower:>12.1f} {upper:>12.1f} {P_opt[i]:>12.4f} {unit_vc[i] / 10000:>14.4f} {revenue:>14.2f}")

print("-" * 120)
print(f"{'合计':<22} {'':>12} {'':>12} {'':>12} {'':>12} {'':>14} {total_revenue:>14.2f}")

silicon_mud = np.sum(Q_opt / 10000) * 200 / 10000
profit_before_tax = best_profit / (1 - 0.15)

print("\n" + "-" * 120)
print(f"变动成本合计: {total_vc:.2f} 万元")
print(f"硅泥收入: {silicon_mud:.2f} 万元")
print(f"固定成本: {fixed_total / 10000:.2f} 万元")
print(f"税前利润: {profit_before_tax:.2f} 万元")
print(f"所得税(15%): {profit_before_tax * 0.15:.2f} 万元")
print(f"净利润: {best_profit :.2f} 元")
print("-" * 120)

# ============================================================
# 十五、保存结果
# ============================================================

with pd.ExcelWriter('问题3_优化结果_最终.xlsx') as writer:
    opt_df = pd.DataFrame({
        '型号': products,
        '最优销量(片)': Q_opt,
        '最优销量(万片)': Q_opt / 10000,
        '销量下限(万片)': [sales_constraints[p]['lower'] / 10000 for p in products],
        '销量上限(万片)': [sales_constraints[p]['upper'] / 10000 for p in products],
        '最优售价(元/片)': P_opt,
        '单位变动成本(元/片)': np.array(unit_vc) / 10000,
        '单位变动成本(元/万片)': unit_vc,
        '销售收入(万元)': Q_opt * P_opt / 10000,
        '变动成本(万元)': np.array(unit_vc) * (Q_opt / 10000) / 10000,
    })
    opt_df.to_excel(writer, sheet_name='最优方案', index=False)

    demand_df = pd.DataFrame({
        '型号': products,
        'a': [demand_params[p]['a'] for p in products],
        'b': [demand_params[p]['b'] for p in products],
    })
    demand_df.to_excel(writer, sheet_name='需求曲线参数', index=False)

print("\n" + "=" * 80)
print("结果已保存到 '问题3_优化结果_最终.xlsx'")
print("=" * 80)
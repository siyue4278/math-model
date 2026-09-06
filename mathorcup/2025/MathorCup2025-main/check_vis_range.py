from pathlib import Path
import numpy as np

root = Path(r"D:\mathorcap\MathorCup2025-main")

idx = 651

true_path = (
    root / "data" / "LocalVal" / "Dataset_1"
    / "Label_file" / f"press_{idx:03d}.npy"
)

pred_path = (
    root / "output" / "Gen_Answer"
    / f"press_{idx:03d}.npy"
)

# 读取
p_true = np.load(true_path).reshape(-1)
p_pred = np.load(pred_path).reshape(-1)

# 和 baseline 的压力处理保持一致
p_true = np.concatenate((p_true[:16], p_true[112:]))

# 绝对误差
abs_error = np.abs(p_pred - p_true)

print("===== Case", idx, "=====")

print("\np_true:")
print("min =", p_true.min())
print("max =", p_true.max())

print("\np_pred:")
print("min =", p_pred.min())
print("max =", p_pred.max())

# 真值和预测共同范围
common_min = min(p_true.min(), p_pred.min())
common_max = max(p_true.max(), p_pred.max())

print("\n共同压力颜色范围:")
print("Min =", common_min)
print("Max =", common_max)

print("\nabs_error:")
print("min =", abs_error.min())
print("max =", abs_error.max())
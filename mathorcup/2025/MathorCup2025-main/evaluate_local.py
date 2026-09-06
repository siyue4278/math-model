from pathlib import Path
import numpy as np

root = Path(r"D:\mathorcap\MathorCup2025-main")

pred_dir = root / "output" / "Gen_Answer"
true_dir = root / "data" / "LocalVal" / "Dataset_1" / "Label_file"

errors = []

for true_file in sorted(true_dir.glob("press_*.npy")):
    pred_file = pred_dir / true_file.name

    if not pred_file.exists():
        print("缺少预测：", pred_file.name)
        continue

    true = np.load(true_file).reshape(-1)
    pred = np.load(pred_file).reshape(-1)

    # 完全按照 baseline 的 load_pressure() 做相同处理
    true = np.concatenate((true[:16], true[112:]))

    if len(true) != len(pred):
        print(
            true_file.name,
            "长度不一致:",
            len(true),
            len(pred),
        )
        continue

    rel_l2 = np.linalg.norm(pred - true) / np.linalg.norm(true)

    errors.append(rel_l2)

    print(
        f"{true_file.name}: "
        f"L2 relative error = {rel_l2:.4f}"
    )

print("\n--------------------")

if errors:
    print("验证车辆数 =", len(errors))
    print("平均 L2 =", np.mean(errors))
    print("中位数 L2 =", np.median(errors))
    print("最好 L2 =", np.min(errors))
    print("最差 L2 =", np.max(errors))
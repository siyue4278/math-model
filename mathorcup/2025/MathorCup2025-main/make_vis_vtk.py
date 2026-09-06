from pathlib import Path
import numpy as np
import meshio

root = Path(r"D:\mathorcap\MathorCup2025-main")

# 选三辆：最好、最差、典型
case_ids = [644, 638, 651]

mesh_dir = root / "data" / "LocalVal" / "Dataset_1" / "Feature_File"
true_dir = root / "data" / "LocalVal" / "Dataset_1" / "Label_file"
pred_dir = root / "output" / "Gen_Answer"
save_dir = root / "output" / "vis_compare"
save_dir.mkdir(parents=True, exist_ok=True)

for idx in case_ids:
    mesh_path = mesh_dir / f"mesh_{idx:03d}.ply"
    true_path = true_dir / f"press_{idx:03d}.npy"
    pred_path = pred_dir / f"press_{idx:03d}.npy"

    mesh = meshio.read(mesh_path)

    true = np.load(true_path).reshape(-1)
    pred = np.load(pred_path).reshape(-1)

    # 与 baseline 保持一致：删去中间96个压力值
    true = np.concatenate((true[:16], true[112:]))

    abs_error = np.abs(pred - true)

    print(f"case {idx}:")
    print("  mesh points =", len(mesh.points))
    print("  true length =", len(true))
    print("  pred length =", len(pred))

    if len(mesh.points) != len(pred):
        raise ValueError(
            f"点数不一致：mesh={len(mesh.points)}, pred={len(pred)} for case {idx}"
        )

    mesh.point_data["p_true"] = true.astype(np.float32)
    mesh.point_data["p_pred"] = pred.astype(np.float32)
    mesh.point_data["abs_error"] = abs_error.astype(np.float32)

    out_path = save_dir / f"compare_{idx:03d}.vtk"
    meshio.write(out_path, mesh)

    print("  saved to:", out_path)

print("\n全部完成。")
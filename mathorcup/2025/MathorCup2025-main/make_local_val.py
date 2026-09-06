from pathlib import Path
import shutil

root = Path(r"D:\mathorcap\MathorCup2025-main")

src = root / "data" / "Training" / "Dataset_1"
dst = root / "data" / "LocalVal" / "Dataset_1"

# 1. 读取 baseline 认可的有效汽车编号
valid_file = src / "watertight_meshes.txt"

valid_ids = [
    int(line.strip())
    for line in valid_file.read_text().splitlines()
    if line.strip()
]

# 2. 再确认 mesh 和真实 pressure 都存在
valid_ids = [
    idx for idx in valid_ids
    if (src / f"mesh_{idx:03d}.ply").exists()
    and (src / f"press_{idx:03d}.npy").exists()
]

print("有效汽车总数：", len(valid_ids))

# 3. 永久拿最后20辆作为验证集
val_ids = valid_ids[-20:]

# 前面的车才允许用于训练
train_pool = valid_ids[:-20]

print("\n固定验证集20辆：")
print(val_ids)

print("\n最多可用于训练的车辆数：", len(train_pool))

# 4. 自动检查未来实验会不会数据泄漏
for n_train in [20, 100, 200, 400]:
    train_ids = valid_ids[:n_train]
    overlap = sorted(set(train_ids) & set(val_ids))

    print(f"\nn_train = {n_train}")
    print("与验证集重叠：", overlap)

    assert len(overlap) == 0, (
        f"发现数据泄漏！n_train={n_train} 与验证集重叠：{overlap}"
    )

print("\n✓ 20 / 100 / 200 / 400 训练方案均无数据泄漏")

# 5. 删除旧 LocalVal，重新创建
if dst.exists():
    shutil.rmtree(dst)

feature_dir = dst / "Feature_File"
label_dir = dst / "Label_file"

feature_dir.mkdir(parents=True)
label_dir.mkdir(parents=True)

# 6. 复制20辆验证车
for idx in val_ids:
    shutil.copy2(
        src / f"mesh_{idx:03d}.ply",
        feature_dir / f"mesh_{idx:03d}.ply",
    )

    shutil.copy2(
        src / f"press_{idx:03d}.npy",
        label_dir / f"press_{idx:03d}.npy",
    )

# 7. 写 baseline 需要的有效编号文件
(dst / "watertight_meshes.txt").write_text(
    "\n".join(f"{idx:03d}" for idx in val_ids)
)

print("\n固定 LocalVal 创建完成：")
print(dst)
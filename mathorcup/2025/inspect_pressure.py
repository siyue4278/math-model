from pathlib import Path
import numpy as np

data_dir = Path(
    r"D:\mathorcap\MathorCup2025-main\data\Training\Dataset_1"
)

# 找 centroid 文件
centroid_files = sorted(data_dir.rglob("centroid_*.npy"))

print("找到 centroid 文件数量：", len(centroid_files))

if centroid_files:
    file = centroid_files[0]
    centroid = np.load(file)

    print("文件：", file)
    print("原始 shape：", centroid.shape)
    print("reshape 后点数：", centroid.reshape(-1, 3).shape)
import numpy as np

# 模型预测
pred = np.load(r"./output/Gen_Answer/press_658.npy").reshape(-1)

# 测试集真实压力
true = np.load(
    r"./data/Test/Dataset_1/Label_file/press_658.npy"
).reshape(-1)

print("原始真实压力数量:", len(true))
print("预测压力数量:", len(pred))

# baseline 中 load_pressure() 使用的同样处理
true = np.concatenate((true[:16], true[112:]))

print("处理后真实压力数量:", len(true))

# L2 相对误差
relative_l2 = np.linalg.norm(pred - true) / np.linalg.norm(true)

print("mesh 658 L2 相对误差 =", relative_l2)

print("真实压力范围:", true.min(), "~", true.max())
print("预测压力范围:", pred.min(), "~", pred.max())
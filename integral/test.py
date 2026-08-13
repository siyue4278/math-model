import matplotlib.pyplot as plt
import matplotlib as mpl

# ==========================================
# 关键设置：告诉 matplotlib 导出真正的文本，而不是线条轮廓
# 对于 SVG 格式：
mpl.rcParams['svg.fonttype'] = 'none' 
# 对于 PDF 格式：
mpl.rcParams['pdf.fonttype'] = 42
# ==========================================

# 1. 正常画图
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')

plt.title("Original Title (You can edit me later!)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 2. 导出为矢量图
# bbox_inches='tight' 保证边缘不会被截断
# transparent=True 可以让背景变透明，方便后期排版
plt.savefig("my_editable_plot.svg", format="svg", bbox_inches='tight', transparent=True)
plt.savefig("my_editable_plot.pdf", format="pdf", bbox_inches='tight', transparent=True)

plt.show()
import sumolib
import matplotlib.pyplot as plt

# 读取路网文件
net = sumolib.net.readNet('E:/ernie/ERNIE-main/simple_nodes.nod.xml')

# 获取路网的边界
xmin, ymin, xmax, ymax = net.getBoundary()

# 创建一个新的图形
fig, ax = plt.subplots(figsize=(10, 10))

# 设置坐标轴范围
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# 绘制每条边（道路）
for edge in net.getEdges():
    # 获取边的形状（一系列坐标点）
    shape = edge.getShape()
    x_coords = [point[0] for point in shape]
    y_coords = [point[1] for point in shape]
    ax.plot(x_coords, y_coords, color='black', linewidth=0.5)

# 保存图像
plt.savefig('network.png', dpi=300)
plt.show()
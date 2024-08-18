import os
import matplotlib.pyplot as plt

# 数据
moment = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
AUC = [0.931, 0.9312, 0.9311, 0.9319, 0.932, 0.9323, 0.9319, 0.9325, 0.9327, 0.9328]

# 创建折线图
plt.plot(moment, AUC, marker='o', linestyle='-', color='b', markersize=6)

# 添加标题和标签
# plt.title('AUC vs. Moment')
plt.xlabel('Moment')
plt.ylabel('AUC')

# 显示网格线
plt.grid(True)

# 检查并创建result文件夹
if not os.path.exists('result'):
    os.mkdir('result')

# 保存图像到result文件夹下
plt.savefig('result/AUC_vs_Moment.png')

# 显示图形
plt.show()

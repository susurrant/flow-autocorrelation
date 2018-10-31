import numpy as np
import math

# 30个点的坐标
points = [(0, 0), (20, 36), (96, 14), (14, 59), (15, 35), (59, 74), (6, 7), (65, 52), (12, 44), (-67, 73), (-23, 0),
          (-61, 68), (-25, 92), (-87, 87), (-81, 11), (-3, 16), (-24, -90), (-31, -50), (-30, -43), (-74, -28),
          (-24, -21), (-6, -30), (-95, -76), (29, -30), (3, -52), (30, -40), (45, -79), (64, -63), (29, -88), (95, -91)]

# 30个点的属性
field = [91, 62, 69, 29, -43, 11, 29, -45, 56, -59, 87, 88, 24, -51, -77, 57, -98, 25, 0, 80, -50, -85, 60, 10, -31, 78,
         99, 51, -42, 98]

# 点的个数：30
count = len(field)

#计算属性平均值
sum = 0.0
for i in range(count):
    sum += field[i]
average = sum / count

#计算权重矩阵
#采用两点之间距离的倒数
w = np.ones((count, count), dtype=float)
for i in range(count):
    for j in range(count):
        if i != j:
            d_ij = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            if d_ij > 0:
                w[i][j] = 1.0 / d_ij

#计算叉积之和
sum1 = 0.0
for i in range(count):
    for j in range(count):
        sum1 += w[i][j] * (field[i] - average) * (field[j] - average)

#计算偏差值平方和
sum2 = 0.0
for i in range(count):
    sum2 += (field[i] - average) ** 2

#计算权重聚合
s = 0.0
for i in range(count):
    for j in range(count):
        s += w[i][j]

Moran_I = count / s * sum1 / sum2

print(Moran_I)

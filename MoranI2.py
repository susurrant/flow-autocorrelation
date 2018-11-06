import numpy as np
import math


class Vector(object):

    def __init__(self, origin, destination, attribute):
        self.origin = origin
        self.destination = destination
        self.attribute = attribute


def get_distance(v1, v2):
    distance = math.sqrt((v1.origin[0] - v2.origin[0]) ** 2 + (v1.origin[1] - v2.origin[1]) ** 2 +
                         (v1.destination[0] - v2.destination[0]) ** 2 + (v1.destination[1] - v2.destination[1]) ** 2)
    return distance


# 模拟流

# 高高聚集，低低聚集
# vectors = [Vector([1, 1], [10, 10], 30), Vector([1, 1], [8, 12], 29), Vector([0, 3], [11, 11], 31),
#           Vector([2, 0], [10, 10], 30), Vector([2, 2], [12, 9], 32), Vector([51, 50], [40, 60], 2),
#           Vector([50, 51], [41, 59], 1), Vector([49, 51], [42, 60], 3), Vector([52, 50], [42, 58], 2)]

# 高低交错
vectors = [Vector([1, 1], [10, 10], 30), Vector([1, 1], [8, 12], 2), Vector([0, 3], [11, 11], 31),
            Vector([2, 0], [10, 10], 3), Vector([2, 2], [12, 9], 32), Vector([51, 50], [40, 60], 2),
            Vector([50, 51], [41, 59], 31), Vector([49, 51], [42, 60], 3), Vector([52, 50], [42, 58], 32)]

# 计算属性平均值
n = len(vectors)
sum = 0.0
for i in range(n):
    sum += vectors[i].attribute
average = sum / n

# 求权重矩阵
w = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        if i == j:
            w[i][j] = 0
        elif i < j:
            d_ij = get_distance(vectors[i], vectors[j])
            if d_ij > 0:
                w[i][j] = 1 / d_ij
            else:
                w[i][j] = 1
        else:
            w[i][j] = w[j][i]

# 计算叉积之和
sum1 = 0.0
for i in range(n):
    for j in range(n):
        sum1 += w[i][j] * (vectors[i].attribute - average) * (vectors[j].attribute - average)

# 计算偏差值平方和
sum2 = 0.0
for i in range(n):
    sum2 += (vectors[i].attribute - average) ** 2

# 计算权重聚合
s = 0.0
for i in range(n):
    for j in range(n):
        s += w[i][j]

moran_i = n * sum1 / s / sum2

print(moran_i)

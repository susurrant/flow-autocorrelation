import numpy as np
import math


class Vector(object):

    def __init__(self, origin, destination, attribute):
        self.origin = origin
        self.destination = destination
        self.attribute = attribute

# 模拟流
def get_sim_flows():
    # 高高聚集，低低聚集
    # vectors = [Vector([1, 1], [10, 10], 30), Vector([1, 1], [8, 12], 29), Vector([0, 3], [11, 11], 31),
    #           Vector([2, 0], [10, 10], 30), Vector([2, 2], [12, 9], 32), Vector([51, 50], [40, 60], 2),
    #           Vector([50, 51], [41, 59], 1), Vector([49, 51], [42, 60], 3), Vector([52, 50], [42, 58], 2)]

    # 高低交错
    vectors = [Vector([1, 1], [10, 10], 30), Vector([1, 1], [8, 12], 2), Vector([0, 3], [11, 11], 31),
                Vector([2, 0], [10, 10], 3), Vector([2, 2], [12, 9], 32), Vector([51, 50], [40, 60], 2),
                Vector([50, 51], [41, 59], 31), Vector([49, 51], [42, 60], 3), Vector([52, 50], [42, 58], 32)]

    return vectors


# 计算流距离（method参数表示选择什么方法计算距离，待补充）
def get_distance(v1, v2, method=0):
    distance = math.sqrt((v1.origin[0] - v2.origin[0]) ** 2 + (v1.origin[1] - v2.origin[1]) ** 2 +
                         (v1.destination[0] - v2.destination[0]) ** 2 + (v1.destination[1] - v2.destination[1]) ** 2)
    return distance


# 计算属性平均值
def get_average(vectors):
    n = len(vectors)
    s = 0.0
    for i in range(n):
        s += vectors[i].attribute
    average = s / n
    return average

# 计算权重矩阵（standardization参数表示是否对权重矩阵标准化）
def get_weighted_matrix(vectors, standardization=False):
    n = len(vectors)
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

    if standardization:
        row_sum = np.sum(w, axis=1).reshape((-1, 1))
        w /= row_sum

    return w


# 计算流的空间自相关指数
def flow_autocorrelation(vectors, standardization=False):
    n = len(vectors)
    average = get_average(vectors)
    w = get_weighted_matrix(vectors)

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

    return moran_i


if __name__ == '__main__':
    flows = get_sim_flows()
    moran_i = flow_autocorrelation(flows)
    print(moran_i)

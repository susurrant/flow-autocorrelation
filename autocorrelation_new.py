# -*- coding: utf-8 -*-：

import numpy as np


# 模拟流
def get_sim_flows():

    vectors_co = np.array([[1, 1, 10, 10], [1, 1, 8, 12], [0, 3, 11, 11], [2, 0, 10, 10], [2, 2, 12, 9],
                           [51, 50, 40, 60], [50, 51, 41, 59], [49, 51, 42, 60], [52, 50, 42, 58]])
    ox = np.array([1,1,0,2,2,51,50,49,52])
    oy = np.array([1,1,3,0,2,50,51,51,50])
    dx = np.array([10,8,11,10,12,40,41,42,42])
    dy = np.array([10,12,11,10,9,60,59,60,58])

    # 高高聚集，低低聚集
    vectors_z = np.array([30, 29, 31, 30, 32, 2, 1, 3, 2])

    # 高低交错
    #vectors_z = np.array([30, 2, 31, 3, 32, 2, 51, 31, 3])

    return ox, oy, dx, dy, vectors_z


# 计算流距离（method参数表示选择什么方法计算距离，待补充）
def get_distance(v1, v2, method=0):
    distance = np.sqrt(np.sum((v1-v2)**2))
    return distance


# 计算权重矩阵（standardization参数表示是否对权重矩阵标准化）
def get_weight_matrix(flows_ox, flows_oy, flows_dx, flows_dy, standardization=False):
    print('calculate weight matrix...')
    print('ox')
    [X, Y] = np.meshgrid(flows_ox, flows_ox)
    w = (X - Y) ** 2
    del X, Y

    print('oy')
    [X, Y] = np.meshgrid(flows_oy, flows_oy)
    w += (X - Y) ** 2
    del X, Y

    print('dx')
    [X, Y] = np.meshgrid(flows_dx, flows_dx)
    w += (X - Y) ** 2
    del X, Y

    print('dx')
    [X, Y] = np.meshgrid(flows_dy, flows_dy)
    w += (X - Y) ** 2
    del X, Y
    
    w = np.sqrt(w)
    w[np.where(w == 0)] = float('inf')
    w = 1/w

    if standardization:
        row_sum = np.sum(w, axis=1).reshape((-1, 1))
        w /= row_sum

    return w


# 计算流的空间自相关指数
def flow_autocorrelation(flows_ox, flows_oy, flows_dx, flows_dy, flows_z, standardization=False):
    print('a')
    n = len(flows_z)
    w = get_weight_matrix(flows_ox, flows_oy, flows_dx, flows_dy)
    dif_z = flows_z-np.average(flows_z)

    # 计算叉积之和
    print('b')
    [X, Y] = np.meshgrid(dif_z, dif_z)
    sum1 = np.sum(X * Y * w)

    # 计算偏差值平方和
    print('c')
    sum2 = np.sum(dif_z**2)

    # 计算权重聚合
    print('d')
    s = np.sum(w)

    moran_i = n * sum1 / s / sum2

    return moran_i


def get_flows_from_file(filename, column_num, minSpeed = 2, maxSpeed = 150):
    si = {}

    with open(filename, 'r') as f:
        f.readline()
        while True:
            line1 = f.readline().strip()
            if line1:
                sl1 = line1.split(',')
                sl2 = f.readline().strip().split(',')
                if sl1[1] == '1' and minSpeed < float(sl1[-2]) < maxSpeed:
                    ogid = int(sl1[-1])
                    dgid = int(sl2[-1])
                    if (ogid, dgid) not in si:
                        si[(ogid, dgid)] = 0
                    si[(ogid, dgid)] += 1
            else:
                break

    flows_ox = []
    flows_oy = []
    flows_dx = []
    flows_dy = []
    flows_z = []
    for k, v in si.items():
        oy, ox, dy, dx = k[0] // column_num, k[0] % column_num, k[1] // column_num, k[1] % column_num
        flows_ox.append(ox)
        flows_oy.append(oy)
        flows_dx.append(dx)
        flows_dy.append(dy)
        flows_z.append(v)

    return np.array(flows_ox), np.array(flows_oy), np.array(flows_dx), np.array(flows_dy), np.array(flows_z)


if __name__ == '__main__':
    flows_ox, flows_oy, flows_dx, flows_dy, flows_z = get_sim_flows()
    #flows_ox, flows_oy, flows_dx, flows_dy, flows_z = get_flows_from_file('./data/sj_051316_1km.csv', 30)
    moran_i = flow_autocorrelation(flows_ox, flows_oy, flows_dx, flows_dy, flows_z)
    print(moran_i)

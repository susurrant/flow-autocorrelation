# -*- coding: utf-8 -*-：

import time
import numpy as np
import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# 模拟数据
def get_sim_flows():

    vectors_co = np.array([[1, 1, 10, 10], [1, 1, 8, 12], [0, 3, 11, 11], [2, 0, 10, 10], [2, 2, 12, 9],
                           [51, 50, 40, 60], [50, 51, 41, 59], [49, 51, 42, 60], [52, 50, 42, 58]])
    # 高高聚集，低低聚集
    vectors_z = np.array([30, 29, 31, 30, 32, 2, 1, 3, 2])

    # 高低交错
    #vectors_z = np.array([30, 2, 31, 3, 32, 2, 51, 31, 3])

    return vectors_co, vectors_z


# 计算流距离（method参数表示选择什么方法计算距离，待补充）
def get_distance(v1, v2, method=0):
    distance = np.sqrt(np.sum((v1-v2)**2))
    return distance


# 计算流的空间自相关指数
def flow_autocorrelation(flows_co, flows_z, standardization=False):
    n = len(flows_z)
    sum1 = 0
    sum_w = 0
    dif_z = flows_z - np.average(flows_z)
    sum2 = np.sum(dif_z ** 2)  # 计算偏差值平方和

    for i in tqdm(range(n)):
        w = 1 / np.sqrt(np.sum((flows_co - flows_co[i]) ** 2, axis=1))  # 计算权重矩阵
        w[i] = 0
        sum1 += np.sum(dif_z[i] * dif_z * w)  # 计算叉积之和
        sum_w += np.sum(w)     # 计算权重聚合

    return n * sum1 / sum_w / sum2


# 读取流数据
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

    flows_co = []
    flows_z = []
    for k, v in si.items():
        oy, ox, dy, dx = k[0] // column_num, k[0] % column_num, k[1] // column_num, k[1] % column_num
        flows_co.append(np.array([ox, oy, dx, dy]))
        flows_z.append(v)

    return np.array(flows_co), np.array(flows_z)


if __name__ == '__main__':
    print('start time: %s' % datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    s_time = time.time()

    # 1. 读取流数据
    #flows_co, flows_z = get_sim_flows()
    flows_co, flows_z = get_flows_from_file('./data/sj_051316_1km.csv', column_num=30)
    # 2. 计算自相关指数
    moran_i = flow_autocorrelation(flows_co, flows_z)

    print('moran\'s I: ' , moran_i)
    print('\nend time: ', datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    print('run time: %.3f secs.' % (time.time() - s_time))
# -*- coding: utf-8 -*-：

import os
import time
from multiprocessing import Pool
import numpy as np
import datetime
from tqdm import tqdm


# 模拟流
def get_sim_flows():
    vectors_co = np.array([[1, 1, 10, 10], [1, 1, 8, 12], [0, 3, 11, 11], [2, 0, 10, 10], [2, 2, 12, 9],
                           [51, 50, 40, 60], [50, 51, 41, 59], [49, 51, 42, 60], [52, 50, 42, 58]])
    # 高高聚集，低低聚集
    vectors_z = np.array([30, 29, 31, 30, 32, 2, 1, 3, 2])

    # 高低交错
    # vectors_z = np.array([30, 2, 31, 3, 32, 2, 51, 31, 3])

    return vectors_co, vectors_z


# 计算流距离（method参数表示选择什么方法计算距离，待补充）
def get_distance(v1, v2, method=0):
    distance = np.sqrt(np.sum((v1 - v2) ** 2))
    return distance


def task(vectors, i, j, n):
    pid = os.getpid()
    print('start process', pid)
    start_time = time.clock()
    l = j - i + 1
    w = np.zeros((l, n), dtype=float)
    for r in range(l):
        if r % 2000 == 0:
            print('process', pid, ':', r, '/', l)
        w[r] = 1 / np.sqrt(np.sum((vectors - vectors[r + i]) ** 2, axis=1))
        w[r, i + r] = 0
    print('process', pid, 'completed: ', '%.3f' % (time.clock() - start_time), 'secs.')

    return w


# 计算权重矩阵（standardization参数表示是否对权重矩阵标准化）
def get_weight_matrix(vectors, num_of_process, standardization=False):
    print('calculate weight matrix...')
    n = len(vectors)
    pool = Pool(processes=num_of_process)
    results = []
    task_allo = [i/num_of_process for i in range(num_of_process+1)]
    for idx in range(num_of_process):
        i = int(task_allo[idx] * n)
        j = int(task_allo[idx + 1] * n)
        print('pid', idx, ':', 'start from', i, 'end in', j)
        results.append(pool.apply_async(task, args=(vectors,i, j, n, )))
    pool.close()
    pool.join()

    return np.vstack((res.get() for res in results))


# 计算流的空间自相关指数
def flow_autocorrelation(flows_co, flows_z, num_of_process, standardization=False):
    n = len(flows_z)
    start_time = time.clock()
    w = get_weight_matrix(flows_co, num_of_process)
    print('compute the weighted matrix: ', '%.3f' % (time.clock() - start_time), 'secs.')


    start_time = time.clock()
    dif_z = flows_z - np.average(flows_z)
    sum1 = 0
    for i in tqdm(range(n)):
        sum1 += np.sum(dif_z[i] * dif_z * w[i])
    print('compute cross product: ', '%.3f' % (time.clock() - start_time), 'secs.')

    print('step c')
    sum2 = np.sum(dif_z**2)

    print('step d')
    s = np.sum(w)

    return n * sum1 / s / sum2


def get_flows_from_file(filename, column_num, minSpeed=2, maxSpeed=150):
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
    print('starting time: \n', datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))

    # flows_co, flows_z = get_sim_flows()
    flows_co, flows_z = get_flows_from_file('./data/sj_051316_1km.csv', 30)
    moran_i = flow_autocorrelation(flows_co, flows_z, num_of_process=5)
    print('moran\'s I: ', moran_i)

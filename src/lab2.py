from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import math


# 获得正态分布的随机点集
def get_random_cluster(mu: np.ndarray, sigma: np.ndarray, num: int) \
        -> np.ndarray:
    dim = mu.shape[0]
    assert mu.shape == (dim,)
    assert sigma.shape == (dim, dim)
    assert num > 0
    x = np.random.randn(num, dim) @ sigma + mu
    assert x.shape == (num, dim)
    return x


# c-mean 聚类
def c_mean(samples: np.ndarray, init_centers: np.ndarray, iter_limit: int = math.inf) \
        -> Tuple[List[np.ndarray], np.ndarray]:
    n, dim = samples.shape
    c = init_centers.shape[0]
    assert init_centers.shape == (c, dim)
    assert iter_limit > 0
    centers = init_centers  # 聚类中心
    labels = np.empty(shape=(n,), dtype=int)  # 每个样本的分类
    iter: int = 1  # 迭代次数
    dist = np.empty(shape=(n, c), dtype=float)  # 每个点到聚类中心的距离
    while iter <= iter_limit:  # 迭代上限
        # 计算到每个聚类中心的距离
        for i in range(c):
            c_center = centers[i, :]
            dist[:, i] = ((samples - c_center) ** 2).sum(axis=1)
        labels = dist.argmin(axis=1)  # 划分新的类别
        new_centers = np.empty(shape=(c, dim))  # 计算新的聚类中心
        for i in range(c):
            new_centers[i, :] = (samples[labels == i]).mean(axis=0)
        # 如果出现某个类没有分到样本,则把聚类中心置为 0
        np.nan_to_num(new_centers, nan=0, copy=False)
        if (new_centers != centers).any():  # 聚类中心有变化,则采用新的中心
            centers = new_centers
        else:  # 聚类中心无变化,则终止迭代
            break
        print(f'\t> iteration {iter}')
        for i in range(c):
            print(f'\t\tcluster {i} center: {centers[i].tolist()}')
        print()
        iter += 1
    return [np.array(samples[labels == i]) for i in range(c)], centers


if __name__ == '__main__':
    sigma = np.array([(1, 0), (0, 1)])
    x1 = get_random_cluster(np.array((2, 2)), sigma, num=500)
    x2 = get_random_cluster(np.array((2, -2)), sigma, num=500)
    x3 = get_random_cluster(np.array((-2, 2)), sigma, num=500)
    x4 = get_random_cluster(np.array((-2, -2)), sigma, num=50)
    x = x1
    # plt.scatter(x1[:, 0], x1[:, 1])
    # plt.scatter(x2[:, 0], x2[:, 1])
    # plt.scatter(x3[:, 0], x3[:, 1])
    x = np.vstack([x1, x2, x3, x4])

    # plt.scatter(x[:, 0], x[:, 1])
    # plt.show()
    clusters, centers = \
        c_mean(x, init_centers=np.array([(0, 10), (3, 2), (-2, -2), (0, 0)]),
               iter_limit=100)
    for cluster in clusters:
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.scatter(centers[:, 0], centers[:, 1], c='black', label='cluster center')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()

    clusters, centers = \
        c_mean(x, init_centers=np.array([(0, 10), (30, 2), (-20, -2), (0, 233)]),
               iter_limit=100)
    for cluster in clusters:
        plt.scatter(cluster[:, 0], cluster[:, 1])
    plt.scatter(centers[:, 0], centers[:, 1], c='black', label='cluster center')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()

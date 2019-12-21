from typing import Tuple, List, Any, Callable, Union
import numpy as np
import matplotlib.pyplot as plt
import math
import random


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

    # 初始聚类中心
    centers = np.array(init_centers, dtype=float)
    # 计算每个样本到每个聚类中心的距离平方
    dist = np.empty(shape=(n, c), dtype=float)
    for i in range(c):
        dist[:, i] = ((samples - centers[i, :]) ** 2).sum(axis=1)
    # 计算初始分类
    labels: np.ndarray = dist.argmin(axis=1)
    # 更新聚类中心
    size = np.array([(labels == i).sum() for i in range(c)])
    for i in range(c):
        if size[i] == 0:
            centers[i, :] = 0
        else:
            centers[i, :] = samples[labels == i].mean(axis=0)
    # 每个类别分到的样本数
    size = np.array([(labels == i).sum() for i in range(c)])

    # 计算 J
    def get_J() -> float:
        return sum((dist[labels == i, i]).sum() for i in range(c))

    # 类进行变化时误差函数的变化
    delta = np.full(shape=(c, c, n), dtype=float, fill_value=np.nan)

    # 更新 delta
    def update_delta(i: int):
        assert 0 <= i < c
        n_i = size[i]
        if n_i == 1:
            # 如果 i 类只有一个元素，则移出不减小 J
            delta[i, :, :n_i] = -(size / (size + 1) * dist[labels == i, :]).T
        elif n_i >= 2:
            delta[i, :, :n_i] = \
                (n_i / (n_i - 1) * dist[labels == i, i:i + 1]
                 - size / (size + 1) * dist[labels == i, :]).T
        delta[i, :, n_i:] = -np.inf  # 阻止移动不存在的元素
        delta[i, i, :] = -np.inf  # 阻止自己向自己移动

    # 初始化 delta
    for i in range(c):
        update_delta(i)

    iter: int = 0
    while iter <= iter_limit:  # 迭代上限
        # 找到最大的 delta
        i_max, j_max, l_max = np.unravel_index(delta.argmax(), delta.shape)
        # 无法使误差平方和再减小
        if delta[i_max, j_max, l_max] <= 0:
            break
        # 寻找要移动的类的位置
        index = 0
        index_in_cluster = 0
        while index < n:
            if labels[index] == i_max:
                if index_in_cluster == l_max:
                    break
                index_in_cluster += 1
            index += 1
        # 移动类
        labels[index] = j_max
        # 更新聚类中心
        assert size[i_max] > 0
        if size[i_max] > 1:
            centers[i_max] += 1 / (size[i_max] - 1) * (centers[i_max] - samples[index])
        else:
            centers[i_max] = 0
        centers[j_max] -= 1 / (size[j_max] + 1) * (centers[j_max] - samples[index])
        # 更新其他数据
        size[i_max] -= 1
        size[j_max] += 1
        dist[:, i_max] = ((samples - centers[i_max]) ** 2).sum(axis=1)
        dist[:, j_max] = ((samples - centers[j_max]) ** 2).sum(axis=1)
        # 更新 delta
        update_delta(i_max)
        update_delta(j_max)

        iter += 1
        print(f'> iteration {iter}: J={get_J()}')

    return [np.array(samples[labels == i]) for i in range(c)], centers


if __name__ == '__main__':
    fig: plt.Figure
    sigma = np.array([(1, 0), (0, 1)])

    # 不同的初始化分
    x1 = get_random_cluster(np.array((2, 1)), np.array([(0.9, 0.1), (0.1, 0.9)]), num=300)
    x2 = get_random_cluster(np.array((2, -2)), np.array([(1, 0), (0, 1)]), num=300)
    x3 = get_random_cluster(np.array((-2, 2)), np.array([(1, 0), (0, 1)]), num=300)
    x4 = get_random_cluster(np.array((-2, -3)), np.array([(0.9, 0.2), (0.2, 0.9)]), num=300)
    x = np.vstack([x1, x2, x3, x4])

    init_centers_list = [
        np.array([(1, 1), (-1, 1)]),
        np.array([(1, 1), (0, 0), (-1, 1)]),
        np.array([(1, 1), (1, -1), (0, 0), (-1, -1), (-1, 1)]),

        np.array([(1, 1), (1, -1), (-1, -1), (-1, 1)]),
        np.array([(1, 0), (-1, 0), (0, 1), (0, -1)]),
        np.array([(2, 2), (2, 2.05), (1, -6), (4, 4)]),
        np.array([(2, 3), (2, 3.1), (1, -6), (4, 4)]),
    ]

    fig, axes = plt.subplots(4, 2)
    # 正确划分
    for x1 in [x1, x2, x3, x4]:
        axes[0, 0].scatter(x1[:, 0], x1[:, 1])
    axes[0, 0].set_title('Actual Cluster')

    for i in range(1, 8):
        ax = axes[i // 2, i % 2]
        init_centers = init_centers_list[i - 1]
        clusters, centers = c_mean(x, init_centers)
        for cluster in clusters:
            ax.scatter(cluster[:, 0], cluster[:, 1])
        ax.scatter(centers[:, 0], centers[:, 1],
                   c='black', label='cluster center')
        ax.legend(loc='best')
        ax.set_title(f'init centers = {init_centers.tolist()}')

    fig.set_size_inches(12, 18)
    # fig.show()
    fig.savefig('../pics/lab2-1.png')

    # 各类分离比较明显与不明显的
    fig, axes = plt.subplots(2, 1)

    x1 = get_random_cluster(np.array((2, 2)), sigma, num=300)
    x2 = get_random_cluster(np.array((2, -2)), sigma, num=300)
    x3 = get_random_cluster(np.array((-2, 2)), sigma, num=300)
    x4 = get_random_cluster(np.array((-2, -2)), sigma, num=300)
    x = np.vstack([x1, x2, x3, x4])
    clusters, centers = c_mean(x, np.array([(1, 0), (-1, 0), (0, 1), (0, -1)]))
    for cluster in clusters:
        axes[0].scatter(cluster[:, 0], cluster[:, 1])
    axes[0].scatter(centers[:, 0], centers[:, 1], color='black', label='cluster center')
    axes[0].legend(loc='best')

    x1 = get_random_cluster(np.array((1, 1)), sigma, num=300)
    x2 = get_random_cluster(np.array((1, -1)), sigma, num=300)
    x3 = get_random_cluster(np.array((-1, 1)), sigma, num=300)
    x4 = get_random_cluster(np.array((-1, -1)), sigma, num=300)
    x = np.vstack([x1, x2, x3, x4])
    clusters, centers = c_mean(x, np.array([(1, 0), (-1, 0), (0, 1), (0, -1)]))
    for cluster in clusters:
        axes[1].scatter(cluster[:, 0], cluster[:, 1])
    axes[1].scatter(centers[:, 0], centers[:, 1], color='black', label='cluster center')
    axes[1].legend(loc='best')

    fig.set_size_inches(6, 9)
    # fig.show()
    fig.savefig('../pics/lab2-2.png')

    # 各类别数目相差很大
    x1 = get_random_cluster(np.array((2, 1)), sigma, num=1000)
    x2 = get_random_cluster(np.array((-2, 1)), sigma, num=100)
    x3 = get_random_cluster(np.array((0, -2)), sigma, num=50)
    x = np.vstack([x1, x2, x3])
    clusters, centers = c_mean(x, np.array([(1, 1), (0, 0), (-1, -1)]))
    # 正确划分
    fig, axes = plt.subplots(2, 1)
    for x in [x1, x2, x3]:
        axes[0].scatter(x[:, 0], x[:, 1], label=f'num={len(x)}')
    axes[0].legend(loc='best')
    axes[0].set_title('Actual Cluster')
    # c-mean 划分
    for cluster in clusters:
        axes[1].scatter(cluster[:, 0], cluster[:, 1])
    axes[1].scatter(centers[:, 0], centers[:, 1], color='black', label='cluster center')
    axes[1].legend(loc='best')
    axes[1].set_title('C-mean')
    fig.set_size_inches(6, 9)
    # fig.show()
    fig.savefig('../pics/lab2-3.png')

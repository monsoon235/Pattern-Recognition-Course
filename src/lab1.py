from typing import List
import math
import numpy as np


def get_w(x1_sample: List[np.ndarray],
          x2_sample: List[np.ndarray],
          learning_rate: float = 1,
          w_init: np.ndarray = None,
          iter_limit: int = math.inf) -> List[np.ndarray]:
    # 所有样本构成增广向量
    sample = [np.append(x, 1) for x in x1_sample] + \
             [np.append(-x, -1) for x in x2_sample]
    # 初始 w
    if w_init is None:
        w_init = np.zeros(shape=sample[0].shape, dtype=float)
    w_list = [w_init]
    iter = 0  # 迭代次数
    w = w_init
    next_iter = True  # 是否继续从第一个样本开始迭代
    while next_iter:
        next_iter = False
        for x in sample:
            if iter >= iter_limit:  # 迭代次数限制
                next_iter = False
                break
            print(f'\t> iteration {iter}')
            print(f'\t\tw={w.tolist()}, x={x.tolist()}, w^T·x={np.dot(w, x)}')
            if np.dot(w, x) <= 0:  # 有样本被分类错误
                w += learning_rate * x  # 更新 w
                next_iter = True  # 有分类错误则要继续迭代所有样本
            w_list.append(w)
            iter += 1
        print()
    return w_list


def test1():
    print('初始 w 为 (0,0,0)')
    s1 = [np.array(x) for x in [(1, 0), (1, 1), (0, 2)]]
    s2 = [np.array(x) for x in [(2, 1), (2, 2), (1, 3)]]
    ws = get_w(s1, s2)
    print(f'最终得到 w = {ws[len(ws) - 1]}\n')


# 改变初始向量
def test2_1():
    print('初始 w 为 (1,1,1)')
    s1 = [np.array(x) for x in [(1, 0), (1, 1), (0, 2)]]
    s2 = [np.array(x) for x in [(2, 1), (2, 2), (1, 3)]]
    w_init = np.ones(shape=(3,))
    ws = get_w(s1, s2, w_init=w_init)
    print(f'最终得到 w = {ws[len(ws) - 1]}\n')


# 改变样本顺序
def test2_2():
    print('改变样本顺序')
    s1 = [np.array(x) for x in [(1, 1), (0, 2), (1, 0)]]
    s2 = [np.array(x) for x in [(1, 3), (2, 2), (2, 1)]]
    ws = get_w(s1, s2)
    print(f'最终得到 w = {ws[len(ws) - 1]}\n')


def test3():
    print('线性不可分的情况')
    s1 = [np.array(x) for x in [(1, 0), (1, 1)]]
    s2 = [np.array(x) for x in [(0, 1), (1, 0)]]
    ws = get_w(s1, s2, iter_limit=100)


if __name__ == '__main__':
    test1()
    test2_1()
    test2_2()
    test3()

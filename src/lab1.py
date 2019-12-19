from typing import List
import math
import numpy as np
import matplotlib.pyplot as plt


def get_w(s1: np.ndarray, s2: np.ndarray,
          learning_rate: float = 1,
          w_init: np.ndarray = None,
          iter_limit: int = math.inf) -> List[np.ndarray]:
    # 所有样本构成增广向量
    sample = np.vstack((
        np.hstack((s1, np.ones(shape=(s1.shape[0], 1)))),
        -np.hstack((s2, np.ones(shape=(s2.shape[0], 1))))
    ))
    # 初始 w
    if w_init is None:
        w_init = np.zeros(shape=sample[0].shape, dtype=float)
    elif w_init.dtype != float:
        w_init = np.array(w_init, dtype=float)
    w_list = [w_init]  # 记录每步迭代之后的 w
    iter = 0  # 迭代次数
    w = w_init
    next_iter = True  # 是否继续从第一个样本开始迭代
    while next_iter:
        next_iter = False
        for x in sample:
            if iter >= iter_limit:  # 迭代次数限制
                next_iter = False
                break
            print(f'\t> iteration {iter + 1}')
            print(f'\t\tw={w.tolist()}, x={x.tolist()}, w^T·x={np.dot(w, x)}')
            if np.dot(w, x) <= 0:  # 有样本被分类错误
                w += learning_rate * x  # 更新 w
                next_iter = True  # 有分类错误则要继续迭代所有样本
            w_list.append(w.copy())
            iter += 1
        print()
    return w_list


def draw(s1: np.ndarray, s2: np.ndarray,
         w_list: List[np.ndarray],  # 绘制多条线
         info_list: List[str] = None,
         title: str = None,
         save_filepath: str = 'a.png'):
    assert len(s1.shape) == 2
    assert len(s2.shape) == 2
    assert s1.shape[1] == 2
    assert s2.shape[1] == 2
    assert all(w.shape == (3,) for w in w_list)
    plt.close()
    # 绘制点
    plt.scatter(s1[:, 0], s1[:, 1], label='label = 1')
    plt.scatter(s2[:, 0], s2[:, 1], label='label = -1')
    # 控制绘图区间
    s = np.vstack((s1, s2))
    x_min, y_min = s.min(axis=0)
    x_max, y_max = s.max(axis=0)
    x_delta, y_delta = x_max - x_min, y_max - y_min
    x_min, x_max = x_min - 0.2 * x_delta, x_max + 0.2 * x_delta
    y_min, y_max = y_min - 0.2 * y_delta, y_max + 0.2 * y_delta
    # 绘制线
    x = np.linspace(x_min, x_max, 100)
    for i, w in enumerate(w_list):
        if w[1] == 0:
            if w[0] == 0:
                x1 = []
                y1 = []
            else:
                x1 = np.full_like(x, fill_value=-w[2] / w[0])
                y1 = np.linspace(y_min, y_max, 100)
        else:
            x1 = x
            y1 = -(w[0] * x + w[2]) / w[1]
        if info_list is None:
            label = f'{w[0]}*x + {w[1]}*y + {w[2]} = 0'
        else:
            label = f'{info_list[i]}: {w[0]}*x + {w[1]}*y + {w[2]} = 0'
        plt.plot(x1, y1, label=label)
    if title is not None:
        plt.title(title)
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
    plt.savefig(save_filepath, dpi=200)


if __name__ == '__main__':
    # 1
    s1 = np.array([(1, 0), (1, 1), (0, 2)])
    s2 = np.array([(2, 1), (2, 2), (1, 3)])
    w1 = get_w(s1, s2)[-1]
    # 2.1 改变初始向量
    w_init = np.array([100, 100, 100])
    w2_1 = get_w(s1, s2, w_init=w_init)[-1]
    # 2.2 改变样本顺序
    s1 = np.array([(1, 1), (0, 2), (1, 0)])
    s2 = np.array([(1, 3), (2, 2), (2, 1)])
    w2_2 = get_w(s1, s2)[-1]

    draw(s1, s2, w_list=[w1, w2_1, w2_2],
         info_list=['init w = [0, 0, 0]',
                    f'init w = {w_init.tolist()}',
                    'change sample order'],
         save_filepath='../pics/lab1-1.png',
         title='Linear Separable Situation')

    # 3
    s1 = np.array([(1, 0), (0, 1)])
    s2 = np.array([(0, 0), (1, 1)])
    w3_list = get_w(s1, s2, iter_limit=100)[-4:]
    draw(s1, s2, w3_list, save_filepath='../pics/lab1-2.png',
         title='Non-linear Separable Situation')

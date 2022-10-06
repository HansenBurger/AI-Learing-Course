import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt


def PCA_numpy(d_m: np.ndarray, k: int) -> list:
    '''
    Downscaling the data matrix to k
    :param: d_m: data matrix(row_n: sample, col_n: feature)
    :param: k: scale set (k <= col_n)
    '''
    r_n, c_n = d_m.shape
    d_m_z = d_m - d_m.mean(axis=0)
    cov_m = 1 / r_n * d_m_z.T.dot(d_m_z)
    eig_v, eig_m = np.linalg.eig(cov_m)
    eig_order = np.flip(np.argsort(eig_v))
    eig_v, eig_m = eig_v[eig_order], eig_m[:, eig_order]
    d_new = d_m.dot(eig_m[:, 0:k])
    ita_k = np.sum(eig_v[0:k]) / np.sum(eig_v) * 100
    return d_new, ita_k


def main():
    iris = datasets.load_iris()
    iris_new, iris_qoi = PCA_numpy(iris.data, 2)
    iris_df = pd.DataFrame(iris_new, columns=['x', 'y'])
    iris_df['tag'] = iris.target
    color_st_s = [{
        'c': 'blue',
        'marker': 'o'
    }, {
        'c': 'red',
        'marker': 's'
    }, {
        'c': 'green',
        'marker': '^'
    }]

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    fig.suptitle('Iris type display after PCA \n (d=2, qoi={0})'.format(
        round(iris_qoi, 2)))
    for i in range(len(color_st_s)):
        cate = np.unique(iris.target)[i]
        iris_cate = iris_df.loc[iris_df['tag'] == cate]
        ax.scatter(iris_cate['x'], iris_cate['y'], label=cate, **color_st_s[i])
    ax.legend()
    fig.tight_layout()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import datasets, decomposition
from matplotlib import pyplot as plt


def PCA_sklearn(d_m: np.ndarray, k: int) -> list:
    '''
    Downscaling the data matrix to k
    :param: d_m: data matrix(row_n: sample, col_n: feature)
    :param: k: scale set (k <= col_n)
    '''
    pca_st = decomposition.PCA(k)
    d_new = pca_st.fit_transform(d_m)
    ita_k = pca_st.explained_variance_ratio_[0]
    return d_new, ita_k


def main():
    iris = datasets.load_iris()
    db_cluste = DBSCAN(eps=0.4, min_samples=9).fit(iris.data)
    iris_label = db_cluste.labels_
    label_error = np.square(iris_label - iris.target).mean()
    iris_new, iris_qoi = PCA_sklearn(iris.data, 2)
    iris_df = pd.DataFrame(iris_new, columns=['x', 'y'])
    iris_df['tag'] = iris_label
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
    fig.suptitle('Iris label error: {0}'.format(round(label_error, 2)))
    for i in range(len(color_st_s)):
        cate = np.unique(iris.target)[i]
        iris_cate = iris_df.loc[iris_df['tag'] == cate]
        ax.scatter(iris_cate['x'], iris_cate['y'], label=cate, **color_st_s[i])
    ax.legend()
    fig.tight_layout()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
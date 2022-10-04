import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt


class Basic():

    def __init__(self) -> None:
        pass


class CPCA(Basic):

    def __init__(self, data_in: np.ndarray) -> None:
        super().__init__()
        self.__data_matrix = data_in
        self.__n, self.__f = data_in.shape
        self.__cov_matrix = np.ndarray([])

    def __ZeroMean(self) -> np.ndarray:
        '''
        Change the axis of the data matrix to the origin
        :return: data matrix after zero-averaging
        '''
        data_m = self.__data_matrix.copy()
        for i in range(self.__f):
            data_m[:, i] = data_m[:, i] - np.mean(data_m[:, i])
        return data_m

    def __CovMatrix_centrial(self) -> None:
        '''
        Calculate the covariance matrix (After zero-averaging)
        '''
        data_m = self.__ZeroMean()
        cov_m = 1 / self.__n * data_m.T.dot(data_m)
        self.__cov_matrix = cov_m

    def __CovMatrix_noncentrial(self) -> None:
        '''
        Calculate the covariance matrix (Without zero-averaging)
        '''
        data_m = self.__data_matrix.copy()
        cov_calculate = lambda X, Y, n: sum((X[i] - np.mean(X)) *
                                            (Y[i] - np.mean(Y))
                                            for i in range(n)) / (n - 1)
        cov_m = np.zeros((self.__f, self.__f))
        for i in range(self.__f):
            for j in range(self.__f):
                cov_m[i, j] = cov_calculate(data_m[:, i], data_m[:, j],
                                            self.__n)
        self.__cov_matrix = cov_m

    def PCA_QOI(self, k: int) -> list:
        '''
        Extracting the data matrix with reduced dimensionality k 
        and the corresponding quantities of information
        :param: k: dimentional selected
        :return: data_new: data matrix after PCA
        :return: ita_k: percentage of QOI 
        '''
        self.__CovMatrix_centrial()
        eig_v, eig_m = np.linalg.eig(self.__cov_matrix)
        data_new = self.__data_matrix.dot(eig_m[:, 0:k])
        ita_k = np.sum(eig_v[0:k]) / np.sum(eig_v) * 100
        return data_new, ita_k


def main():
    iris = datasets.load_iris()
    cpca_p = CPCA(iris.data)
    iris_new, iris_qoi = cpca_p.PCA_QOI(2)
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
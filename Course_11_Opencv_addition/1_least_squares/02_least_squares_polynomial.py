import numpy as np
import seaborn as sns
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


def least_square(x: np.ndarray, y: np.ndarray, m: int) -> tuple:
    '''
    :param: x: array of coordinates value in x-axis
    :param: y: array of coordinates value in y-axis
    :param: m: polynomial order
    :return: theta: polynomial coefficients
    '''
    assert x.shape[0] >= m
    assert x.shape[0] == y.shape[0]

    N = x.shape[0]
    X = np.zeros((N, m + 1))  # turn x to X
    for i in range(N):
        X_i = np.zeros((1, m + 1))  # size of each
        for j in range(m + 1):
            X_i[0][j] = x[i]**(m - j)
        X[i] = X_i
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y.T)
    return theta


def least_square_interface(x: np.ndarray, y: np.ndarray, m: int):

    p_init = np.random.rand(m + 1)
    residuals_func = lambda p, x, y: np.poly1d(p)(x) - y

    p_lsq, _ = leastsq(residuals_func, p_init, args=(x, y))
    return p_lsq


class CoordsGenerator():

    def __init__(self, p_n: int, p_r: tuple, p_f: str, **kwargs) -> None:
        '''
        :param: p_n: number of points to be added
        :param: p_r: range of points to be added
        :param: p_f: function of points to be added
        '''
        self.__x = np.linspace(*p_r, p_n)
        self.__y = self.__func_choose(p_f, **kwargs)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def __sin(self, A: float = 1.0, w: float = 2 * np.pi, phi: float = 0.0):
        return A * np.sin(w * self.__x + phi)

    def __basic(self, k: float = 1.0, b: float = 0.0):
        return k * self.__x + b

    def __func_choose(self, func: str, **kwargs):
        if func == "basic":
            return self.__basic(**kwargs)
        elif func == "sin":
            return self.__sin(**kwargs)

    def add_noise(self, loc: float = 0, var: float = 1):
        noise = np.random.normal(loc, var, self.__x.shape)
        self.__y = self.__y + noise


def main():
    coords = CoordsGenerator(100, (0, 1), "sin")
    coords.add_noise(0, 0.1)
    x = coords.x
    y = coords.y
    m_st = [1, 3, 9]

    coords_refer = CoordsGenerator(10, (0, 1), "sin")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(len(axes)):
        theta = least_square(x, y, m_st[i])
        # theta = least_square_interface(x, y, m_st[i])
        axes[i].plot(coords_refer.x,
                     coords_refer.y,
                     "bo",
                     label="origin(ReSampled)")
        axes[i].plot(
            x,
            np.poly1d(theta)(x),  # Generate fitted polynomial
            "r-",
            linewidth=2.0,
            label="least square")
        axes[i].legend()
        axes[i].set_title("LeastSquare(M = {0})".format(m_st[i]))

    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
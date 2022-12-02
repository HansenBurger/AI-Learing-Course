import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def least_square(arr_0: np.ndarray, arr_1: np.ndarray) -> tuple:
    '''
    arr_0: array of coordinates value in x-axis
    arr_1: array of coordinates value in y-axis
    '''
    k, b = 0, 0
    if arr_0.shape[0] != arr_1.shape[0]:
        return k, b
    else:
        N = arr_0.shape[0]
        p_00 = N * np.dot(arr_0, arr_1.T)
        p_10 = N * np.dot(arr_0, arr_0.T)
        p_01 = np.sum(arr_0) * np.sum(arr_1)
        p_11 = np.sum(arr_0) * np.sum(arr_0)

        k = (p_00 - p_01) / (p_10 - p_11)
        b = np.mean(arr_1) - k * np.mean(arr_0)
        return k, b


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
    coords = CoordsGenerator(100, (0, 1), "basic", k=1.2, b=-1)
    coords.add_noise(0, 0.1)
    x = coords.x
    y = coords.y
    k, b = least_square(x, y)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(x, y, "bo", label="origin")
    ax.plot(x, k * x + b, "r-", linewidth=2.0, label="least square")
    ax.legend()
    ax.set_title("Manual Least Squares")
    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

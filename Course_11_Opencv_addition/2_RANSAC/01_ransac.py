import numpy as np
from scipy.optimize import leastsq
from matplotlib import pyplot as plt


def least_squares(x: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:
    p_init = np.random.rand(m + 1)
    differ_func = lambda p, x, y: np.poly1d(p)(x) - y
    p_lsq, _ = leastsq(differ_func, p_init, args=(x, y))
    return p_lsq


class Basic():

    def __init__(self) -> None:
        pass


class RANSAC(Basic):

    def __init__(self,
                 random_size: int,
                 exp_times: int,
                 inner_differ: float = 0.5,
                 inner_ratio: float = 1.0) -> None:
        super().__init__()
        self.__n = random_size
        self.__k = exp_times
        self.__w = inner_ratio
        self.__th = inner_differ

    def __group_split(self, x: np.ndarray, y: np.ndarray):
        inner_gp_i = np.random.choice(np.arange(x.shape[0]), self.__n)
        outer_gp_i = np.delete(np.arange(x.shape[0]), inner_gp_i)
        inner = np.array([x[inner_gp_i], y[inner_gp_i]])
        outer = np.array([x[outer_gp_i], y[outer_gp_i]])
        return inner, outer

    def main(self, x: np.ndarray, y: np.ndarray):
        inner, outer = self.__group_split(x, y)
        a = 1


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
    true_coords = CoordsGenerator(1000, (-1, 3), "basic", k=-0.8, b=3.5)
    true_coords.add_noise(var=0.1)
    false_coords = CoordsGenerator(300, (-1, 3), "basic", k=-0.1, b=3.5)
    false_coords.add_noise(var=0.2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(true_coords.x, true_coords.y, "ko")
    ax.plot(false_coords.x, false_coords.y, "ro")
    fig.tight_layout()

    x_tot = np.concatenate((true_coords.x, false_coords.x))
    y_tot = np.concatenate((true_coords.y, false_coords.y))
    ransac = RANSAC(50, 100)
    ransac.main(x_tot, y_tot)
    a = 1
    pass


if __name__ == "__main__":
    main()
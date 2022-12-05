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
                 inner_differ: float = 0.5) -> None:
        super().__init__()
        self.__n = random_size
        self.__k = exp_times
        self.__th = inner_differ

    def __group_split(self, x: np.ndarray, y: np.ndarray):
        inner_gp_i = np.random.choice(np.arange(x.shape[0]), self.__n)
        outer_gp_i = np.delete(np.arange(x.shape[0]), inner_gp_i)
        inner = np.array([x[inner_gp_i], y[inner_gp_i]])
        outer = np.array([x[outer_gp_i], y[outer_gp_i]])
        return inner, outer

    def __fit_model(self, func_st: any, **kwargs):
        fit_param = func_st(**kwargs)
        fit_func = np.poly1d(fit_param)
        return fit_func, fit_param

    def __fit_perform(self, y_o: np.ndarray, y_n: np.ndarray):
        count = 0
        for i in range(y_o.shape[0]):
            #TODO Differentiation optimization
            diff = np.abs(y_o[i] - y_n[i])
            if diff > self.__th:
                continue
            else:
                count += 1
        return count

    def main(self, x: np.ndarray, y: np.ndarray, m: int):
        inner, outer = self.__group_split(x, y)
        fit_round, in_size = self.__k, 0
        fit_param_s, fit_size_s = [], []

        while (fit_round):
            fit_tmp, param_tmp = self.__fit_model(least_squares,
                                                  x=inner[0],
                                                  y=inner[1],
                                                  m=m)
            out_tmp = fit_tmp(outer[0])
            in_size_i = self.__fit_perform(outer[1], out_tmp)
            fit_param_s.append(param_tmp)
            fit_size_s.append(in_size_i)

            fit_round -= 1

        fit_param = fit_param_s[np.array(fit_size_s).argmax(axis=0)]

        return np.poly1d(fit_param), fit_param


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
    true_coords = CoordsGenerator(800, (-1, 3), "basic", k=-0.9, b=3.5)
    true_coords.add_noise(var=0.1)
    false_coords = CoordsGenerator(200, (-1, 3), "basic", k=0, b=3.8)
    false_coords.add_noise(var=0.5)

    x_tot = np.concatenate((true_coords.x, false_coords.x))
    y_tot = np.concatenate((true_coords.y, false_coords.y))
    ransac = RANSAC(10, 1000, 0.001)  # threshold must be determined by dataset
    rs_f, rs_p = ransac.main(x_tot, y_tot, 1)
    ls_f = np.poly1d(least_squares(x_tot, y_tot, 1))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(x_tot, y_tot, "ko", label="Samples")
    ax.plot(x_tot, ls_f(x_tot), "r-", linewidth=5.0, label="LeastSquares")
    ax.plot(x_tot, rs_f(x_tot), "g-", linewidth=5.0, label="RANSAC")
    ax.set_title("LeastSquare(R),RANSAC(G)")
    ax.legend()

    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
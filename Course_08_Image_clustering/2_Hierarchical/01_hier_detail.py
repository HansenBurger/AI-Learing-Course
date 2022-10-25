import numpy as np
from numpy import linalg


class Basic():

    def __init__(self) -> None:
        pass


class HierCluster(Basic):

    def __init__(self, method: str = "AverageLink") -> None:
        super().__init__()
        self.__link_way = self.__Method(method)
        self.__arr2multiD = lambda a: np.array([a]) if len(a.shape) < 2 else a
        self.__data_st = None
        self.__data_lb = None

    def __Euclidean(self, arr_0: np.ndarray, arr_1: np.ndarray) -> np.ndarray:
        arr_0, arr_1 = self.__arr2multiD(arr_0), self.__arr2multiD(arr_1)
        dist_result = np.zeros((arr_0.shape[0], arr_1.shape[0]))
        for i in range(arr_0.shape[0]):
            for j in range(arr_1.shape[0]):
                dist_result[i, j] = np.round(linalg.norm(arr_0[i] - arr_1[j]),
                                             3)

        return dist_result.ravel()

    def __Method(self, way_n: str) -> any:
        if way_n == "SingleLink":
            link_way = lambda a0, a1: self.__Euclidean(a0, a1).min()
        elif way_n == "CompleteLink":
            link_way = lambda a0, a1: self.__Euclidean(a0, a1).max()
        elif way_n == "AverageLink":
            link_way = lambda a0, a1: self.__Euclidean(a0, a1).mean()
        return link_way

    def fit(self, d: np.ndarray, k: int):
        total_gp = d.tolist()
        gp_index = np.linspace(0, len(total_gp), len(total_gp),
                               endpoint=False).astype(np.int32)
        loop_rs = []

        while (gp_index.shape[0] != 1):
            gp = [self.__arr2multiD(np.array(total_gp[i])) for i in gp_index]
            sz = gp_index.shape[0]
            dist_rs = np.zeros((sz, sz))

            for i in range(sz):
                for j in range(sz):
                    if i == j:
                        dist_rs[i, j] = -1
                    else:
                        v_i, sz_i = gp[i], gp[i].shape[0]
                        v_j, sz_j = gp[j], gp[j].shape[0]
                        if sz_i == 1 or sz_j == 1:
                            dist_rs[i, j] = self.__Euclidean(v_i, v_j)[0]
                        else:
                            dist_rs[i, j] = self.__link_way(v_i, v_j)

            min_dist = dist_rs.ravel()[np.where(dist_rs.ravel() >= 0)].min()
            ind_0, ind_1 = np.where(dist_rs == min_dist)[0]
            new_gp = np.array([gp[ind_0], gp[ind_1]])
            new_gp = new_gp.reshape((1, new_gp.size, d[0].size))

            total_gp.append(new_gp.tolist())
            gp_index = np.delete(gp_index, [gp_index[ind_0], gp_index[ind_1]])
            gp_index = np.append(gp_index, len(total_gp) - 1)

            loop_rs.append(
                [gp_index[ind_0], gp_index[ind_1], min_dist, new_gp.size])

            a = 1


test_arr = np.array([16.9, 38.5, 39.5, 80.8, 82, 34.6, 116.1])
test_arr = test_arr.reshape(1, test_arr.shape[0], np.array(test_arr[0]).size)
a = test_arr.tolist()


def main():
    main_p = HierCluster()
    main_p.fit(test_arr, 2)


if __name__ == "__main__":
    main()
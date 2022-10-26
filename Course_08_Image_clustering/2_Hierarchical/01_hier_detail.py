import numpy as np
from numpy import linalg
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


class Basic():

    def __init__(self) -> None:
        pass


class HierCluster(Basic):

    def __init__(self, method: str = "AverageLink") -> None:
        super().__init__()
        self.__link_way = self.__Method(method)
        self.__arr2multiD = lambda a: np.array([a]) if len(a.shape) < 2 else a

    def __Euclidean(self, arr_0: np.ndarray, arr_1: np.ndarray) -> np.ndarray:
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

    def __DistMatrix(self, arr_: np.ndarray, m_sz: int) -> np.ndarray:

        dist_m = np.zeros((m_sz, m_sz))

        for i in range(m_sz):
            for j in range(m_sz):
                if i == j:
                    dist_m[i, j] = -1
                else:
                    v_i, sz_i = arr_[i], arr_[i].shape[0]
                    v_j, sz_j = arr_[j], arr_[j].shape[0]
                    if sz_i == 1 or sz_j == 1:
                        dist_m[i, j] = self.__Method('AverageLink')(v_i, v_j)
                    else:
                        dist_m[i, j] = self.__link_way(v_i, v_j)

        return dist_m

    def fit(self, d: np.ndarray, k: int):
        total_gp = d.reshape(d.shape[0], np.array(d[0]).size).tolist()
        gp_ind = np.linspace(0, d.shape[0], d.shape[0],
                             endpoint=False).astype(np.int32)
        loop_rs = []
        gp_inds = [gp_ind]

        while (gp_ind.shape[0] != 1):
            gp = [np.array(total_gp[i]) for i in gp_ind]
            sz = gp_ind.shape[0]
            dist_rs = self.__DistMatrix(gp, sz)

            min_dist = dist_rs.ravel()[np.where(dist_rs.ravel() >= 0)].min()
            ind_0, ind_1 = np.array(np.where(dist_rs == min_dist)).T[0]
            new_gp = np.append(gp[ind_0], gp[ind_1])

            total_gp.append(new_gp.flatten().tolist())
            loop_rs.append(
                [gp_ind[ind_0], gp_ind[ind_1], min_dist, new_gp.size])

            gp_ind = np.delete(gp_ind, [ind_0, ind_1])
            gp_ind = np.append(gp_ind, len(total_gp) - 1)
            gp_inds.append(gp_ind)

        lb_v = np.linspace(0, k, k, endpoint=False).astype(np.int32)
        lb_d = np.zeros((d.shape[0]), dtype=np.int32)
        for i in range(k):
            gp_slt = [np.where(d == j)[0][0] for j in total_gp[gp_inds[-k][i]]]
            for j in gp_slt:
                lb_d[j] = lb_v[i]

        return lb_d, loop_rs


test_arr = np.array([16.9, 38.5, 39.5, 80.8, 82, 34.6, 116.1])


def main():
    main_p = HierCluster()
    label_s, loop_i = main_p.fit(test_arr, 2)
    print(label_s)

    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(loop_i)
    plt.xlabel(
        "Number of points in node (or index of point if no parenthesis).")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
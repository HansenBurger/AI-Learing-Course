import cv2
import numpy as np
from numpy import linalg

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'


class Basic():

    def __init__(self) -> None:
        pass


class K_Means(Basic):

    def __init__(self,
                 k_st: int,
                 stop_rounds_st: int = 10000,
                 tolerance_st: float = 0.0001) -> None:
        super().__init__()
        self.__k = k_st
        self.__st_r = stop_rounds_st
        self.__st_t = tolerance_st
        self.__p_st = None

    def __GetPartial(self, arr_i: np.ndarray):
        if len(self.__p_st.shape) == 1:
            arr_ = self.__p_st[arr_i]
        else:
            arr_ = self.__p_st[arr_i, :]
        return arr_

    def __RandBarycenter(self) -> np.ndarray:
        '''
        Randomly generated barycenter of samples
        :return: p_c: value of barycenters
        '''
        p_c_i = np.random.choice(self.__p_st.shape[0], self.__k, replace=False)
        p_c = self.__GetPartial(p_c_i)

        return p_c

    def __EuclideanDist(self, p_c: np.ndarray) -> np.ndarray:
        '''
        Calculate the Euclidean distance in each group and group by min value
        :param: p_c: value of barycenters
        :return: p_g: grouped results in (len(p_c), unknow)
        :return: p_g_i: grouped index results in (len(p_c), unknow)
        '''
        n_sample, n_group = self.__p_st.shape[0], self.__k
        euc_dist = np.zeros((n_sample, n_group))

        for i in range(n_sample):
            for j in range(n_group):
                euc_dist[i, j] = linalg.norm(p_c[j] - self.__p_st[i])

        dist_map = []
        for i in range(n_sample):
            min_v = euc_dist[i, :].min()
            min_j = lambda x: round(x, 2) - min_v <= 0.01
            map_v = [int(min_j(j)) for j in euc_dist[i, :]]
            dist_map.append(map_v)

        dist_map = np.array(dist_map)

        p_g, p_g_i = [], []

        for k in range(n_group):
            group_index = np.where(dist_map[:, k] == 1)[0]
            group_k = self.__GetPartial(group_index)
            p_g.append(group_k.tolist())
            p_g_i.append(group_index.tolist())

        return p_g, p_g_i

    def __GenBarycenter(self, p_g: np.ndarray) -> np.ndarray:
        '''
        Regenerate the baycenter based on grouping
        :param: p_group: grouping result
        :return: p_c: value of barycenters in (len(p_g), p_st.shape[1])
        '''

        try:
            n_feature, n_group = self.__p_st.shape[1], self.__k
        except:
            n_feature, n_group = 1, self.__k

        p_c = np.zeros((n_group, n_feature))

        for i in range(n_group):
            p_st_partial = np.array(p_g[i])
            for j in range(n_feature):
                try:
                    p_c[i, j] = p_st_partial[:, j].mean()
                except:
                    p_c[i, j] = p_st_partial.mean()

        return p_c

    def fit(self, data_set: np.ndarray) -> np.ndarray:
        n_round = self.__st_r

        self.__p_st = data_set.copy()
        p_c = self.__RandBarycenter()

        while (n_round):
            p_g, _ = self.__EuclideanDist(p_c)
            p_c_new = self.__GenBarycenter(p_g)
            p_c_error = np.square(p_c - p_c_new).mean()

            if p_c_error <= self.__st_t:
                break
            else:
                p_c = p_c_new
                n_round -= 1

        self.__p_st = None
        self.__p_c = p_c

        return p_g


img = cv2.imread(lenna_loc)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# arr = np.array([[0, 0], [1, 2], [3, 1], [8, 8], [9, 10], [10, 7]])
main_p = K_Means(4)
cluster = main_p.fit(gray.ravel())
a = 1

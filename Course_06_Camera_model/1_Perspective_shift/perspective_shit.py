import cv2
import numpy as np
from numpy import linalg

DISTORT_IMG = r'Course_06_Camera_model\StaticStorage\photo.jpg'


class Basic():

    def __init__(self) -> None:
        pass


class PerspectTrans(Basic):
    '''
    Perspective Transmition between 2-dimention matrix
    :param: __p_st_s: points from matrix in, shape in [4, 2]
    :param: __q_st_s: points from matrix out, shape in [4, 2]
    :param: __trans_m: tansmition matrix, shape in [3, 3]
    :func: Transmition: get the transmition coordinate from point
    '''

    def __init__(self, p_st_s: np.ndarray, q_st_s: np.ndarray) -> None:

        super().__init__()
        self.__p_st_s = p_st_s
        self.__q_st_s = q_st_s
        self.__trans_m = self.__GenTransMatrix()

    @property
    def trans_m(self):
        return self.__trans_m

    def __GenTransMatrix(self) -> np.ndarray:
        '''
        Obtain the transmition matrix
        :return: trans_m: transmition matrix, shape in [3, 3]
        '''
        eq_m_left = []
        eq_m_right = self.__q_st_s.ravel().reshape(8, 1)
        for i in range(self.__p_st_s.shape[0]):
            p_x, p_y = self.__p_st_s[i]
            q_x, q_y = self.__q_st_s[i]
            eq_r_0 = [p_x, p_y, 1, 0, 0, 0, -p_x * q_x, -p_y * q_x]
            eq_r_1 = [0, 0, 0, p_x, p_y, 1, -p_x * q_y, -p_y * q_y]
            eq_m_left.append(eq_r_0)
            eq_m_left.append(eq_r_1)
        eq_m_left = np.array(eq_m_left)
        trans_m = np.mat(eq_m_left).I * eq_m_right
        trans_m = np.array(trans_m).T[0]
        trans_m = np.append(trans_m, 1.0)
        trans_m = trans_m.reshape((3, 3))

        return trans_m

    def Transmition_I2O(self, p_in: tuple) -> list:
        '''
        Get the coordinate value of the point (input to output)
        :param: p_in: point's input coordinate
        :return: p_out: point's coordinate after transmition
        '''
        p_in = np.append(np.array(p_in), 1)
        p_in = p_in.reshape((3, 1))
        p_out = self.__trans_m * p_in
        p_out = p_out.reshape((1, 3))
        p_out = [round(i) for i in p_out.ravel()[0:2]]

        return p_out.tolist()

    def Transmition_O2I(self, p_out: tuple) -> tuple:
        '''
        Get the coordinate value of the point (output to input)
        :param: p_out: point's output coordinate
        :return: p_in: point's coordinate before transmition
        '''
        p_out = np.append(np.array(p_out), 1)
        p_out = p_out.reshape((3, 1))
        p_in = linalg.inv(np.mat(self.__trans_m)) * p_out
        p_in = np.array(p_in).astype(np.int32).reshape((1, 3))[0]
        p_in = p_in - p_in[-1]
        p_in = [int(i) for i in p_in.ravel()[0:2]]

        return p_in


class DistortCorrect(Basic):

    def __init__(self, img_loc: str) -> None:
        super().__init__()
        self.__img_r = cv2.imread(img_loc)
        self.__img_p = self.__PreProcessing()
        self.__cnt, self.__vtx = self.__ContourDetection()
        self.__trans_m = self.__GenTransM()

    def __PreProcessing(self) -> np.ndarray:
        '''
        Preprocess for len's distortion correction
        1. Image graying
        2. Gaussian smoothing
        3. Rectangular dilation
        :param: img_loc: image location
        :return: img: mage matrix after processing
        '''
        img = self.__img_r.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.dilate(img,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return img

    def __ContourDetection(self) -> list:
        '''
        Detect the edge of the image and approximate it with polygon
        + Since findContours only detects external contours, multiple contours 
        + may be caused by setting the canny threshold too small or too large, 
        + which needs to be adjusted dynamically.
        
        1. Enlarge the low threshold if the contours are too many 
        2. Polygon fitting from outside to inside until the first quad appears
        3. Narrowing the low threshold if the fitting condition is not satisfied
        '''
        img = self.__img_p.copy()
        th_qua = np.quantile(img.ravel(), 0.25)
        th_tqua = np.quantile(img.ravel(), 0.75)

        th_low, th_high = th_qua, th_tqua
        cnt_outer = np.array([])

        while (cnt_outer.size == 0):

            edge = cv2.Canny(img, th_low, th_high, apertureSize=3)
            k_delita = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edge = cv2.dilate(edge, k_delita)
            try:
                _, cnt_s, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            except:
                cnt_s, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            cnt_s = sorted(cnt_s, key=cv2.contourArea, reverse=True)

            if len(cnt_s) >= 10:
                th_low = round(th_low * 1.1)
                continue

            for cnt in cnt_s:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if not len(approx) == 4:
                    continue
                else:
                    cnt_outer = cnt
                    cnt_vertex = approx
                    break

            th_low = round(th_low * 0.8)

        return cnt_outer, cnt_vertex

    def __GenTransM(self):
        '''
        
        '''
        dist_c = lambda l_0, l_1, x: linalg.norm(l_0[x] - l_1[x])
        p_v, p_v_r, = self.__vtx, np.roll(self.__vtx, 2)
        p_l_s = list(dist_c(p_v, p_v_r, i) for i in range(4))
        side_0 = round(np.mean([p_l_s[0], p_l_s[2]]))
        side_1 = round(np.mean([p_l_s[1], p_l_s[3]]))
        q_v = np.array([[0, 0], [0, side_1], [side_0, side_1], [side_0, 0]])

        img_trans = np.zeros(shape=(side_1, side_0, self.__img_r.shape[2]),
                             dtype=self.__img_r.dtype)
        trans_p = PerspectTrans(p_v.reshape(4, 2), q_v.reshape(4, 2))

        return trans_p.trans_m
        # p_i_max = self.__img_r.shape[0] - 1
        # p_j_max = self.__img_r.shape[1] - 1

        # for i in range(img_trans.shape[0]):
        #     for j in range(img_trans.shape[1]):
        #         p_i, p_j = trans_p.Transmition_O2I((i, j))
        #         p_i = p_i_max if p_i > p_i_max else p_i
        #         p_j = p_j_max if p_j > p_j_max else p_j
        #         img_trans[i, j, :] = self.__img_r[p_i, p_j, :]

        # img_2 = cv2.warpPerspective(self.__img_r, trans_p.trans_m, (347, 488))

        # cv2.imshow('img_1', img_trans)
        # cv2.imshow('img_2', img_2)
        # cv2.waitKey(0)

        return img_trans


if __name__ == "__main__":
    main_p = DistortCorrect(DISTORT_IMG)

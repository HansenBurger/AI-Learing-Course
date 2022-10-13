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
    :param: __p_st_s: points from matrix in, shape in [1, 4] 
    :param: __q_st_s: points from matrix out, shape in [1, 4]
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
        eq_m_right = self.__q_st_s.ravel()
        for i in range(self.__p_st_s.shape[1]):
            p_x, p_y, q_x, q_y = self.__p_st_s[i], self.__q_st_s[i]
            eq_r_0 = [p_x, p_y, 1, 0, 0, 0, -p_x * q_x, -p_y * q_x]
            eq_r_1 = [0, 0, 0, p_x, p_y, 1, -p_x * q_y, -p_y * q_y]
            eq_m_left.append(eq_r_0, eq_r_1)
        eq_m_left = np.array(eq_m_left)
        trans_m = eq_m_right * linalg.inv(eq_m_left)
        trans_m = np.append(trans_m, 1)
        trans_m = trans_m.transpose(3, 3)

        return trans_m

    def Transmition(self, p_in: tuple) -> tuple:
        '''
        Get the coordinate value of the point after transmition
        :param: p_in: point's origin coordinate
        :return: p_out: point's coordinate after transmition
        '''
        p_in = np.append(np.array(p_in), 1)
        p_in = p_in.transpose(3, 1)
        p_out = self.__trans_m * p_in
        p_out = p_out.transpose(1, 3)[0:2]
        p_out = tuple(map(tuple, p_out))

        return p_out


class DistortCorrect(Basic):

    def __init__(self, img_loc: str) -> None:
        super().__init__()
        self.__img = cv2.imread(img_loc)
        self.__img_canny = None

    def __PreProcessing(self) -> np.ndarray:
        img = self.__img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        pass

    def __EdgeDetection(self):
        img = self.__img.copy()
        img = cv2
        pass
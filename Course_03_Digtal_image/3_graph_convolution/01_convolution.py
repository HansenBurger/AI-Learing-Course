import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

img_loc = "Course_03_Digtal_image\StaticStorage\lenna.png"

k_mean = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9],
                   [1 / 9, 1 / 9, 1 / 9]])
k_gauss = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8],
                    [1 / 16, 1 / 8, 1 / 16]])
k_laplace_0 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
k_laplace_1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
k_soble_0 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
k_soble_1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


class Basic:

    def __init__(self) -> None:
        pass


class Convolution(Basic):
    '''
    convolution for matrix (single & multiple)
    :param: p_st: padding setting
    :param: s_st: stride setting
    '''

    def __init__(self, p_st: int, s_st: int) -> None:
        super().__init__()
        self.__p_st = p_st
        self.__s_st = s_st

    def __EmptyMatrix(self, h: int, w: int, f: int):
        '''
        creat zeros matrix save convolution results
        :param: h: input shape 0
        :param: w: input shape 1
        :param: f: filter shape 0 or shape 1
        '''
        h_new = np.around((h - f + 2 * self.__p_st) / self.__s_st + 1)
        w_new = np.around((w - f + 2 * self.__p_st) / self.__s_st + 1)
        matrix_new = np.zeros(shape=(h_new, w_new, c), dtype=np.int8)

        return matrix_new

    def __AddPadding(self, matrix_in: np.ndarray):
        '''
        add padding in matrix
        :param:  
        :param:
        '''
        pass
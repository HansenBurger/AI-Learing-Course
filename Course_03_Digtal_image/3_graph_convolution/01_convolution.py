import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

img_loc = "Course_03_Digtal_image\StaticStorage\lenna.png"

k_itself = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
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

    def __EmptyMatrix(self, m_in: np.ndarray, f: int):
        '''
        creat zeros matrix save convolution results
        :param: m_in: matrix for padding
        :param: f: filter shape 0 or shape 1
        :return: m_new: zeros matrix for results adding
        '''
        h, w = m_in.shape
        h_new = np.around((h - f + 2 * self.__p_st) / self.__s_st + 1)
        w_new = np.around((w - f + 2 * self.__p_st) / self.__s_st + 1)
        m_new = np.zeros(shape=(int(h_new), int(w_new)), dtype=m_in.dtype)

        return m_new

    def __AddPadding(self, m_in: np.ndarray):
        '''
        add padding in matrix
        :param: m_in: matrix for padding
        '''
        h_new = m_in.shape[0] + 2 * self.__p_st
        w_new = m_in.shape[1] + 2 * self.__p_st
        m_pad = np.zeros(shape=(h_new, w_new), dtype=m_in.dtype)
        for i in range(m_pad.shape[0]):
            for j in range(m_pad.shape[1]):
                if i < self.__p_st or j < self.__p_st:
                    continue
                elif i > m_in.shape[0] or j > m_in.shape[0]:
                    continue
                else:
                    m_pad[i, j] = m_in[i - self.__p_st, j - self.__p_st]

        return m_pad

    def ConvFilt(self, base: np.ndarray, filter_: np.ndarray):
        '''
        main process of convolution
        :param: base: input volume
        :param: filter: convolution kernal
        '''

        f = filter_.shape[0]
        base_pad = self.__AddPadding(base)
        conv_out = self.__EmptyMatrix(base, f)
        for i in range(conv_out.shape[0]):
            for j in range(conv_out.shape[1]):
                i_ = i + (f - 1) / 2 + self.__s_st - 1  # coord in base pad
                j_ = j + (f - 1) / 2 + self.__s_st - 1  # coord in base pad
                m_p = base_pad[int(i_ - (f - 1) / 2):int(i_ + (f + 1) / 2),
                               int(j_ - (f - 1) / 2):int(j_ + (f + 1) / 2)]
                conv_v = np.sum((m_p * filter_).ravel())
                conv_out[i, j] = int(round(conv_v))

        return conv_out


class ImageFiltering(Basic):

    def __init__(self, img_loc: str) -> None:
        super().__init__()
        self.__img = cv2.imread(img_loc)
        self.__h, self.__w, self.__c = self.__img.shape  # gray img still got multi channel

    def __ImgFilt(self,
                  filter_: np.ndarray,
                  p_st: int,
                  s_st: int,
                  img: np.ndarray = np.array([])):
        img_new_s = []
        for c in range(self.__c):
            img_c = self.__img[:, :, c] if img.size == 0 else img[:, :, c]
            img_new_s.append(Convolution(p_st, s_st).ConvFilt(img_c, filter_))
        img_new = np.dstack(img_new_s)
        return img_new

    def ImgItself(self, p_st: int = 1, s_st: int = 1):
        img_ = self.__ImgFilt(k_itself, p_st, s_st)
        return img_

    def ImgSmooth_mean(self, p_st: int = 1, s_st: int = 1):
        img_ = self.__ImgFilt(k_mean, p_st, s_st)
        return img_

    def ImgSmooth_gauss(self, p_st: int = 1, s_st: int = 1):
        img_ = self.__ImgFilt(k_gauss, p_st, s_st)
        return img_

    def ImgSharpening(self, p_st: int = 1, s_st: int = 1):
        img_ = self.__ImgFilt(k_laplace_0, p_st, s_st)
        return img_

    def ImgSoble_hor(self, p_st: int = 1, s_st: int = 1):
        img_ = self.__ImgFilt(k_soble_0, p_st, s_st)
        return img_

    def ImgSoble_ver(self, p_st: int = 1, s_st: int = 1):
        img_ = self.__ImgFilt(k_soble_1, p_st, s_st)
        return img_


class ConvCalculation(Basic):

    def __init__(self) -> None:
        super().__init__()


def ImgProcessDisplay(img: np.ndarray, ax: any, title: str):
    ax.axis("off")
    ax.set_title(title)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    filt_p = ImageFiltering(img_loc)
    fig, ((ax_0_0, ax_0_1, ax_0_2), (ax_1_0, ax_1_1,
                                     ax_1_2)) = plt.subplots(2, 3)
    ImgProcessDisplay(filt_p.ImgItself(), ax_0_0, 'Origin')
    ImgProcessDisplay(filt_p.ImgSmooth_mean(), ax_0_1, 'Mean')
    ImgProcessDisplay(filt_p.ImgSmooth_gauss(), ax_0_2, 'Gauss')
    ImgProcessDisplay(filt_p.ImgSharpening(), ax_1_0, 'Laplace')
    ImgProcessDisplay(filt_p.ImgSoble_hor(), ax_1_1, 'Soble(h)')
    ImgProcessDisplay(filt_p.ImgSoble_ver(), ax_1_2, 'Soble(v)')

    fig.suptitle("Image Filtering")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

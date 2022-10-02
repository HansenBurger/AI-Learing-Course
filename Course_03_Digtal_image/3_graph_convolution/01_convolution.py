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

test_volume = np.array([[[1, 1, 0], [0, 0, 2], [2, 1, 2], [2, 0, 0], [2, 1,
                                                                      0]],
                        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1,
                                                                      1]],
                        [[2, 0, 1], [0, 0, 1], [2, 2, 2], [2, 0, 0], [2, 0,
                                                                      2]],
                        [[1, 2, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0,
                                                                      0]],
                        [[1, 0, 0], [0, 0, 1], [0, 1, 1], [2, 0, 0], [1, 0,
                                                                      1]]])

filter_0_w = np.array([[[1, -1, 1], [0, -1, 1], [0, 0, 0]],
                       [[-1, -1, 0], [0, -1, -1], [0, 1, 1]],
                       [[0, 1, 1], [-1, 0, 1], [1, 0, 1]]])

filter_0_b = 1

filter_1_w = np.array([[[1, 0, 0], [-1, 1, -1], [0, -1, 0]],
                       [[-1, 0, 1], [0, 0, 1], [0, 0, -1]],
                       [[0, 0, -1], [1, 1, 0], [-1, 1, 1]]])

filter_1_b = 0


class Basic:

    def __init__(self) -> None:
        pass


class Convolution(Basic):
    '''
    convolution for matrix (single) -same mode
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
                i_ = (f - 1) / 2 + self.__s_st * i  # coord in base pad
                j_ = (f - 1) / 2 + self.__s_st * j  # coord in base pad
                m_p = base_pad[int(i_ - (f - 1) / 2):int(i_ + (f + 1) / 2),
                               int(j_ - (f - 1) / 2):int(j_ + (f + 1) / 2)]
                conv_v = np.sum((m_p * filter_).ravel())
                conv_out[i, j] = int(round(conv_v))

        return conv_out


class ImageFiltering(Basic):

    def __init__(self, img_loc: str) -> None:
        super().__init__()
        self.__img = cv2.imread(img_loc)
        _, _, self.__c = self.__img.shape  # gray img still got multiple channel

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


class MatrixCalculation(Basic):

    def __init__(self, matrix_in: np.ndarray, p_st: int, s_st: int) -> None:
        super().__init__()
        self.__m_in = matrix_in
        self.__conv = Convolution(p_st, s_st)

    def Convlution(self, w_in: np.ndarray, b_in: np.ndarray):
        '''
        Calculation for convlution of matrixes
        :param: w_in: kernal (num of channel must be equal)
        :param: b_in: bias of the kernal
        :return: conv_out: results of convlution
        '''
        if len(self.__m_in.shape) != len(w_in.shape):
            return
        else:
            if len(self.__m_in.shape) == 2:
                conv_out = self.__conv.ConvFilt(self.__m_in, w_in) + b_in
            else:
                conv_out_s = [
                    self.__conv.ConvFilt(self.__m_in[:, :, c], w_in[:, :, c])
                    for c in range(self.__m_in.shape[2])
                ]
                conv_out = np.zeros(conv_out_s[0].shape,
                                    dtype=conv_out_s[0].dtype)
                for i in conv_out_s:
                    conv_out += i
                conv_out += b_in
        return conv_out


def ImgProcessDisplay(img: np.ndarray, ax: any, title: str):
    ax.axis("off")
    ax.set_title(title)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    filt_p = ImageFiltering(img_loc)
    fig_0, ((ax_0_0, ax_0_1, ax_0_2), (ax_1_0, ax_1_1,
                                       ax_1_2)) = plt.subplots(2, 3, figsize=(12, 8))
    fig_0.suptitle("Image Filtering")

    ImgProcessDisplay(filt_p.ImgItself(), ax_0_0, 'Origin')
    ImgProcessDisplay(filt_p.ImgSmooth_mean(), ax_0_1, 'Mean')
    ImgProcessDisplay(filt_p.ImgSmooth_gauss(), ax_0_2, 'Gauss')
    ImgProcessDisplay(filt_p.ImgSharpening(), ax_1_0, 'Laplace')
    ImgProcessDisplay(filt_p.ImgSoble_hor(), ax_1_1, 'Soble(h)')
    ImgProcessDisplay(filt_p.ImgSoble_ver(), ax_1_2, 'Soble(v)')

    fig_0.tight_layout()

    fig_1, ax_es = plt.subplots(nrows=4, ncols=1, figsize=(9, 12))
    fig_1.suptitle('Matrix Convlution (pad=1, stride=2)')

    conv_cal = MatrixCalculation(test_volume, p_st=1, s_st=2)
    conv_out_0 = conv_cal.Convlution(filter_0_w, filter_0_b)
    conv_out_1 = conv_cal.Convlution(filter_1_w, filter_1_b)

    for ax in ax_es:
        ax.remove()

    gridspec = ax_es[0].get_subplotspec().get_gridspec()
    subfigs = [fig_1.add_subfigure(gs) for gs in gridspec]

    subfigs[0].suptitle('Input volume')
    ax_row_0 = subfigs[0].subplots(nrows=1, ncols=3)

    for i in range(len(ax_row_0)):
        sns.heatmap(test_volume[:, :, i],
                    linewidths=0.2,
                    annot=True,
                    cbar=False,
                    ax=ax_row_0[i],
                    cmap='rocket')

    subfigs[1].suptitle('Filter w0, b0 = {0}'.format(filter_0_b))
    ax_row_1 = subfigs[1].subplots(nrows=1, ncols=3)

    for i in range(len(ax_row_1)):
        sns.heatmap(filter_0_w[:, :, i],
                    linewidths=0.2,
                    annot=True,
                    cbar=False,
                    ax=ax_row_1[i],
                    cmap='YlOrBr')

    subfigs[2].suptitle('Filter w1, b1 = {0}'.format(filter_1_b))
    ax_row_2 = subfigs[2].subplots(nrows=1, ncols=3)

    for i in range(len(ax_row_2)):
        sns.heatmap(filter_1_w[:, :, i],
                    linewidths=0.2,
                    annot=True,
                    cbar=False,
                    ax=ax_row_2[i],
                    cmap='YlOrBr')

    subfigs[3].suptitle('Output Volume')
    ax_row_3 = subfigs[3].subplots(nrows=1, ncols=2)
    sns.heatmap(conv_out_0,
                linewidths=0.2,
                annot=True,
                cbar=False,
                ax=ax_row_3[0],
                cmap='Blues')
    sns.heatmap(conv_out_1,
                linewidths=0.2,
                annot=True,
                cbar=False,
                ax=ax_row_3[1],
                cmap='Blues')

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

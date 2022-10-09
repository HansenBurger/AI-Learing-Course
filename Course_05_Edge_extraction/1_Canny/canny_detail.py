import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

Gauss_kernal = 1 / 273 * np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4],
                                   [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]])

Prewitt_kernal_s = [
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
]

Sobel_kernal_s = [
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
]

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'


class Basic():

    def __init__(self) -> None:
        pass


class Canny(Basic):

    def __init__(self,
                 low_limit_st: int = 0,
                 high_limit_st: int = 100,
                 smooth_kernal: np.ndarray = Gauss_kernal,
                 hor_detect_kernal: np.ndarray = Sobel_kernal_s[0],
                 ver_detect_kernal: np.ndarray = Sobel_kernal_s[1]) -> None:
        super().__init__()
        '''
        Init a canny processor for image edging detection
        :param: low_limit_st: low limit for two threshold detection
        :param: high_limit_st: high limit for two threshold detection
        :param: smooth_kernal: Kernel function for smoothing
        '''
        self.__l_th = low_limit_st
        self.__h_th = high_limit_st
        self.__s_k = smooth_kernal
        self.__h_d_k = hor_detect_kernal
        self.__v_d_k = ver_detect_kernal

    def __ValidationGraying(self, img_loc: str) -> np.ndarray:
        '''
        Validte whether image existing and gray it.
        :param: img_loc: image location
        :return: img: image graying scale in matrix
        '''
        if not Path(img_loc).is_file():
            print('Img not exists !')
            img = np.ndarray([])
        else:
            try:
                img = cv2.imread(img_loc)
                img = rgb2gray(img)
            except:
                img = np.ndarray([])
        return img

    def __ImgSmoothing(self, img: np.ndarray) -> np.ndarray:
        '''
        Smooth a grayed-out image
        :param: img: image grayed in matrix
        :return: img_smooth: image after smoothing
        '''
        img_smooth = ndimage.convolve(img, self.__s_k, mode='constant', cval=0)
        return img_smooth

    def __HorVerDetection(self, img: np.ndarray) -> tuple:
        '''
        Detect horizontal and vertical edges of images
        :param: img: image input in matrix
        :return: img_hv: image after horizontal and vertical edge detecting
        '''
        cov_x = ndimage.convolve(img, self.__h_d_k, mode='constant', cval=0)
        cov_y = ndimage.convolve(img, self.__v_d_k, mode='constant', cval=0)
        grad_v = np.hypot(cov_x, cov_y)
        # Normalized and enlarged for easy threshold screening
        grad_v = grad_v / grad_v.max() * 255
        grad_t = np.arctan(cov_y, cov_x)
        return grad_v, grad_t

    def __NonMaxSuppression(self, G_v: np.ndarray,
                            G_t: np.ndarray) -> np.ndarray:
        '''
        Finding local maxima and suppressing non-maxima
        :param: G_v: gradient magnitude matrix
        :param: G_t: gradient direction matrix (curvature)
        :return: g_v_out: gradient magnitude after NMS
        '''
        g_v_out = G_v.copy()
        g_angle = G_t / np.pi * 180  # Radian to angle conversion

        w_count = lambda angle: np.abs(np.tan(angle * np.pi / 180))
        interp_v = lambda x, y, w: w * x + (1 - w) * y

        for i in range(1, g_v_out.shape[0] - 1):
            for j in range(1, g_v_out.shape[1] - 1):
                g_c = g_v_out[i, j]
                g_a = g_angle[i, j]

                if 0 <= g_a < 45 or -180 <= g_a < -135:
                    p_0, p_1 = g_v_out[i + 1, j - 1], g_v_out[i, j - 1]
                    p_2, p_3 = g_v_out[i - 1, j + 1], g_v_out[i, j + 1]
                    g_t_0 = interp_v(p_0, p_1, w_count(g_a))
                    g_t_1 = interp_v(p_2, p_3, w_count(g_a))
                elif 45 <= g_a < 90 or -135 <= g_a < -90:
                    p_0, p_1 = g_v_out[i + 1, j - 1], g_v_out[i + 1, j]
                    p_2, p_3 = g_v_out[i - 1, j + 1], g_v_out[i - 1, j]
                    g_t_0 = interp_v(p_0, p_1, 1 / w_count(g_a))
                    g_t_1 = interp_v(p_2, p_3, 1 / w_count(g_a))
                elif 90 <= g_a < 135 or -90 <= g_a < -45:
                    p_0, p_1 = g_v_out[i + 1, j + 1], g_v_out[i + 1, j]
                    p_2, p_3 = g_v_out[i - 1, j - 1], g_v_out[i - 1, j]
                    g_t_0 = interp_v(p_0, p_1, 1 / w_count(g_a))
                    g_t_1 = interp_v(p_2, p_3, 1 / w_count(g_a))
                elif 135 <= g_a < 180 or -45 <= g_a < 0:
                    p_0, p_1 = g_v_out[i + 1, j + 1], g_v_out[i, j + 1]
                    p_2, p_3 = g_v_out[i - 1, j - 1], g_v_out[i, j - 1]
                    g_t_0 = interp_v(p_0, p_1, w_count(g_a))
                    g_t_1 = interp_v(p_2, p_3, w_count(g_a))

                if g_c < g_t_0 or g_c < g_t_1:
                    g_v_out[i, j] = 0
                else:
                    continue

        return g_v_out

    def __TwoThresholdDetection(self, img: np.ndarray) -> np.ndarray:
        '''
        Distinguish between strong and weak edges using two thresholds
        :param: img: image matrix after processing
        :return: img_out: image after two thresholds detecting
        '''
        img_out = img.copy()

        for i in range(1, img_out.shape[0] - 1):
            for j in range(1, img_out.shape[1] - 1):
                if img[i, j] <= self.__l_th:
                    img_out[i, j] = 0
                elif img[i, j] > self.__h_th:
                    img_out[i, j] = 255
                else:
                    img_part = img[i - 1:i + 2, j - 1:j + 2]
                    if img_part.ravel().any() > self.__h_th:
                        img_out[i, j] = 255
                    else:
                        img_out[i, j] = 0

        return img_out

    def EdgeDetecting(self, img_loc: str) -> np.ndarray:
        img_ = self.__ValidationGraying(img_loc)
        img_ = self.__ImgSmoothing(img_)
        g_v, g_t = self.__HorVerDetection(img_)
        img_ = self.__NonMaxSuppression(g_v, g_t)
        img_ = self.__TwoThresholdDetection(img_)
        return img_


if __name__ == '__main__':
    canny_p = Canny(10, 30)
    img_ = canny_p.EdgeDetecting(lenna_loc)
    plt.axis('off')
    plt.imshow(img_, cmap='gray')
    plt.show()
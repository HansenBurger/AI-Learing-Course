import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'


class Basic():

    def __init__(self) -> None:
        pass


class ImgKMeans(Basic):
    '''
    Classes for image clustering
    :private param: img: image array from local
    :private param: arr: re-processed image sequences(2D-1D)
    :private param: criteria: cluster stopping condition
    '''

    def __init__(self, img_loc: str) -> None:
        super().__init__()
        self.__img = cv2.imread(img_loc)
        self.__arr = np.float32(self.__img.reshape((-1, 3)))
        self.__criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                           10, 1.0)

    @property
    def img(self):
        return self.__img

    def main(self, k_st: int):
        '''
        Main process for image clustering
        :param: k_st: k group st after clustering
        :return: res: img after clustering
        '''
        _, label, center = cv2.kmeans(self.__arr, k_st, None, self.__criteria,
                                      10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res = res.reshape((self.__img.shape))
        return res


def main():
    k_means_p = ImgKMeans(lenna_loc)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(cv2.cvtColor(k_means_p.img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Origin")
    axes[0 ,0].axis("off")
    for i in range(2):
        for j in range(3):
            if i == j and i == 0:
                continue
            else:
                k_st = int(math.pow(2, i * 3 + j))
                k_img = k_means_p.main(k_st)
                axes[i, j].imshow(cv2.cvtColor(k_img, cv2.COLOR_BGR2RGB))
                axes[i, j].set_title("K = {0}".format(k_st))
                axes[i ,j].axis("off")

    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

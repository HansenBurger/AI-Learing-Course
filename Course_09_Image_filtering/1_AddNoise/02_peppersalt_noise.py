import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'


class Basic():

    def __init__(self) -> None:
        pass


class PepperSaltNoise(Basic):

    def __init__(self) -> None:
        super().__init__()

    def __random_choose(self, shape_st: tuple, ratio_st: float) -> list:
        '''
        Get the random coordinates for noise adding
        :param: shape_st: image shape(row, col)
        :param: ratio_st: percentage of pixels
        :return: coord_s: coord from random searches
        '''

        coord_size = int(ratio_st * shape_st[0] * shape_st[1])
        coord_s = np.zeros((coord_size, 2), dtype=np.int32)

        for coord in range(coord_size):
            coord_x = random.randint(0, shape_st[0] - 1)
            coord_y = random.randint(0, shape_st[1] - 1)
            coord_s[coord] = np.array([coord_x, coord_y])

        return coord_s

    def __addon_noise(self) -> int:
        '''
        Add pepper salt noise on request
        :param: gray: gray value inputing
        '''
        gray_ = random.choice([0, 255])
        return gray_

    def main(self, src_img: np.ndarray, per: float):
        dst_img = src_img.copy()
        coords = self.__random_choose(src_img.shape, per)
        for c in coords:
            gray_add = self.__addon_noise()
            dst_img[c[0], c[1]] = gray_add
        return dst_img


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    gray = cv2.cvtColor(cv2.imread(lenna_loc), cv2.COLOR_BGR2GRAY)

    noise_adder = PepperSaltNoise()

    axes[0][0].imshow(gray, cmap="gray")
    axes[0][0].set_title("raw")

    def ImgShows(ax: any, per_st: float, title: str) -> None:
        dst = noise_adder.main(gray, per_st)
        ax.imshow(dst, cmap="gray")
        ax.set_title(title)

    ImgShows(axes[0][1], 0.3, "30%GN")
    ImgShows(axes[1][0], 0.5, "50%GN")
    ImgShows(axes[1][1], 0.8, "80%GN")

    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

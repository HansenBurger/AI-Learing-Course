import cv2
import numpy as np
from skimage import util
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'
hamming_count = lambda x, y: sum([x[i] != y[i] for i in range(len(x))])


def aHash(img: np.ndarray, scale_st: tuple = (8, 8)):
    img = img.astype(np.int16)
    img_ds = cv2.resize(img, scale_st)
    img_gray = rgb2gray(img_ds)
    img_flat = img_gray.flatten()
    img_mean = np.mean(img_flat)
    img_ahash = np.array([1 if i > img_mean else 0 for i in img_flat])

    return img_ahash


def main():
    origin = cv2.cvtColor(cv2.imread(lenna_loc), cv2.COLOR_BGR2RGB)
    gaussin = util.random_noise(origin, "gaussian")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(origin)
    axes[0].set_title("Origin")
    axes[1].imshow(gaussin)
    axes[1].set_title("Gaussin")
    fig.tight_layout()

    ahash_0 = aHash(origin)
    ahash_1 = aHash(gaussin * origin.max())
    print("OriginImg Hash:\t{0}".format(ahash_0))
    print("NoisedImg Hash:\t{0}".format(ahash_1))
    print("HammingDistance:\t{0}".format(hamming_count(ahash_0, ahash_1)))

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
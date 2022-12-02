import cv2
from skimage import util
from matplotlib import pyplot as plt

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'
'''
util.random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs)
mode:
    'gaussian': Gaussian-distributed additive noise.
    'localvar': Gaussian-distributed additive noise, with specified local variance at each point of image.
    'poisson':  Poisson-distributed noise generated from the data.
    'salt':     Replaces random pixels with 1.
    'pepper':   Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
    's&p':      Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
    'speckle':  Multiplicative noise using out = image + n*image, where n is Gaussian noise with specified mean & variance.
seed: If seed is None the numpy.random.Generator singleton is used. If seed is an int, a new Generator instance is used
clip: If True (default), the output will be clipped after noise applied for modes(if not, extend size)
'''


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    raw = cv2.cvtColor(cv2.imread(lenna_loc), cv2.COLOR_BGR2GRAY)
    gaussin = util.random_noise(raw, "gaussian")
    saltpepper = util.random_noise(raw, "s&p")

    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("raw")

    axes[1].imshow(gaussin, cmap="gray")
    axes[1].set_title("gaussian")

    axes[2].imshow(saltpepper, cmap="gray")
    axes[2].set_title("salt & pepper")

    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

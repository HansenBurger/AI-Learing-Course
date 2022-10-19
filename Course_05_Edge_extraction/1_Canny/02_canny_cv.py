import cv2
import numpy as np
from matplotlib import pyplot as plt

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'


def main():
    img = cv2.imread(lenna_loc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 200, 300)

    fig, ax_s = plt.subplots(1, 2, figsize=(6, 3))
    ax_s[0].axis('off')
    ax_s[0].imshow(gray, cmap='gray')
    ax_s[0].set_title('Gray')

    ax_s[1].axis('off')
    ax_s[1].imshow(edge, cmap='gray')
    ax_s[1].set_title('Canny')

    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
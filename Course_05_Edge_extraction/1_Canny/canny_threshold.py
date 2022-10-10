import cv2
import numpy as np

lenna_loc = 'Course_03_Digtal_image\StaticStorage\lenna.png'
window_name = 'Canny Batch'
slider_name = 'Low threshold'
low_th_inital = 0
low_th_maxium = 100
high_th_ratio = 3

img = cv2.imread(lenna_loc)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def CannyThreshold(l_th: int):
    img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    img_edge = cv2.Canny(img_blur, l_th, l_th * high_th_ratio, apertureSize=3)
    img_dst = cv2.bitwise_and(img, img, mask=img_edge)  # add color
    cv2.imshow(window_name, img_dst)


def main():

    cv2.namedWindow(window_name)
    cv2.createTrackbar(slider_name, window_name, low_th_inital, low_th_maxium,
                       CannyThreshold)
    CannyThreshold(low_th_inital)  # initialization

    if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from numpy import linalg

DISTORT_IMG = r'Course_06_Camera_model\StaticStorage\photo.jpg'


class Basic():

    def __init__(self) -> None:
        pass


class PerspectTrans(Basic):
    '''
    Perspective Transmition between 2-dimention matrix
    :param: __p_st_s: points from matrix in, shape in [4, 2]
    :param: __q_st_s: points from matrix out, shape in [4, 2]
    :param: __trans_m: tansmition matrix, shape in [3, 3]
    :func: Transmition: get the transmition coordinate from point
    '''

    def __init__(self, p_st_s: np.ndarray, q_st_s: np.ndarray) -> None:

        super().__init__()
        self.__p_st_s = p_st_s
        self.__q_st_s = q_st_s
        self.__trans_m = self.__GenTransMatrix()

    @property
    def trans_m(self):
        return self.__trans_m

    def __GenTransMatrix(self) -> np.ndarray:
        '''
        Obtain the transmition matrix
        :return: trans_m: transmition matrix, shape in [3, 3]
        '''
        eq_m_left = []
        eq_m_right = self.__q_st_s.ravel().reshape(8, 1)
        for i in range(self.__p_st_s.shape[0]):
            p_x, p_y = self.__p_st_s[i]
            q_x, q_y = self.__q_st_s[i]
            eq_r_0 = [p_x, p_y, 1, 0, 0, 0, -p_x * q_x, -p_y * q_x]
            eq_r_1 = [0, 0, 0, p_x, p_y, 1, -p_x * q_y, -p_y * q_y]
            eq_m_left.append(eq_r_0)
            eq_m_left.append(eq_r_1)
        eq_m_left = np.array(eq_m_left)
        trans_m = np.mat(eq_m_left).I * eq_m_right
        trans_m = np.array(trans_m).T[0]
        trans_m = np.append(trans_m, 1.0)
        trans_m = trans_m.reshape((3, 3))

        return trans_m

    def Transmition_I2O(self, src: tuple) -> list:
        '''
        Get the coordinate value of the point (input to output)
        :param: p_in: point's input coordinate
        :return: p_out: point's coordinate after transmition
        '''
        p_in = np.append(np.array(p_in), 1)
        p_in = p_in.reshape((3, 1))
        p_out = self.__trans_m * p_in
        p_out = p_out.reshape((1, 3))
        p_out = [round(i) for i in p_out.ravel()[0:2]]

        return p_out.tolist()

    def Transmition_O2I(self, dst: tuple) -> tuple:
        '''
        Get the coordinate value of the point (output to input)
        :param: p_out: point's output coordinate
        :return: p_in: point's coordinate before transmition
        '''
        dst = np.append(np.array(dst), 1)
        dst = dst.reshape((3, 1))
        src = linalg.inv(np.mat(self.__trans_m)) * dst
        src = np.array(src) / np.array(src)[-1]
        src = np.round(src).astype(np.int32).reshape((1, 3))[0]
        src = [i for i in src.ravel()[0:2]]

        return src


class DistortCorrect(Basic):

    def __init__(self, img_loc: str) -> None:
        super().__init__()
        self.__img_r = cv2.imread(img_loc)
        self.__img_p = self.__PreProcessing()
        self.__cnt, self.__vtx = self.__ContourDetection()
        self.__img_t = self.__PerspectiveTrans()

    def __PreProcessing(self) -> np.ndarray:
        '''
        Preprocess for len's distortion correction
        1. Image graying
        2. Gaussian smoothing
        3. Rectangular dilation
        :param: img_loc: image location
        :return: img: mage matrix after processing
        '''
        img = self.__img_r.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.dilate(img,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return img

    def __ContourDetection(self) -> list:
        '''
        Detect the edge of the image and approximate it with polygon
        + Since findContours only detects external contours, multiple contours 
        + may be caused by setting the canny threshold too small or too large, 
        + which needs to be adjusted dynamically.
        
        1. Enlarge the low threshold if the contours are too many 
        2. Polygon fitting from outside to inside until the first quad appears
        3. Narrowing the low threshold if the fitting condition is not satisfied
        '''
        img = self.__img_p.copy()
        th_qua = np.quantile(img.ravel(), 0.25)
        th_tqua = np.quantile(img.ravel(), 0.75)

        th_low, th_high = th_qua, th_tqua
        cnt_outer = np.array([])

        while (cnt_outer.size == 0):

            edge = cv2.Canny(img, th_low, th_high, apertureSize=3)
            k_delita = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edge = cv2.dilate(edge, k_delita)
            try:
                _, cnt_s, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            except:
                cnt_s, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            cnt_s = sorted(cnt_s, key=cv2.contourArea, reverse=True)

            if len(cnt_s) >= 10:
                th_low = round(th_low * 1.1)
                continue

            for cnt in cnt_s:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if not len(approx) == 4:
                    continue
                else:
                    cnt_outer = cnt
                    cnt_vertex = approx
                    break

            th_low = round(th_low * 0.8)

        return cnt_outer, cnt_vertex

    def __PerspectiveTrans(self) -> list:
        '''
        Generate perspective transformation results
        :return: img_dst: image for perspective transformation
        '''
        dist_c = lambda l_0, l_1, x: linalg.norm(l_0[x] - l_1[x])
        src, src_r, = self.__vtx, np.roll(self.__vtx, 2)
        src_side_s = list(dist_c(src, src_r, i) for i in range(4))
        src_side_0 = round(np.mean([src_side_s[0], src_side_s[2]]))
        src_side_1 = round(np.mean([src_side_s[1], src_side_s[3]]))

        # approxPolyDP output coordinates satisfy the counterclockwise distribution
        dst = np.array([[0, 0], [0, src_side_1], [src_side_0, src_side_1],
                        [src_side_0, 0]])

        trans_p = PerspectTrans(src.reshape(4, 2), dst.reshape(4, 2))
        img_dst = cv2.warpPerspective(self.__img_r, trans_p.trans_m,
                                      (src_side_0, src_side_1))

        return img_dst

    def Display(self) -> None:
        '''
        Display image boundaries and perspective transformation results
        '''
        for i in range(self.__vtx.shape[0]):
            vtx_coord = tuple(self.__vtx[i, 0, :])
            cv2.circle(self.__img_r, vtx_coord, 10, (255, 0, 0), -1)
        cv2.drawContours(self.__img_r, self.__cnt, -1, (0, 255, 0), 5)
        cv2.imshow('OriginImage', self.__img_r)
        cv2.imshow('TransImage', self.__img_t)
        cv2.waitKey(0)


if __name__ == "__main__":
    main_p = DistortCorrect(DISTORT_IMG)
    main_p.Display()

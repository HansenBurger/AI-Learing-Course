import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

img_loc = "Course_03_Digtal_image\StaticStorage\lenna.png"


class ImageBasic:
    def __init__(self) -> None:
        pass


class Equilibrium(ImageBasic):
    """
    func: class for array equilibrium
    a_in: one deminsion array (private)
    df: dataframe contain processed info (public)
    """

    def __init__(self, one_d_array: np.ndarray, v_range:int=256):
        super().__init__()
        self.__a_in = one_d_array if type(one_d_array) == np.ndarray else np.array(one_d_array)
        self.__v_range = v_range
        self.__pix_map = None

    @property
    def pix_map(self):
        return self.__pix_map

    def __InitDistribution(self):
        """
        func: save hist info to dataframe and sort by var size
        """
        gray_, pix_n = np.unique(self.__a_in, return_counts=True)
        hist_df = pd.DataFrame(dict(zip(['pix', 'n'], [gray_, pix_n])))
        hist_df = hist_df.sort_values(by='pix', ignore_index=True)
        hist_df = hist_df.set_index('pix',drop=True)
        self.__pix_map = hist_df

    def __MapCalculation(self):
        """
        func: get the mapping value for each var after equilibrium
        """
        map_df = self.__pix_map.copy()
        gray_ratio = map_df['n'] / len(self.__a_in)
        gray_map_v = np.cumsum(gray_ratio) * self.__v_range - 1
        gray_map_v = [0 if i < 0 else i for i in gray_map_v ]
        map_df['v'] = np.around(gray_map_v)
        map_df['v'] = map_df['v'].astype('int')
        map_df = map_df.drop(['n'], axis=1)
        self.__pix_map = map_df


    def Equalization(self):
        self.__InitDistribution()
        self.__MapCalculation()

class HistEqualization_by_interface(ImageBasic):
    def __init__(self, img_loc:str) -> None:
        super().__init__()
        self.__img = cv2.imread(img_loc)
        _, _, self.__c = self.__img.shape

    @property
    def img(self):
        return self.__img

    def HistBalance(self):
        for c in range(self.__c):
            src_c = self.__img[:, :, c]
            dst_c = cv2.equalizeHist(src_c) # only use in single channle
            self.__img[:, :, c] = dst_c


class HistEqualization_by_formular(ImageBasic):
    def __init__(self, img_loc:str):
        super().__init__()
        self.__img = cv2.imread(img_loc)
        self.__h, self.__w, self.__c = self.__img.shape

    @property
    def img(self):
        return self.__img

    def __EmptyImgGen(self, height:int, width:int, channel:int=0, type_=None):
        channel_st = self.__c if channel == 0 else channel
        type_st = self.__img.dtype if not type_ else type_
        img = np.zeros(shape=(height, width, channel_st), dtype=type_st)
        return img

    def HistBalance(self):
        img = self.__EmptyImgGen(self.__h, self.__w)
        for c in range(self.__c):
            # do a histbalance each channel
            src_c = self.__img[:, :, c]
            dst_c = img[:, :, c]

            balance_p = Equilibrium(src_c.flatten())
            balance_p.Equalization()

            for i in balance_p.pix_map.index:
                # use matrix filter and addtion to avoid traversal
                pix_type = i
                pix_mapv = balance_p.pix_map.loc[i,'v']
                dst_c += (src_c == pix_type).astype(np.uint8) * pix_mapv
        
        self.__img = img
        return


def main():
    sns.set_style('whitegrid')
    # img_process = HistEqualization_by_formular(img_loc)
    img_process = HistEqualization_by_interface(img_loc)
    color_order = ['b', 'g', 'r']

    fig_0, (ax_0, ax_1) = plt.subplots(1, 2)
    dist_info_0 = [img_process.img[:, :, i].ravel() for i in range(len(color_order))]
    ax_0.imshow(cv2.cvtColor(img_process.img, cv2.COLOR_BGR2RGB))
    ax_0.axis('off')
    ax_0.set_title('origin lenna')
    

    img_process.HistBalance()
    dist_info_1 = [img_process.img[:, :, i].ravel() for i in range(len(color_order))]
    ax_1.imshow(cv2.cvtColor(img_process.img, cv2.COLOR_BGR2RGB))
    ax_1.axis('off')
    ax_1.set_title('balanced lenna')

    fig_0.suptitle('Histogram equalization')
    fig_0.tight_layout()

    fig_1, ax_es_1 = plt.subplots(3, 2, figsize=(12, 8))
    
    for i in range(0, 3, 1):
        ax_es_1[i][0].hist(dist_info_0[i], 100, color=color_order[i], density=1)
        ax_es_1[i][1].hist(dist_info_1[i], 100, color=color_order[i], density=1)
    
    fig_1.suptitle('Histogram Comparison')
    fig_1.tight_layout()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

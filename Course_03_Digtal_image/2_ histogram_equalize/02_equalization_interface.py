import cv2
import seaborn as sns
import matplotlib.pyplot as plt

img_loc = "Course_03_Digtal_image\StaticStorage\lenna.png"

class ImageBasic:
    def __init__(self) -> None:
        pass


class HistEqualization(ImageBasic):
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

def main():
    sns.set_style('whitegrid')
    # img_process = HistEqualization_by_formular(img_loc)
    img_process = HistEqualization(img_loc)
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
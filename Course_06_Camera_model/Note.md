# 相机模型

## 一、相机模型

小孔成像为例的相机模型:
![camra_model](https://lhoangan.github.io/assets/images/posts/2018-07-30/camera_model.png)

可以得到几个坐标系:

1. 世界坐标系 $P(X_W, Y_W, Z_W)$
2. 摄像机坐标系 $P_O(X_C, Y_C, Z_C)$
3. 图像物理坐标系 $P'(X', Y')$
4. 图像像素坐标系 $P(u, v)$

### 1.1 坐标系

四个坐标系的描述和单位:

1. 世界坐标系: 物体真实世界的坐标, 单位是 **长度单位**。
2. 相机坐标系: 相机光心坐标系原点, 单位是 **长度单位**。
3. 图像物理坐标系: 以主光轴和图像平面交点为坐标原点, 单位**长度单位**。
4. 图像像素坐标系: 以图像的顶点为坐标原点, 单位是**像素**

> 这里需要注意图物理坐标系和图像的像素坐标系之间存在转换关系(像素的(0,0)点并不在图像的中心)

### 1.2 相机成像流程

物体 -> 世界坐标 -> 摄像机坐标 -> 图像物理坐标系 -> 图像像素坐标

### 1.3 世界坐标到相机坐标

变换公式(欧式变换，旋转平移):

$$
\begin{equation}
\begin{bmatrix}
X_C \\
Y_C \\
Z_C \\
1
\end{bmatrix} =
\begin{bmatrix}
R & t \\
0^T & 1 \\
\end{bmatrix}
\begin{bmatrix}
X_W \\
Y_W \\
Z_W \\
1
\end{bmatrix} =
L_W
\begin{bmatrix}
X_W \\
Y_W \\
Z_W \\
1
\end{bmatrix}
\end{equation}
$$

其中旋转矩阵为 $R$，平移矩阵为 $t$。对于用上述公式描述的原因，首先两个坐标系之间存在的位置变换可以概括为旋转和平移，而欧式变换正具备这样能力。

欧式变换(euclidean transformation)的坐标系变化示意图:

![euclidean_trans](http://motion.cs.illinois.edu/RoboticSystems/figures/modeling/rotation3d.svg)

其次，写成齐次坐标的形式可以把线性方程改成矩阵形式，对于多次的欧式变换更有利。

$$
\begin{equation}
b=T_1a,c=T_2b \Longrightarrow c=T_2T_1a
\end{equation}
$$

### 1.4 相机坐标系到物理坐标系

相机坐标(三维)->物理坐标系(二维)：投影

X在坐标转换中的投影过程:
![projection](https://img2020.cnblogs.com/blog/1483773/202107/1483773-20210717161119951-1917024256.png)

根据图中的相似三角形，可以容易推导到:

$$
\begin{equation}
\begin{align*}
X' = f \frac{X_C}{Z_C} \\
Y' = f \frac{Y_C}{Z_C}
\end{align*}
\end{equation}
$$

转换到矩阵形式就有:

$$
\begin{equation}
Z_c
\begin{bmatrix}
X \\
Y \\
1
\end{bmatrix}=
\begin{bmatrix}
f & 0 & 0 \\
0 & f & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X_C \\
Y_C \\
Z_C
\end{bmatrix}
\end{equation}
$$

### 1.5 物理坐标系到像素坐标系

物理呈现坐标系(二维)->像素坐标系(二维)：比例变换(单位转换)

1. 根据一个像素占多少长度，将长度单位转换为像素单位
2. 将转换后的像素坐标进行平移，使坐标原点为左上角

具体转换方式如下所示:

![coordinate_photo2pixel](https://pic1.imgdb.cn/item/6344c42d16f2c2beb183fc2b.png)

$o_{uv}$ 为像素坐标系的原点，$o$ 为图像物理坐标系的原点，$(u_0, v_0)$ 为其在像素坐标系中的坐标，对于物理坐标系中的点 $p'(X',Y')$ 可以得到其在像素坐标系中的 $p'(u,v)$，计算公式如下:

$$
\begin{equation}
\begin{align*}
u = \frac{X'}{dx} + u_0 \\
v = \frac{Y'}{dy} + v_0
\end{align*}
\end{equation}
$$

其中 $dx$ 与 $dy$ 是x和y方向上一个像素分别占多少个(**可能是小数**)长度单位，dist per pixel

转换为矩阵表示可以得到:

$$
\begin{equation}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=
\begin{bmatrix}
\frac{1}{dx} & 0 & u_0 \\
0 & \frac{1}{dy} & v_0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X' \\
Y' \\
1
\end{bmatrix}
\end{equation}
$$

### 1.6 公式整理合并

将之前从世界坐标系一直变换到像素坐标系的变换公式进行汇总

1.世界坐标系->镜头坐标系

$$
\begin{bmatrix}
X_C \\
Y_C \\
Z_C \\
1
\end{bmatrix} =
\begin{bmatrix}
R & t \\
0^T & 1 \\
\end{bmatrix}
\begin{bmatrix}
X_W \\
Y_W \\
Z_W \\
1
\end{bmatrix} =
L_W
\begin{bmatrix}
X_W \\
Y_W \\
Z_W \\
1
\end{bmatrix}
$$

2.相机坐标系 -> 物理成像坐标系

$$
Z_c
\begin{bmatrix}
X' \\
Y' \\
1
\end{bmatrix}=
\begin{bmatrix}
f & 0 & 0 \\
0 & f & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X_C \\
Y_C \\
Z_C
\end{bmatrix}
$$

3.物理成像坐标系->像素坐标系

$$
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=
\begin{bmatrix}
\frac{1}{dx} & 0 & u_0 \\
0 & \frac{1}{dy} & v_0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X' \\
Y' \\
1
\end{bmatrix}
$$

由后向前整合可以得到相机模型变换的最终方程:

$$
\begin{equation}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=
\frac{1}{Z_C}
\begin{bmatrix}
\frac{1}{d_x} & 0 & u_0 \\
0 & \frac{1}{d_y} & v_0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
f & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
R & t \\
0^T & 1 \\
\end{bmatrix}
\begin{bmatrix}
X_W \\
Y_W \\
Z_W \\
1
\end{bmatrix}
\end{equation}
$$

其中在相机到物理成像变换中矩阵多了一列，**是为了凑成3x4的矩阵，因为后面欧式变换矩阵是4x4的矩阵**。

由于存在相机内部参数 $u_0$，$v_0$ 以及焦距 $f$ 等**内部参数**，以及旋转变换 $R$，平移变换 $t$ 等**外部参数**，又可以分为内参和外参两部分。

内参相乘可以得到:

$$
\begin{equation}
\begin{bmatrix}
\frac{1}{d_x} & 0 & u_0 \\
0 & \frac{1}{d_y} & v_0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
f & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}=
\begin{bmatrix}
\frac{f}{d_x} & 0 & u_0 & 0 \\
0 & \frac{f}{d_y} & v_0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}=
\begin{bmatrix}
f_x & 0 & u_0 & 0 \\
0 & f_y & v_0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\end{equation}
$$

最终方程就可以简化成内参和外参相乘的形式:

$$
\begin{equation}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}=
\frac{1}{Z_C}
\begin{bmatrix}
f_x & 0 & u_0 & 0 \\
0 & f_y & v_0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
R & t \\
0^T & 1 \\
\end{bmatrix}
\begin{bmatrix}
X_W \\
Y_W \\
Z_W \\
1
\end{bmatrix}=
\frac{1}{Z_C}KT
\begin{bmatrix}
X_W \\
Y_W \\
Z_W \\
1
\end{bmatrix}
\end{equation}
$$

其中 $Z_C$ 因为是相机坐标系下物体的Z坐标(垂直于成像平面)，可以作为**坐标归一化系数**。

## 二、镜头畸变

镜头畸变分为: **径(半径方向)向畸变**和**切(切线方向)向畸变**两类

原因: 由于制造精度一集组装工艺的偏差引入畸变

两种畸变类型示意图:

![lens_distortion](https://learnopencv.com/wp-content/uploads/2020/04/tangential-and-radial-distortion-effect.jpg)

### 2.1 径向畸变

类型: 枕形畸变(向内), 桶形畸变(向外)

枕形畸变和桶形畸变示意图:

![barrel_pincusion](https://assets.learningwithexperts.com/90f71ef4-d822-4e75-b52b-cb616a922bf2/765x431)

### 2.2 切向畸变

切向畸变产生的原因: 透镜本身和相机传感器平面不平行，透镜被安装到镜头模组的安装偏差导致。

切向畸变示意图:

![tangential_distortion](https://www.researchgate.net/publication/332199146/figure/fig5/AS:743978198642690@1554389633677/Tangential-distortion.png)

### 2.3 畸变矫正

根据相机生产得到的相机标定参数(k1, k2, p1, p2, p3)，利用算法解决

#### 透视变换(Perspective Transformation)

透视变换的目的是把原始图像矫正为新图像，可以是直线到斜线，也可以是斜线到直线，由需求决定。其中存在特例**仿射变换(Affine Transformation)**，即图像从一个向量空间进行一次线性变换和一次平移, 变换到另一个向量空间。

仿射变换和透视变换类似，但前者用于**二维坐标变换**，后者用于**三维坐标变换**，仿射变换可以用齐次方程组表示:

$$
\begin{equation}
\begin{bmatrix}
x \\
y
\end{bmatrix}=
\begin{bmatrix}
a_{11} & a_{12} & b_{1} \\
a_{21} & a_{22} & b_{2} \\
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
1 \\
\end{bmatrix}
\end{equation}
$$

>从已知关系点找到对应关系(变换矩阵)，求得未知关系点。本质是**已知的原图像得到一个新图像**。

对于透视变换，根据欧式变换的齐次方程，可以设一个变换矩阵 $M$，用于矩阵变换的处理:

$$
\begin{equation}
\begin{bmatrix}
X \\
Y \\
Z \\
\end{bmatrix}=
M
\begin{bmatrix}
u \\
v \\
1 \\
\end{bmatrix}=
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}  \\
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
1 \\
\end{bmatrix}
\end{equation}
$$

又因为 $(X,Y)$ 和 $(x,y)$ 已知，除以 $Z$ 进行归一化后可以得到:

$$
\begin{equation}
\begin{align*}
X' = \frac{X}{Z} \\
Y' = \frac{Y}{Z} \\
Z' = \frac{Z}{Z} = 1
\end{align*}
\end{equation}
$$

带入变换矩阵后，可以求得：

$$
\begin{equation}
\begin{align*}
X' = \frac{a_{11}u+a_{12}v+a_{13}}{a_{31}u+a_{32}v+a_{33}} \\
Y' = \frac{a_{21}u+a_{22}v+a_{23}}{a_{31}u+a_{32}v+a_{33}}
\end{align*}
\end{equation}
$$

因为两个方程式中都有 $a_{33}$，且与 $(u, v)$ 无关，可以假定 $a_{33}=1$，带入后展开表达式可得:

$$
\begin{equation}
\begin{align*}
X' = a_{11}u+a_{12}v+a_{13}-a_{31}uX'-a_{32}vX' \\
Y' = a_{21}u+a_{22}v+a_{23}-a_{31}uY'-a_{32}vY'
\end{align*}
\end{equation}
$$

求解这里的八个未知数，需要变换前和变换后的各四个点坐标，假设

变换前坐标: $(u_0, v_0)$，$(u_1, v_1)$，$(u_2, v_2)$，$(u_3, v_3)$

变换后坐标: $(X'_0, Y'_0)$，$(X'_1, Y'_1)$，$(X'_2, Y'_2)$，$(X'_3, Y'_3)$

> **备注:**</br>这里关于点的选择，为什么只有平面坐标，和之前的空间坐标中的Z无关，我认为，在处理镜头畸变过程中的透视变换，其实可以看做透视结果在XY平面上的投影，因此不管透视前还是透视后，Z的坐标都没有影响。所以变换前和变换后的坐标都可以直接从平面图中指代。

带入可以得到齐次方程组:

$$
\begin{equation}
\begin{bmatrix}
u_0 & v_0 & 1 & 0 & 0 & 0 & -u_0X'_0 & -v_0X'_0 \\
0 & 0 & 0 & u_0 & v_0 & 1 & -u_0Y'_0 & -v_0Y'_0 \\
u_1 & v_1 & 1 & 0 & 0 & 0 & -u_1X'_1 & -v_1X'_1  \\
0 & 0 & 0 & u_1 & v_1 & 1 & -u_1Y'_1 & -v_1Y'_1  \\
u_2 & v_2 & 1 & 0 & 0 & 0 & -u_2X'_2 & -v_2X'_2  \\
0 & 0 & 0 & u_2 & v_2 & 1 & -u_2Y'_2 & -v_2Y'_2  \\
u_3 & v_3 & 1 & 0 & 0 & 0 & -u_3X'_3 & -v_3X'_3  \\
0 & 0 & 0 & u_3 & v_3 & 1 & -u_3Y'_3 & -v_3Y'_3  
\end{bmatrix}
\begin{bmatrix}
a_{11} \\
a_{12} \\
a_{13} \\
a_{21} \\
a_{22} \\
a_{23} \\
a_{31} \\
a_{32}
\end{bmatrix}=
\begin{bmatrix}
X'_0 \\
Y'_0 \\
X'_1 \\
Y'_1 \\
X'_2 \\
Y'_2 \\
X'_3 \\
Y'_3
\end{bmatrix}
\end{equation}
$$

解出齐次方程组就能得到对应的变换矩阵，利用变换矩阵就能给输入图像的其他像素坐标计算变换后的坐标，就能得到整个处理畸变后的图像。

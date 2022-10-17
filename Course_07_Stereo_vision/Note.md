# 立体视觉

## 一、立体视觉

1. 定义: 计算机视觉技术，一种计算机视觉技术，通过2及2以上图像推理出图像中每个像素的深度(距离)信息，**实际上现在也可以用单张图推导出深度信息**
2. 应用: 机器人(工厂, 扫地机器人)、辅助驾驶、无人机，(车的摄像头两个以上的, 支持自动跟车, 车道保持自动变道)
3. 原理: 借鉴人双眼视差原理，左右眼对真实世界某一物体观测存在差异，辨别物体远近
4. 误区: **输入和输出本身并不一定包含三维信息**, 研究的是图像中的三维信息

## 二、双目系统

### 2.1 单目系统

典型的单目系统类似于相机模型

![Monocular_systems](https://img2020.cnblogs.com/blog/1483773/202107/1483773-20210724234103453-1429771425.png)

其中 $O$ 为相机的光心，$\pi$ 为摄像头成像的平面，$P$，$Q$ 是物理世界中的点。单目系统对于 $P$，$Q$ 在成像平面上重叠，无法判断距离差异。

### 2.2 双目系统

双目系统可以在多个物体遮挡的时候来辨别谁近谁远。

![Binocular_system](https://www.researchgate.net/profile/Dongsheng-Zhang-5/publication/338730626/figure/fig1/AS:878241237704704@1586400436140/Schematics-of-binocular-stereo-vision-model.ppm)

其中:

1. 极平面(Epipolar plane): $O_{cl}$，$O_{cr}$，$P$ 三点确定的平面(光心和目标点的三角形)
2. 极点(Epipolar point): $O_{cl}$，$O_{cr}$ 连线与像平面 $I_1$, $I_2$ (数学模型中没有大小限制)的交点
3. 基线(Baseline): 光心得连线，**深度信息就是 $P$ 到基线的距离**。
4. 极线(Epipolar line): 极平面与像平面得交线 $o_l$，$o_r$

对双目系统模型俯视精简以后就可以得到一个双目系统的数学模型

![Bincular_in_math](https://pic1.imgdb.cn/item/634d34d116f2c2beb129539c.png)

其中：$P_L$，$P_R$ 分别 $P$ 在两个像平面上的对应的点，$B$ 为 双目的间距，$Z$ 为 $P$ 到基线的距离，也就是需要计算的深度信息，根据三角形相似原理:

$$
\begin{equation}
\frac{Z-f}{Z} = \frac{P_LP_R}{B}
\end{equation}
$$

又因为成像平面的宽度 $w$ 和 $P$ 在像平面上的坐标已知，可以得到:

$$
\begin{equation}
P_LP_R = B - (X_L - \frac{w}{2}) - (\frac{w}{2} - X_R) = B - X_L + X_R
\end{equation}
$$

带入公式(1)后可以得到:

$$
\begin{equation}
\frac{Z-f}{Z} = \frac{B - X_L + X_R}{B}
\end{equation}
$$

化简后可以得到:

$$
\begin{equation}
Z = \frac{B\times f}{X_L - X_R}
\end{equation}
$$

令 $D=X_L - X_R$，带入公式就有:

$$
\begin{equation}
Z = \frac{B\times f}{D}
\end{equation}
$$

而 $D$ 通常表示视差(Disparity)，即同一物体在两个像平面上的坐标差距。

> **补充:** 即使像平面不与基线平行，依然可以通过角度变换实现

变换示意图:

![Bincular_math_trans](https://pic1.imgdb.cn/item/634d34d116f2c2beb129539f.png)

### 2.3 视差(Disparity)

由双目系统计算深度信息的公式可得，视差 $D$ 与深度 $Z$ 成**反比*，即离基线越近的物体，产生的视差越大，相反则越小。

这也就是可以用肉眼，或者双摄像头来实现距离判断的原因。

## 三、点云模型

### 3.1 点云与三维图像

1. 三维图像: 是一种特殊的信息表达式，特征是表达空间中三个维度的数据。

2. 点云信息: 扫描资料以点的形式记录，每一个点包含有三维坐标，有些可能含有颜色信息(RGB)或反射强度信息(Intensity)

### 3.2 点云

概念: 获取物体表面每个采样点的空间坐标后，得到的是点的集合，称为点云(Point Cloud)

内容:

1. 激光测量: 三维坐标(XYZ), 激光反射强度(Intensity)
2. 摄影测量: 三维坐标(XYZ), 颜色信息(RGB)

### 3.3 点云处理

三种层次处理内容不同，没有优劣区分

1. 低层次处理: 滤波, 关键点检测(SIFT)
2. 中层次处理: 特征描述(spin image), 分割(K-Means)和分类
3. 高层次处理: 配准，SLAM图优化，三维重建，点云数据管理

### 3.4 Spin image

定义:

1. 基于点云空间分布最经典的特征描述
2. 将一定区域的点云分布转换为二维的spin image, 对场景和模型的spin images进行相似性度量

(三维转二维, 比较区别)

处理步骤:

1. 定义一个Oriented point
2. 以Oriented point为轴生成一个圆柱坐标系
3. 定义Spin image的参数，Spin image是一个具有一定大小(行数列数)、分辨率(二维网格大小)的二维图像(或者说网格)
4. 将圆柱体内的三维坐标投影到二维Spin image，这一过程可以理解为一个Spin image绕着法向量 $n$ 旋转360度，Spin image扫到的三维空间的点会落到Spin image的网格中。
5. 根据spin image中的每个网格中落入的点不同，计算每个网格的强度 $I$

#### 1) Oriented Point

定义三维网格顶点 $p$ 为定向点

![orient_p](https://pic1.imgdb.cn/item/634d499816f2c2beb1505c82.png)

其中:

1. $P$: 切面
2. $n$: 法向量
3. $x$: $p$ 附近三维网格上另一个顶点
4. $\alpha$: $x$ 在 $P$ 的投影到 $p$ 的距离
5. $\beta$: $x$ 到 $P$ 的距离

#### 2) 坐标系

定向点就是以 $p$ 为圆心，$n$ 为法向量的圆柱坐标系，三维网格中的点都可以在这个坐标系中表示

#### 3) 参数设置

1. 分辨率: 二维图像中指的是像素的实际尺寸，三维网格中则用**边的平均值**表示，计算公式为 $r=\frac{1}{N}\sum^N_{i=1}|e_i|$，其中 $e$ 为三维网格的边，$N$ 为网格中边的总数

2. 大小: spin image的行数和列数(一般相等)

3. support angle: 法向量夹角的大小限制(网格之间的点的法向量夹角)，超出阈值的点不被计算在内

#### 4) 旋转

将圆柱体内的三维坐标投影到二维Spin image，这一过程可以理解为一个Spin image绕着法向量n旋转360度，Spin image扫到的三维空间的点会落到Spin image的网格中。

#### 5) 计算强度

根据每个网格落入点不同, 计算网格强度(灰度值)

![interp_spin-image](https://d3i71xaburhd42.cloudfront.net/4c09532c6ef9afd5f0dd1f3d2b0af313199a8520/41-Figure2-4-1.png)

因为旋转后的点并不会一定在spin-image的格子位置，需要用双线性插值(逆运算)，改变外周四个格子的像素值，也就是强度

#### 6) 拓展

三维图像中任意一个点，会落在spin-image的哪个网格中的计算公式

$$
\begin{equation}
i = \frac{w/2 - β}{b}
\end{equation}
$$

$$
\begin{equation}
j = \frac{\alpha}{b}
\end{equation}
$$

其中 $b$ 为网格大小，$w$ 为整体的尺寸

spin-image计算结果示意图:

![spin-image_example](https://d3i71xaburhd42.cloudfront.net/4c09532c6ef9afd5f0dd1f3d2b0af313199a8520/39-Figure2-2-1.png)

**需要注意:**

1.取定向点会影响spin-image的最终结果(和模型相似度比较)

![orient_choose](https://d3i71xaburhd42.cloudfront.net/4c09532c6ef9afd5f0dd1f3d2b0af313199a8520/24-Figure1-1-1.png)

2.对角度限制以后，那些相当于切面的“凹点（大于90°）”被剔除，保留了主要信息，降低了后续的计算量。

> 一般角度限制范围为60°~90°之间。

![angle_threshold](https://d3i71xaburhd42.cloudfront.net/c802e72edfd7d298aa9ea715de89080ed0051065/13-Figure5-1.png)

参考: [spinimage原理](https://www.pianshen.com/article/509682284/)

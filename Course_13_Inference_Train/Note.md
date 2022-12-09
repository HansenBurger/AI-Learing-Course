# 推理和训练(过程)

## 一、推理和训练

### 1.1 概念

监督和无监督:

1. 监督学习: 有标签, 关键方法分类和回归, 比如逻辑回归和BP神经网络
2. 无监督学习: 无标签, 关键是规则学习和聚合, **k-means**

推理和训练(都是有监督):

1. 训练(Training): 一个初始神经网络通过不断优化自身参数来变准确(**自身调整**)
2. 推理(Inference): 对现场数据(live data)进行识别，准确率高，则性能好

> 训练的和推理的**相互对应**，训练判断猫狗只能判断猫狗，对于未知类无法判断

推理和训练的主要区别可以参考下图:

![main_differ](https://cdn.shopify.com/s/files/1/2158/1497/files/inference_aspects.jpg?3687)

### 1.2 优化和泛化

深度学习的根本问题是优化和泛化的对立:

1. 优化(Optimization): 调节模型以在**训练数据**上得到最佳性能(训练)
2. 泛化(Generalization): 指训练好的模型在**前所未见的数据**上的性能好坏(推理)

最终的目标是能在泛化中获得更高准确度

### 1.3 数据集的分类

#### 1) 数据集划分

1. 训练集: 实际**训练算法的数据集**，用来计算梯度，确定每次迭代中网络权值的更新
2. 验证集: 用于跟踪其学习效果的数据集，是一个指示器，用于表明训练数据点之间所形成的网络函数发生(**中间结果的检查**)
3. 测试集: 用于产生最终结果的数据集

#### 2) 测试集要有效反应网络泛化能力

1. 测试集**不能以任何形式用于训练网络**，即使是用于同一组备选网络中挑选网络。测试集只能在所有的训练和模型选择完成后使用
2. 测试集必须代表网络使用中涉及的所有情形

补充:

1. 为了简便有时候验证集会省略, 主要用途是阶段测试
2. 训练集中不一定要包含负样本, 一个输出可以表示二分类

#### 3) 交叉验证

交叉验证的目的是用有限的数据量提升训练准确度。

交叉验证示意图:

![cross_validation](https://miro.medium.com/max/1400/1*AAwIlHM8TpAVe4l2FihNUQ.png)

### 1.4 BP神经网络

BP网路(Back-Propagtion Network)，即误差**逆向传播**算法训练的多层前馈网络，其示意图如下:

![BPN](https://www.researchgate.net/publication/330819898/figure/fig1/AS:721745224929280@1549088879336/Structure-of-BP-neural-network.png)

通过图可以看出，他和ANN的主要区别是多了**误差反向传播**的过程，即神经网络训练过程之一。

正向传播和反向传播:

1. 正向传播(输入到输出): 权值一开始随机，输出得到值与期望过大，计算误差反向传播
2. 反向传播(输出到输入): 按照**梯度下降**，不断调整神经元的连接**权值**和阈值

### 1.7 神经网络训练

利用神经网络去解决图像分割, 边界探测等问题 $y = f(x)$，其中**f不是一个简单的线性函数**，抽象化概念。

神经网络
    ├── 训练: (正向传播，反向传播)
    └── 推理: 用数据进行正向传播(和训练一致，不判断误差)

训练: 迭代是机器自己调整参数，而调参是人为的调整参数

不同硬件计算会不同:

1. cpu: 卷积
2. gpu: 二叉树

硬件会影响计算结果

## 二、训练的相关概念

### 2.1 神经网络训练概念

1. 代(Epoch): 全部数据进行**一次完整**训练，称为“一代训练”
2. 批大小(Batch Size): 使用训练集的一笑部分样本进行一次反向传播, 这一小批样本为一批
3. 迭代(Iteration): 使用**一个Batch数据进行一次参数更新**，称为"一次训练"(一次迭代)，每一次迭代得到的结果会作为下一次的初始。

> 一个迭代 = 一次正向通过 + 一次反向通过(更新权重)
> batch数 = 训练集大小 / batch大小

如: 训练集500样本，batchsize = 10；训练完样本: iteration=50，epoch=1(为了训练效果更好, 也可以增加epoch数)。**不同epoch之间使用同一个训练集，模型权重更新不同(后一个epoch在前一个基础上更新)**

此时，$(n,c,h,w)$ 中的n就代表着batch size。

### 2.2 训练流程

1. 提取特征向量作为输入
2. 定义神经网络结构，隐藏层，激活函数
3. 通过训练利用反向传播算法不断优化权重值，使之达到合理水平
4. 使用训练好的神经网络来预测未知数据(推理)。训练好的网络：**权重达到最优的情况**

一次训练结束可以得到一组权重

### 2.3 神经网络训练过程

1. 选择样本集合的一个样本 $(A_i, B_i)$，前为数据，后为标签
2. 送入网络，计算网络的实际输出Y，此时网络中权重应该都是随机量(**初始权值随机**)
3. 计算误差 $D=B_i - Y$ (预测值和实际值的差)
4. 根据误差 $D$ 来调整权重矩阵 $W$
5. 对样本重复上述过程，直到整个样本集，误差不超过规定范围

### 2.4 训练过程的细节

1. 参数随机初始化
2. 前向传播计算每个样本输出节点激活函数值
3. 计算损失函数
4. 反向传播计算偏导数

[可视化的训练过程](https://playground.tensorflow.org/)

## 三、训练的步骤和涉及的问题

### 3.1 参数的随机初始化

1. 随机初始化(避免神经元功能一致导致的**冗余**)
2. 初始化方式很多:
    1. 区间随机
    2. XAvier初始化(sigmoid、tanh)
    3. 正态分布(relu)
    4. MSRA(relu)，方差不同

### 3.2 标准化

进行分类器和模型的建立训练时，去除单位限制，转换为无量纲的纯数值，便于不同单位量级的指标进行比较加权

1. 归一化(0~1): $y = (x-x_{min}) / (x_{max}-x_{min})$
2. 归一化(-1~1): $y = (x-x_{mean}) / (x_{max}-x_{min})$
3. z-score标准化(零均值归一化)
    1. 处理后数据均值均为0，标准差为1(正态分布)
    2. $\mu$ 为均值，$\sigma$ 为方差
    3. $ y=(x-\mu) / \sigma$

### 3.3 损失函数

描述模型预测值与真实值差距。一般有**均值平方差(MSE)**和**交叉熵**。

#### 1) 均值平方差(MSE)

均方误差，预测值和真实值的差的平方和的均值:

$$
\begin{equation}
MSE = \sum^n_{i=1} \frac{1}{n} (f(x_i)- y_i)^2
\end{equation}
$$

#### 2) 交叉熵(cross entropy)

用于预测输入的样本在哪一类的概率，**值越小预测结果越准**

$$
\begin{equation}
C = - \frac{1}{n} \sum_x[y\ln a + (1-y)\ln (1-a)]
\end{equation}
$$

损失函数的选取取决于输入标签数据的类型:

1. 输入实数、无界数，使用MSE
2. 输入位矢量(分类标志)，使用交叉熵

> 反向传播中传递的就是计算的误差

### 3.4 梯度下降法(更新权重)

1. 梯度的方向函数值增大的方向，梯度的模表示函数值**增大的速率**
2. 不断将参数值向梯度**反方向更新**，得到损失函数最小值(全局最小值/局部最小值)
3. 一般利用梯度更新会乘以一个小于1的**学习速率(learning rate)**，这是因为往往梯度的模比较大，直接更新参数会使函数值不算波动，很难收敛(一般取0.8-1之间)

具体的过程可以参考下图:

![reducing_loss](https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentGradientStep.svg)

根据一组参数(权重)，计算损失函数 $f$ 以及损失函数对应的梯度方向 $\nabla f$，在梯度**负方向**上乘以一个learning rate，就能得到下次实验开始参数。用公式表示如下:

$$
\begin{equation}
\theta_{t+1} = \theta_t - \alpha_t * \nabla f(θ_t)
\end{equation}
$$

其中 $\theta_t$ 代表权重 $t$ 次迭代的权重，$\theta_{t+1}$ 代表更新后的权重，$\alpha_t$ 是 $t$ 次的学习率，这里用负的原因，是梯度下降，找反方向。

#### 1) 学习率

1. **学习率**是一个重要超参数，控制基于损失梯度调整神经网络权值的速度
2. 学习率越小, 沿着损失梯度下降速度越慢(可以避免错过局部最优解，但会增加收敛时间)
3. 新权值 = 当前权值 - 梯度 * 学习率

合适的学习率在梯度下降中的表现:

![learningrate_fit](https://developers.google.com/static/machine-learning/crash-course/images/LearningRateJustRight.svg)

#### 2) 汇总公式

整个梯度下降法，可以用如下公式表示:

$$
\frac{\partial E}{\partial w_{jk}} = -(t_k - o_k) \times sigmoid(\sum_jw_{jk}o_j)(1-sigmoid(\sum_jw_{jk}o_j)) \times o_j
$$

### 3.5 泛化能力

1. 欠拟合: 模型不能很好表现数据结构
2. 拟合: 测试误差与训练误差差距较小
3. 过拟合: 模型过分拟合训练样本, 对于测试样本预测准确率不高

三者比较示意图:

![fit_differ](https://media.geeksforgeeks.org/wp-content/uploads/20210323204619/imgonlinecomuaresizeLOjqonkALC.jpg)

#### 1) 过拟合

原因: 数据有噪声，会把噪声也拟合

结果: 造成模型比较**复杂**，使得**泛化能力差**

出现的原因:

1. 样本选取
2. 噪声过大
3. 假设模型无法合理存在
4. 参数太多导致模型复杂度过高
5. 神经网络: a)分类决策不唯一，b)迭代次数足够多但又缺少代表性的特征

解决方法:

1. 减少特征: 删除与目标不相关特征
2. Early stopping
3. 更多的训练样本
4. 重新清洗数据
5. Dropout(简单算法)

#### 2) Early Stopping

每一个Epoch结束时，计算准确率(最优)，准确率(10代内)不再提升就停止

#### 3) Dropout

dropout通过修改神经网络本身结构来实现:

1. 训练开始时, 随机删除隐藏层神经元
2. 根据BP对参数更新，下一次迭代，**继续随机删除**(降低隐藏层的复杂度)

> Dropout通过修改ANN中隐藏的神经元个数来减少过拟合，**这里的删除是指不更新权重，而非去掉神经元**

Dropout示意图:

![dropout](https://miro.medium.com/max/1044/1*iWQzxhVlvadk6VAJjsgXgg.png)

能实现解决过拟合原因:

1. 每次训练网络不一样
2. 隐藏节点以概率随机出现，不能保证2个隐节点每次都同时出现

通过交叉验证, 隐藏节点drop out率等于0.5效果最好(dropout可以视作一种添加噪声的方法)

优点:

1. 有效的神经网络模型平均方法(平均预测概率)
2. 不同模型在不同训练集训练，最后用相同权重"融合"

缺点:

1. 训练时间是没有dropout的2-3倍

### 3.7 计算实例

层数越深容易导致梯度消失(浅层)

对于如下的一个全连接(FNN)误差逆传播(BP)算法的网络，可以简单的计算其参数优化的过程:

![FNNBP_struct](https://pic.imgdb.cn/item/63908c46b1fccdcd368629d4.png)

1. 第一层为输入层($i_1，i_2$)，接收input，没有权重；
2. 第二层为隐藏层($h_1，h_2$)，提取特征，输入层到隐藏层间的权重分别为 $w_1, w_2, w_3, w_4$，偏置为 $b_1$；
3. 第三层为输出层($h_1，h_2$)，接收隐藏层，**并通过BP优化权重**，其权重分别为 $w_5, w_6, w_7, w_8$，偏置为 $b_2$

输入和输出以及偏置设置值为:

| type | value |
| ---- | ---- |
| $i_1$ | 0.05 |
| $i_2$ | 0.10 |
| $o_1$ | 0.01 |
| $o_2$ | 0.99 |
| $b_1$ | 0.35 |
| $b_2$ | 0.60 |

随机的权重设置值为:

| h-layer | h-w | o-layer | o-w |
| ---- | ---- | ---- | ---- |
| $w_1$ | 0.15 | $w_5$ | 0.40 |
| $w_2$ | 0.20 | $w_6$ | 0.45 |
| $w_3$ | 0.25 | $w_7$ | 0.50 |
| $w_4$ | 0.30 | $w_8$ | 0.55 |

具体到其中每一个神经元(neuron)的结构如下:

![neurou_fnn](https://pic1.imgdb.cn/item/639095a6b1fccdcd3695ea71.png)

#### 1) 前向传播

前向传播，主要根据FNN中权重的初始值，从输入到输出，计算出结果。

由神经元结构可隐藏层 $h_1$ 在的输出 $z_{h_1}$ 为:

$$
\begin{equation}
z_{h_1} = w_1 * i_1 + w_2 * i_2 + b_1
\end{equation}
$$

激活函数为sigmoid，因此易得隐藏层 $h_1$ 的输出 $a_{h_1}$ 为

$$
\begin{equation}
a_{h_1} = \frac{1}{1+\exp(-z_{h_1})} = \frac{1}{1+exp(-w_1 * i_1 - w_2 * i_2 - b_1)}
\end{equation}
$$

同理可得隐藏层 $h_2$ 的输出 $a_{h_2}$ 为

$$
\begin{equation}
a_{h_1} = \frac{1}{1+\exp(-z_{h_1})} = \frac{1}{1+exp(-w_3 * i_1 - w_4 * i_2 - b_1)}
\end{equation}
$$

在通过输出层就可以得到第一次前向传播的结果 $a_{o_1}, a_{o_2}$ 分别为

$$
\begin{equation}
a_{o_1} = \frac{1}{1+exp(-w_5 * a_{h_1} - w_6 * a_{h_2} - b_2)}
\end{equation}
$$

$$
\begin{equation}
a_{o_2} = \frac{1}{1+exp(-w_7 * a_{h_1} - w_8 * a_{h_2} - b_2)}
\end{equation}
$$

#### 2) 计算损失函数

得到输出结果，需要比较 $a_{o_1}, a_{o_2}$ 与真实值 $o_1, o_2$ 间的差距，这里用的是MSE。

首先根据MSE(均方误差)可得

$$
\begin{equation}
E_{total} = \sum_{i=1}^n \frac{1}{2}(a_{o_i}- o_i) ^ 2 = E_{o_1} + E_{o_2}
\end{equation}
$$

而 $E_{o_1}，E_{o_2}$ 可以分别由MSE计算得到:

$$
\begin{equation}
E_{o_1} = \frac{1}{2}(o_1-a_{o_1}) ^ 2
\end{equation}
$$

$$
\begin{equation}
E_{o_2} = \frac{1}{2}(o_2-a_{o_2}) ^ 2
\end{equation}
$$

得到传播结果和真实结果的误差后，需要进行BP来修正参数(基于 $E_{total}$)

#### 3) 反向传播(输出-隐藏)

要知道权重对于整体损失的影响，可以计算损失函数对权重的偏导，以权重 $w_5$ 为例:

![neurou_bp](https://pic1.imgdb.cn/item/6390a02bb1fccdcd36a9058e.png)

作用部分，可以分为在加权求和中的影响，在

$$
\begin{equation}
\frac{\partial E_{tot}}{\partial w_5} = \frac{\partial z_{o_1}}{\partial w_5} \times  \frac{\partial a_{o_1}}{\partial z_{o_1}} \times \frac{\partial E_{tot}}{\partial a_{o_1}}
\end{equation}
$$

Step.1对加权求和的作用为:

$$
\begin{equation}
\frac{\partial z_{o_1}}{\partial w_5} = a_{h_1}
\end{equation}
$$

根据sigmoid求导方法，得到Step.2对于激活函数的作用:

$$
\begin{equation}
(\frac{1}{1+e^{-x}})' = \frac{e^{-x}}{(1+e^{-x})^2} =  \frac{1 + e^{-x} - 1}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}(1- \frac{1}{1+e^{-x}})
\end{equation}
$$

$$
\begin{equation}
\frac{\partial a_{o_1}}{\partial z_{o_1}} = \frac{1}{1+\exp(-z_{o_1})} =z_{o_1}(1 - z_{o_1})  
\end{equation}
$$

最后得到Step.3中对于损失函数的作用:

$$
\begin{equation}
\frac{\partial E_{tot}}{\partial a_{o_1}} = (\frac{1}{2}(o_1-a_{o_1}) ^ 2)' = -(o_1 - a_{o_1})
\end{equation}
$$

最后三者相乘可以得到:

$$
\begin{equation}
\frac{\partial E_{tot}}{\partial w_5} =-(o_1-a_{o_1})\times z_{o_1}(1 - z_{o_1})\times a_{h_1}
\end{equation}
$$

最后，根据学习率更新 $w_5$，可以得到:

$$
\begin{equation}
w_5^+ = w_5 - \eta \frac{\partial E_{tot}}{\partial w_5}
\end{equation}
$$

#### 4) 反向传播(隐藏-输入)

优化输入层到隐藏层的权重，和之前的BP环节相同，但**这层的权重会同时对** $o_1, o_2$ 产生影响(但作用仅限**h1**这一个神经元)，原因如图:

![ann_bp_w_1](https://pic.imgdb.cn/item/63915ebdb1fccdcd366f83e4.png)

因此求偏导的公式可以写成:

$$
\begin{equation}
\frac{\partial E_{tot}}{\partial w_1} = \frac{\partial(E_{o_1}+E_{o_2})}{\partial w_1}
\end{equation}
$$

在根据之前作用的流图，拆解偏导方程可以得到:

$$
\begin{equation}
\frac{\partial E_{tot}}{\partial w_1} = (\frac{\partial E_{o_1}}{\partial a_{h_1}} + \frac{\partial E_{o_2}}{\partial a_{h_1}}) \times \frac{\partial a_{h_1}}{\partial z_{h_1}} \times i_1
\end{equation}
$$

又因为损失函数对隐藏层输出(**输出层的输入**)的偏导可以展开，得到:

$$
\begin{equation}
\frac{\partial E_{o_1}}{\partial a_{h_1}} = \frac{\partial E_{o_1}}{\partial a_{o_1}} \times \frac{\partial a_{o_1}}{\partial z_{o_1}} \times w_5
\end{equation}
$$

$$
\begin{equation}
\frac{\partial E_{o_2}}{\partial a_{h_2}} = \frac{\partial E_{o_2}}{\partial a_{o_2}} \times \frac{\partial a_{o_2}}{\partial z_{o_2}} \times w_6
\end{equation}
$$

展开后的公式为:

$$
\begin{equation}
\frac{\partial E_{tot}}{\partial w_1} = -((o_1-a_{o_1}) z_{o_1}(1 - z_{o_1})+(o_2-a_{o_2}) z_{o_2}(1 - z_{o_2}))\times z_{h_1}(1 - z_{h_1})\times i_1
\end{equation}
$$

最后，根据学习率更新 $w_5$，可以得到:

$$
\begin{equation}
w_1^+ = w_1 - \eta \frac{\partial E_{tot}}{\partial w_1}
\end{equation}
$$

#### 5) 循环迭代

反向传播算法就完成了，最后我们再把更新的权值重新计算，不停地迭代

### 3.8 总结

BP算法是一个迭代算法，它的基本思想如下:

1. 将训练集数据输入到神经网络输入层，经过隐藏层，最后达到输出层并输出结果，这就是前向传播
2. 利用损失函数计算估算值和真实值的误差，将误差从输出层想隐藏层反向传播，直到输入层
3. 根据误差调整神经元权重(乘以学习率)，使得总损失函数减少
4. 迭代三个步骤，直到满足条件停止

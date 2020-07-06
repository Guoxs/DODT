###  NIPS 2018: Trajectory Convolution for Action Recognition 论文笔记

date: 2018-12-12

------
​	2018 NurIPS 关于行为识别 (视频分类) 的论文, 行为识别任务是根据一小段视频, 然后预测视频中表现的行为类别. 基本来说就是个分类的任务, 只不过输入是视频流, 所以如何有效捕获帧间的信息, 提取时域的特征是该领域的研究重点. 

#### 1.1 方法回顾
​	深度学习之前, 效果最好的行为识别方法是基于密集轨迹的方法, 即先在图像中生成密集的轨迹，再沿着轨迹提取特征，从而获得视频整体的编码。
经典的基于轨迹的方法有 [DT][1], [iDT][2].  DT (dense trajectories) 算法的框架图如下图所示, 包括: 

- **密集采样特征点** : 从8个不同的空间尺度采样, 尺度因子 $\frac{1}{\sqrt(2)}$ , 采样的点只需要简单地每间隔W = 5个像素取一个点即可

- **特征点轨迹跟踪** : 下一个点位置通过密集光流场估计, 对采样点逐帧追踪. 为了防止轨迹点的漂移，密集轨迹最多追踪 L 帧, 当在一个 $W*W$ 的邻域内没有发现追踪点，那么采样一个点。构造轨迹编码局部的动作模式，通过偏移量序列 $S = (\Delta P_t, ... , \Delta P_{t+L-1})$ 描述这条轨迹。

- **基于轨迹的特征提取**:  沿轨迹的描述子. HOG(Histogram Of Gradient, 梯度方向直方图) , HOF(Histogram Of Flow, 光流方向直方图), MBH(Motion Boundary Histograms). 

>  对于 HOG 特征, 其统计的是灰度图像梯度的直方图; 对于 HOF 特征, 其统计的是光流（包括方向和幅度信息）的直方图. 而对于 MBH 特征, 它的处理方法是将 x 方向和 y 方向上的光流图像视作两张灰度图像，然后提取这些灰度图像的梯度直方图. 即 MBH 特征是分别在图像的 x 和 y 方向光流图像上计算 HOG 特征.

最后利用 SVM 对提取的特征进行分类. 

![DT](/home/mooyu/.config/Typora/typora-user-images/1544605998409.png)

​	之后, DT 算法的作者又进行了算法改进, 提出了 iDT (improved DT) 算法 (深度学习之前效果最好的行为识别算法). iDT算法则主要是增加了视频帧间的对齐，从而尽可能地消除相机运动带来的影响。在DT和iDT方法中，采用的都还是人工设计的传统特征. 

​	在深度学习流行后, Yuanjun Xiong 前辈提出了 [TDD][3] 算法，如下图所示，主要是将 iDT 算法中的传统手工设计特征替换为深度网络学习的特征，获得了一定的效果提升。

![TDD](/home/mooyu/.config/Typora/typora-user-images/1544618525387.png)

​	虽然轨迹类的方法符合人类对视频的直观理解，但此前的这几种轨迹方法都存在着比较大的缺陷从而难以获得更好的应用. 首先在这些方法中，轨迹的提取以及特征的提取是独立的过程，一方面在实现是比较**繁琐**，另外一方面也**不能够进行端到端的学习**；其次，这些方法最后都依赖于 Fisher Vector 或 VLAD 编码，通常会产生非常**高维的特征向量**，在储存和计算等方面效率都比较差。因此，最近几年基本上没有啥新的轨迹类方法。

> Fisher Vector : 本质上是用似然函数的梯度向量来表达一幅图像. 对高斯分布 $\mathcal{L}(X|\lambda) = \sum^T_{t=1} log \mathcal{p}(x_i|\lambda)$ 的变量求偏导, 也就是对权重, 均值, 标准差求偏导, 最后归一化处理.
>
> VLAD 编码: 局部特征聚合描述符(Vector of Locally Aggregated Descriptions). 将海量局部描述符聚合到一个单独的向量, 使用 K-mean 方法对特征进行聚类.

​	而对于深度学习的其他行为识别方法 大致可分为 **双流网络** 和 **3D 卷积网络 **两类. 近两年大量基于 3D卷积网络 的工作主要针对如何增加3D网络容量、降低3D网络计算开销、更好地进行时序关联和建模等问题进行了研究。其中，很多方法采取的思路是将3D卷积分解为2D空间卷积加上1D的时序卷积，如 [Separable-3D(S3D)][4] 、[R(2+1)D][5]等等.

​	文章作者认为, 直接在时间维度上进行卷积隐含了一个很强的假设，即认为帧间的特征是很好地对齐地，而事实上人或者物体在视频中可能存在着很大地位移或是形变。因此，作者认为**沿着轨迹来做时序上的卷积**是更合理的方式。

#### 1.2 方法介绍

**轨迹卷积**

​	本文提出的轨迹卷积主要受到可变形卷积网络 [DCN][6] 的启发. 可变形的详细解析见下一篇博文. 可变形网络主要想法如下图所示, 简单来说, 网络通过学习每次卷积的offset, 来实现非规则形状的卷积. 

![1544621453730](/home/mooyu/.config/Typora/typora-user-images/1544621453730.png) 

​	文中作者在时序上将轨迹的偏移向量直接作为可变形卷积的 offset, 从而实现轨迹卷积. 如下图所示, 传统的 3D 卷积或是时序卷积在时序方向上的感受野是对齐的, 而轨迹卷积则是按照轨迹的偏移在时序上将卷积位置偏移到对应的点上去, 从而实现沿着轨迹的卷积.

> Farmally, parameterized by the filter weight $\{w_\tau : \tau \in [-\Delta t, \Delta t]\}$ with kernel size $(2\Delta t + 1)$, the output feature $y_t(p)$ is calculated as
> $$
> y_t(p) = \sum^{\Delta t}_{\tau = - \Delta t} w_{\tau}x_{t+\tau}(\tilde{p}_{t+\tau}) 
> $$
> the point $p_t$ at frame t can be tracked to position $\tilde{p}_{t+1}$ at next frame (t+1) in the presence of a forward dense optical flow $\vec{\omega} = (u_t, v_t) = \mathcal{F}(I_t, I_{t+1})$ using the following equation
> $$
> \tilde{p}_{t+1} = (h_{t+1}, \omega_{t+1}) = p_t + \vec{\omega}(p_t) = (h_t, \omega_t) + \vec{\omega}|_{h_t, \omega_t}
> $$
> For $\tau > 0$, the same position $\tilde{p}_{t+\tau}$ can be calculated by applying Eq.(2) iteratively. To track to the previous frame (t-1), a backward dense optical field $\vec{\omega} = (u_t, v_t) = \mathcal{F}(I_t, I_{t-1})$ is used likewise.

![1544621736553](/home/mooyu/.config/Typora/typora-user-images/1544621736553.png)

​	在实现环节, 轨迹卷积可以看做是 3D 可变形卷积的一个特例, 卷积核大小设置为 3x1x1, 即沿着时序卷积, 偏移量方面则将时序偏移设置为0，只考虑空间上的偏移。与可变形卷积网络不同的是，轨迹卷积中的空间偏移量并不是通过网络学习得到，而是设定为相邻帧之间轨迹的偏移量。因此，基于可变形卷积网络的代码，轨迹卷积是非常易于实现的。

**表观与运动特征结合**

​	轨迹卷积实际上是**沿着运动方向对表观特征进行结合**，作者认为这样的方式对运动信息的表达还不够。参考DT算法的思路，可以直接将**轨迹偏移量**信息作为特征。在这篇文章中，作者则将轨迹的偏移量图直接和原始的表观特征图进行了堆叠，从而进行了信息的融合。这样的融合方式比起双流网络中 late fusion 的方式要更高效和自然一些。此处的轨迹偏移量图为**降采样的运动场图**（比如光流图）。

**网络结构**

轨迹卷积网络直接将 Separable-3D 网络 (ResNet 18 architecture) 里中层的 1D 时序卷积层替换为轨迹卷积层.

![1544624163876](/home/mooyu/.config/Typora/typora-user-images/1544624163876.png)

![1544623505507](/home/mooyu/.config/Typora/typora-user-images/1544623505507.png)

**轨迹的学习**

​	文中所采用的密集轨迹通常是通过光流的方式呈现。光流的提取有很多方式，传统的方式通过优化的方法计算光流，而近几年基于深度学习的方法则获得了很好的效果。为了能够将轨迹的生成也纳入网络一起学习，本方法采用了[MotionNet][6]网络，将预训练的 MotionNet 和轨迹卷积网络一起训练。在此处的训练过程中，并不采用真实光流的监督信息，而是采用了 MotionNet 论文中提出的无监督辅助损失函数。最后的实验结果表明不采用辅助损失函数直接finetune会带来效果的降低，而添加辅助损失函数则能带来效果的上升。



#### 1.3 实验结果

​	该论文在 Something-Something-V1 和 Kinetics 这两个大规模视频分类数据集上进行了实验，并比起baseline（S3D) 获得了一定的效果提升。具体效果如下图所示。

![1544625306719](/home/mooyu/.config/Typora/typora-user-images/1544625306719.png)

​	从结果可以看出，基于较小的基础网络，轨迹卷积网络也获得了不错的效果，表明轨迹卷积网络的有效性。另外一方面，行为识别方法的速度也很重要，下图则展示了S3D网络以及轨迹卷积网络的单次前向速度。可以看出，目前轨迹卷积网络的速度还有较大的提升空间。

![1544625393848](/home/mooyu/.config/Typora/typora-user-images/1544625393848.png)



参考知乎专栏: 

[[NIPS 2018论文笔记] 轨迹卷积网络 TrajectoryNet](https://zhuanlan.zhihu.com/p/51550622)



[1]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5995407
[2]: https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Wang_Action_Recognition_with_2013_ICCV_paper.pdf
[3]: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Action_Recognition_With_2015_CVPR_paper.pdf
[4]: http://openaccess.thecvf.com/content_ECCV_2018/papers/Saining_Xie_Rethinking_Spatiotemporal_Feature_ECCV_2018_paper.pdf
[5]: http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2648.pdf

[6]: https://arxiv.org/pdf/1704.00389




### ICCV2017: Deformable Convolutional Networks 论文笔记

2018-12-13

---

#### Introduction

​	物体识别任务中，主要的挑战就是怎么去解决由于 `scale、pose、viewpoint、part deformation`等原因导致的识别困难的问题。 目前该问题主要由两种方法来处理:

	① 数据扩增, 构建一个包含各种变化的数据集, 对训练数据进行增强，使得网路对这些变换更加鲁棒.
	② 使用具有形变不变性 (transformation-invariant) 的特征和算法 (例如 SIFT, 基于滑动窗口的算法).

​	然而以上方法存在如下缺点:

	① 使用的数据增强的方法都是基于已知的变换进行的，因此如果在新的数据集上出现了其他的变换，则模型很难处理;
	② 手工设计的特征或算法无法应对过度复杂的形变，即便该形变是已知的.

​	CNN 的特征提取主要是基于卷积核实现，传统的卷积核有比较明显的局限性，因为他只能采集到固定的局部信息，而且卷积核的形状都是固定的，因此难以适用于形状变化较大的物体。此外，在 `high-level layer `中，所有 `kernel` 的 `respective field` 都是相同的，这显然不符合不同物体的规律.

> a convolution unit samples the input feature map at fixed locations; a pooling layer reduces the spatial resolution at a fixed ratio; a RoI (region-of-interest) pooling layer separates a RoI into fixed spatial bins; There lacks internal mechanisms to handle the geometric transformations. This causes noticeable problems. For one example, the receptive field sizes of all activation units in the same CNN layer are the same.  

​	为了解决或减轻这个问题，论文引入了两种新的模块**可变形卷积** (deformable convolution) 和 **可变形感兴趣区域池化** (deformable RoI pooling) ，来提高对形变的建模能力。这两个模型都是基于一个平行网络学习offset（偏移），使卷积核在input feature map的采样点发生偏移，集中于我们感兴趣的区域或目标。

> 题外话: DCN 与 STN
>
> ​	可变形卷积网络 deformable convolutional networks, 也算是在 STN 之后的一个新的变换——STN 是说CNN Kernel 定死了(比如3*3大小), 但是可以通过图片变换让 CNN 效果更好; 而 deformable 是说既然图片可能各种情况，那我索性 CNN 的 Kernel 本身是不规整的, 比如可以有 dilation, 也可以旋转的，或者看起来完全没有规则的。STN主要是学习 global and sparse 的空间变换特征（旋转缩放等）去做整图的归一化，DCN所引入学习的是更加 local and dense 的变换特征，针对每一个 filter 都预测了 offset，使其能更加有效地覆盖 target 所在的区域。
> ​	如果要做语义分割，只需要用 Deformable Convolution 即可，但如果要做目标检测，則需要另外使用Deformable RoI Pooling，对多个 Region of Interest (ROI) 进行处理，缩放成相同大小的 feature map.

#### Deformable Convolution

​	传统的CNN中, 假设卷积的grid $\mathcal{R}$ 为
$$
\mathcal{R} = (-1,-1),(-1,0),...,(1,1)
$$
假设 $\mathcal{R}$ 集合中元素的个数为 N, 卷积的计算方式是
$$
y(p_0) = \sum_{p_n\in \mathcal{R}} w(p_n)\cdot x(p_0+p_n)
$$
在可变形卷积中, 每个位置都有一个微小的变化, 卷积可以表示为
$$
y(p_o) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0+p_n+\Delta p_n)
$$
其中 $\Delta p_n$ 是微小的变化, 因此在计算该位置的输入时, 使用 **双线性插值** 进行计算
$$
x(p)= \sum_q max(0, 1-|q_x-p_x|) \cdot max(0, 1-|q_y-p_y|) \cdot x(q)
$$
其中 $p=(p_0+p_n+\Delta p_n)$ , max() 函数将 q 的集合范围约束为距离 p 最近的 4 个 grid。

![1544708537264](/home/mooyu/.config/Typora/typora-user-images/1544708537264.png)

​	可变形卷积可视化如下图所示, 把原来的卷积过程分成两路，共享 input feature maps。上面一路用一个额外的 conv 层来学习offset $\Delta p_n$ , 得到 $H∗W∗2N $ 的输出（offset）, $N = |\mathcal{R}|$ 表示 grid 中像素个数，2N 的意思是有 x, y 两个方向的 offset 。有了这个 offset 以后，对于原始卷积的每一个卷积窗口，都不再是原来规整的sliding window（input feaure map中的绿框），而是经过平移后的 window（蓝框），取到数据后计算过程和卷积一致。即 input feature maps 和 offset 共同作为 deformable conv 层的输入，deformable conv 层操作采样点发生偏移，再进行卷积。

![1544715305092](/home/mooyu/.config/Typora/typora-user-images/1544715305092.png)

#### Deformable RoI Pooling

​	给定 input feature map $x$ 和 $w×h$ 的 RoI 以及左上角 $p_0$ ，传统 RoI pooling 把 RoI 分成 $k \times k$ 个 bins，对每个bin 内的多个像素做 average pooling （或 max-pooling, 本文不是用 max-pooling, 有可能是为了与位置敏感 RoI Pooling 兼容），最后输出 $k \times k$ 的 output feature map，对于 (i, j)-th bin ($0 \leq i, j \lt k$)，有:
$$
y(i,j) = \sum_{p\in bin(i,j)} x(p_0 + p)/n_{ij}
$$
 其中 $n_{ij}$ 是一个 bin 中的像素个数, (i, j)-th bin 的范围为 $i \left \lfloor \frac{w}{k} \right \rfloor \leq p_x \lt  (i+1) \left \lceil \frac{w}{k} \right \rceil, j \left \lfloor \frac{h}{k} \right \rfloor \leq p_x \lt  (j+1) \left \lceil \frac{h}{k} \right \rceil$  .

而 deformable RoI pooling 类似 deformable convolution，在spatial binning positions 中也增加了offsets $\{\Delta p_{ij}|0\leq i,j \lt k\}$, 公式变为
$$
y(i,j) = \sum_{p\in bin(i,j)} x(p_0 + p + \Delta p_{ij})/n_{ij}
$$
 和前面一样，因为offset是带小数的，不能直接得到像素，需要用双线性插值算法得到每一个像素值。

如下图所示

![1544715336612](/home/mooyu/.config/Typora/typora-user-images/1544715336612.png)


>​	和 deformable convolution 的区别在于用的是 FC 层，原因是 RoI pooling 之后的结果是固定大小的 $k∗k$ feature map，直接用 FC 得到 $k∗k$ 个offset。但是这些offset不能直接用，因为 RoI 区域大小不一，并且input feature map的 w 和 h 也是大小不一。作者提出的方法是用一个 scalar（element-wise product with RoI’s w and h ）:
>$$
>\Delta p_{ij} =\gamma \cdot \hat{p}_{ij} \circ (w,h)
>$$
>其中，$\gamma$ 是一个预定义的 scalar，根据经验设为0.1。

#### Deformable Position-Sensitive RoI Pooling

​	这个结构主要是针对 R-FCN 的 position-sensitive RoI pooling 层做的修改. Deformable Position-Sensitive RoI Pooling 和 Deformable RoI pooling 层的区别就是把之前公式中的  input feature map $x$ 替换为了位置敏感分值图中的 $x_{ij}$（9个颜色的feature map块之一）。

![1544715414852](/home/mooyu/.config/Typora/typora-user-images/1544715414852.png)

#### 分析与结果

Deformable convolution 的一大优势就是可以使用学习的方式，对 convolution kernel 的形状进行修改，从而使得网络对物体变形、尺寸变化等问题能够有更好的性能。在实际的任务中 deformable convnets 学习到的采样点如下所示:

![1544708708910](/home/mooyu/.config/Typora/typora-user-images/1544708708910.png)

实验结果数据

![1544715523360](/home/mooyu/.config/Typora/typora-user-images/1544715523360.png)

### Deformable ConvNets V2

前不久 DCN 的改进版本放出来了, 目前挂在 arxiv 上.

V2 说 V1 存在的问题是在 RoI 外部的这种几何变化适应性表现得不好，导致特征会受到无关的图像内容影响.

> this support may nevertheless extend well beyond the region of interest，causing features to be influenced by irrelevant image content

为了分析Deformable ConvNet（DCN），首先介绍本文提到的三个概念：

- **有效感受野 ** (Effective receptive fields): 网络中每个节点都会计算 feature map 的一个像素点，而这个点就有它自己的感受野，但是不是感受野中的所有像素对这个点的响应的贡献都是相同的，大小与卷积核权重有关，因此文中用有效感受野来表示这种贡献的差异。

- **有效采样/bin位置 ** (Effective sampling/bin locations): 对于卷积核的采样点和 RoI pooling 的 bin 的位置进行可视化有助于理解 DCN，有效位置在反应采样点位置的基础上还反应了每个位置的贡献。

- **错误边界显著性区域**  (Error-bounded saliency regions): 最近关于图像显著性的研究表明，对于网络的每个节点的响应，不是图像上所有的区域对其都有影响，去掉一些不重要的区域，节点的响应可以保持不变。根据这一性质，文章将每个节点的 support region 限制到了最小的可以和整幅图产生相同的响应的区域，并称之为错误边界显著性区域。

![1544716364153](/home/mooyu/.config/Typora/typora-user-images/1544716364153.png)

![1544716508237](/home/mooyu/.config/Typora/typora-user-images/1544716508237.png)

![1544716542699](/home/mooyu/.config/Typora/typora-user-images/1544716542699.png)

​	上图展示了普通卷积的有效采样位置，有效感受野和错误边界显著性区域，作者发现虽然采样点始终是矩形，但是普通的卷积可以通过卷积核的参数适应一定的几何形变。反观DCN，能够使得卷积操作更集中在想要关心的位置。
本文对 V1 做了3方面的改进：**增加可变形卷积的层数**，**增加可调节的可变形模块**，**采用蒸馏的方法模仿RCNN的特征**.

#### Stacking More Deformable Conv Layers

​	v1 中使用的 ResNet-50，只将 conv5 中的共 3 层 3x3 卷积换成了可变形卷积，本文则将 conv3，conv4 和conv5 中一共 12 个 3x3 的卷积层都换成了可变形卷积。v1 中发现对于 pascal voc 这样比较小规模的数据集来说，3 层可变形卷积已经足够了。同时错误的变形也许会阻碍一些更有挑战性的 benchmark 上的探索。作者实验发现在 conv3 到 conv5 中使用可变形卷积，对于 COCO 上的 object detection 来说，是效率和精度上最好的均衡。

#### Modulated Deformable Modules

​	v1 仅仅给普通的卷积的采样点加了偏移，v2 在此基础上还允许调节每个采样位置或者 bin 的特征的amplitude，就是给这个点的特征乘以个系数，如果系数为0，就表示这部分区域的特征对输出没有影响，所以这也是一种调节 support region 的方法。
$$
y(p_o) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0+p_n+\Delta p_n) \cdot \Delta m_k
$$

上面卷积层的分辨率和 $x$ 相同，但是输出有 $3K$ 个通道，$2K$ 对应每个采样点的 $\Delta p_k$ (x，y) 两个方向，$K$ 个对应$\Delta m_k$ (要经过 sigmoid)。特别重要的是得到 $\Delta p_k$ 和 $\Delta m_k$ 的卷积核的参数一开始一定要初始为 0， $\Delta p_k$ 和 $\Delta m_k$ 的初始值则为 0 和 0.5。这些新加入的卷积层的学习率则是现有的层的0.1。

可调节的RoIpooling也是类似的，公式如下
$$
y(i,j) = \sum_{p\in bin(i,j)} x(p_0 + p + \Delta p_{ij}) \cdot \Delta m_k/n_{ij}
$$
上式是求第 k 个 bin 的特征值，该 bin 对应的像素点个数为 $n_k$ 个，$x(p_{kj} + \Delta p_k)$ 代表 $bin_k$ 内的像素点 j 偏移后的像素值，由双线性插值得到。

#### R-CNN Feature Mimicking

​	作者发现对于 RoI 分类时，普通 CNN 或者 DCN V1 的错误边界显著性区域都会延伸到 RoI 之外，于是与 RoI不相关的图像内容就会影响 RoI 特征的提取，从而可能影响目标检测的结果。不过 R-CNN 在进行分类时，结果完全是依赖于 RoI 的，因为 R-CNN 的分类 branch 的输入就 RoI 的 cropped image。

> 但有人说并不是没有额外的 context 就一定不好，不然也不会有那么多研究怎么利用context的了.

​	对于 V2 来说，将在 RoIpooling 中 RoI 之外的调节系数设为 0 可以直接去除无关的 context，但是实验表明这样还是不能很好的学习 RoI 的特征，这是由于 Faster R-CNN 本身的损失函数导致的，因此需要额外的监督信号来改善 RoI 特征提取的训练。于是作者采用 feature mimicking 的手段，强迫 Faster R-CNN 中的 RoI 特征能够接近 R-CNN 提取的特征。不过对于非目标的位置，过于集中的特征或许不好，因为需要 context 来保证不会产生假正的检测结果，因此 feature mimic loss 只用在 positive RoI 上。网络训练的框架如上图所示，右侧其实就是一个R-CNN，不过后面只接了分类的branch。

![1544717838755](/home/mooyu/.config/Typora/typora-user-images/1544717838755.png)

Feature mimic loss定义如下：
$$
\mathcal{L}_{mimic} = \sum_{b \in \Omega} [1-cos(f_{RCNN}(b), f_{FRCNN}(b))]
$$
其中 $\Omega$ 代表 RoI 的集合。此外还引入了 R-CNN 的分类损失，使用 cross entropy 实现的，新加入的这两个损失的系数是原本损失的 0.1。还有就是虽然上图画了两个网络，但是 R-CNN 和Faster R-CNN 中相同的模块是共用的相同的参数，比如 Modulated Deformable Convolutions 和Modulated Deformable RoIpooling 部分。inference 的时候就只需要运行 Faster R-CNN 的部分就行了。

#### 实验结果

在不同图像尺寸下的对比实验，还没有加入R-CNN feature mimicking的方法，可以看到，可变形卷积的层数的堆叠对性能的提升是很明显的

![1544718330637](/home/mooyu/.config/Typora/typora-user-images/1544718330637.png)



参考博文:

[目标检测7 - Deformable Convolutional Networks](https://blog.csdn.net/ibunny/article/details/79397832)

[Deformable ConvNets](https://littletomatodonkey.github.io/2018/12/02/2018-12-02-Deformable%20ConvNets/)

[论文阅读：Deformable ConvNets v2: More Deformable, Better Results](https://blog.csdn.net/qq_37014750/article/details/84659473)

[Deformable ConvNets V2](https://littletomatodonkey.github.io/2018/12/02/2018-12-02-Deformable%20ConvNets%20V2/)
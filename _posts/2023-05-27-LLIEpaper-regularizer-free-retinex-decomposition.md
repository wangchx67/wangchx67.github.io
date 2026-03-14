---
layout: post
title: LLIEpaper-Regularizer-Free Retinex Decomposition (RFR)
date: 2023-05-27 00:00:00
description: CVPR2023低光照增强论文：不使用额外先验和正则项的Retinex分解方法
tags: paper-reading
categories: notes
---
 
本文介绍了CVPR2023的低光照增强论文：You Do Not Need Additional Priors or Regularizers in Retinex-based Low-light Image Enhancement

该论文介绍了一种不使用额外的先验和正则项去实现Retinex分解并且取得了SOTA的performance。

## Brief intro of Retinex theory

我们首先简单回顾Retinex原理：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Retinex成像原理

人眼看到一张图像是由入射光打在物理上并反射到人眼的过程：

$$
I = R \circ L
$$

其中 $I$ 代表观测到的图像，$R$ 是反射分量（reflectance），$L$ 是入射分量（illumination）。入射分量 $L$ 由外部环境决定，反射分量 $R$ 由物体本身的反射率决定。

在低光照图片增强任务中，一般默认低光照是由于入射光不足导致的亮度不足，所以对于成对的低光照和正常光照图像，反射分量应该是相同的。

虽然理论上可以将图片分成两个分量，但是实际上没有任何方法可以真正将这两个分量提取出来，传统方法通过高斯模糊去近似入射分量。

一般来说，实现Retinex分解的优化目标为：

$$
\min \|I - \tilde{R} \circ \tilde{L}\| + \phi(\tilde{R}, \tilde{L})
$$

其中 $\tilde{R}$，$\tilde{L}$ 是分解出的反射分量和入射分量，$\phi(\tilde{R}, \tilde{L})$ 是对应的正则项，第一项可以理解成重构损失。

## Existing DNN-based Retinex decomposition method

这里介绍两篇比较经典的关于Retinex分解的论文：RetinexNet 和 KinD。

### RetinexNet

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


RetinexNet的框架如图所示。这里用到的正则项为：
- 第一项用到了不同光照条件下反射光应该一样的原理
- 第二项是一个TV正则，使得分解出的分量具有结构特性

总体来说比较简单，这也导致Retinex的performance不是很好，但是这是首次提出这么结合Retinex理论的深度学习架构，非常具有启发性，特别常用的LOL数据集也来自该文章。

### KinD

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/3.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


KinD 对Retinex分解做了进一步改进，其实就是在分解上多了些约束，然后在分解后专门有个去噪网络和定制化的入射分量增强网络。具体约束如下：

- 入射分量的平滑约束
- 反射分量的结构保持约束
- 反射分量在不同光照下的一致性约束

（论文report的指标在入射分量增强部分直接使用GT/Input作为系数，其实不太合理。。。但是可视化结果确实很平滑）

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/4.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


## Limitations of using regularization

论文中提到，这些正则项对场景很敏感，很难找到适应多种场景的正则项，并且过多的约束会影响performance（因为这些正则项其实都不是指标指向的，比如过于使用一些平滑正则就会导致PSNR很低）。

## Method

该论文提出了Regularizer-Free Retinex decomposition and synthesis network (RFR)，利用对比学习和知识蒸馏的方法对Retinex分解做约束。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/5.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


### 框架概述

首先，该论文指出，传统的入射分量与反射分量相乘会放大噪声，采用合成的方式会缓解这个情况。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/6.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


所以该论文在分解之后不做相乘，而是concat一起输入到一个合成网络。不过这块如果这么合成是不是和Retinex原理相悖了，不相乘而是靠concat一起让网络去学就相当于分解出来的东西的物理意义没有了。

按照这个模式可以从图片分解出来很多特征，然后concat一起再去输出一张图片，不一定非要是Retinex的分解。

所以优化问题就转化成了如下所示：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/7.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

这几个网络分别是预测入射分量、反射分量以及合成入射分量和反射分量的网络。

论文指出，这么直接去优化这个目标函数是不行的，因为对于第二项对于反射分量的约束，只要预测反射分量的网络一直输出同样的东西（比如全0）就可以使得这项最小。

相当于如果直接优化，预测反射分量的网络输出一样的东西，然后利用第一项损失函数主要训练合成网络和入射分量预测网络。

### Retinex Decomposition with Contrastive Learning

所以该论文使用对比学习的方式去优化这个函数。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/8.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


将低光照对应的GT当成正向样本，不对应的就当成负向样本。除此之外，作者发现低光照数据集（比如LOL）对重复场景会拍多组图片的情况，在选取正负样本的时候采用计算SSIM的方式，提出了一个改良版的对比学习损失函数 Weighted Normalized Temperature-Scaled CrossEntropy Loss (WNT-Xent Loss)：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/9.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

其中 $\omega$ 为在batch size为B的一个batch中的样本与对应图片的SSIM。

其实这个新设计的loss被当成了一个contribution，但是这个其实是用了数据集的先验，对提升performance应该很有用。

### Retinex Decomposition with Self-knowledge Distillation

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/18.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

知识蒸馏目标函数：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/10.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

学生网络和教师网络输入差太多会导致学生网络学不到东西，所以为了拉近学生网络和教师网络的域差距，使用了mix augment：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/11.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

另外这部分的mix augment还可以用在对比学习那部分做数据增强，同时作者提到这么做可以让网络适应不同的曝光度的输入。

## Experiments

实验设定：在LOL上训练，LOL测试集上测试，VE-LOL测试集上测试，无参考数据集上做测试。

Note：事实上，VE-LOL的100张测试集就是LOL-V2（LOL扩展版）的测试集，这一百张图片包含在了LOL的训练集上，所以理论上这种在LOL上训练在VE-LOL上测试，去证明网络具有泛化性很不合理。


### 实验结果

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/12.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

以上是实验结果，主要就是cross-dataset evaluation上的结果不太能接受。另外这个指标确实很高，不知道有没有跟LLFlow一样在最后再乘上一个跟GT相比的系数。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/13.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/14.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/15.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

以上是可视化结果以及user study。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/13.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/13.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

这部分有一块不理解是如果 $\omega$ 不同这个利用数据集先验的WX损失函数，而用原始的会有多大的差别（即 $\omega_{WX} \times P_{L2}$）。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_retinex/13.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

另外，论文做了对以前Retinex网络分解后不相乘而是用合成网络去预测的结果，实验结果上证明了这么做的有效性，但是应该引入了一定的参数量，作者没有给出详细的解释，而且SSIM的提升也太高了。

## Reference

[1] Wei C, Wang W, Yang W, et al. Deep retinex decomposition for low-light enhancement[J]. arXiv preprint arXiv:1808.04560, 2018.

[2] Zhang Y, Zhang J, Guo X. Kindling the darkness: A practical low-light image enhancer[C]//Proceedings of the 27th ACM international conference on multimedia. 2019: 1632-1640.

[3] Wang Y, Wan R, Yang W, et al. Low-light image enhancement with normalizing flow[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(3): 2604-2612.

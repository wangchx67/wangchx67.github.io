---
layout: post
title: LLIEpaper-Low-Light Image Enhancement via Structure Modeling and Guidance
date: 2023-05-27 00:00:00
description: CVPR2023低光照增强论文：基于结构建模和引导的图像增强方法
tags: paper-reading
categories: notes
---
 
本文介绍CVPR2023低光照增强论文Low-Light Image Enhancement via Structure Modeling and Guidance[1]。该论文介绍了一种基于结构先验的图像增强方式，主要研究如何从低光照图像中提取到的好的边缘信息并用来指导增强。

## Motivation

1. 现有低光照图像增强方法忽视了在低光照区域结构信息建模对增强的作用（ignore the explicit modeling of structural details in dark areas）从而导致增强效果不理想，比如细节模糊。
2. 虽然有些方法提出利用边缘结构信息去增强，但是他们经常不能得到理想的边缘结构信息，因为低光照的影响。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

该论文提出了一种基于GAN Loss的模型去对结构信息建模，通过获得的结构信息指导增强。如上图，低光照图片中很难提取到好的结构信息，该论文提取到的结构信息会好很多，并且用于指导图像增强也取得更好的效果。

## Method

该论文提出了一种Low-Light Image Enhancement via Structure Modeling and Guidance，结构图如下：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Appearance Modeling

首先通过一个简单的Unet去对外观建模，可以理解成先简单地学习到一个正常光照的coarse version。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/3.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

损失函数为一个重建损失加感知损失：

$$
\mathcal{L}_{appearance} = \mathcal{L}_{reconstruction} + \mathcal{L}_{perceptual}
$$

### Structure Modeling

对于结构信息建模部分，借鉴了GAN prior[2]的思路，但是不用预训练的GAN模型。这里设计到一个结构特征的提取Structure-Aware Feature Extractor (SAFE)以及基于StyleGAN的结构信息生成Structure-Aware StyleGAN Generator (SAG)。

#### Structure-Aware Feature Extractor (SAFE)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/4.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

对于输入特征图，首先计算八个方向的梯度，使用一阶梯度，得到的是八个方向的梯度特征图加上一个原本的空间特征图。然后分别通过一个Long-Range Encoder (LRE)模块和Short-Range Encoder (SRE)模块得到全局特征和局部特征，其实就是Transformer和CNN双支路。然后再通过一个Long-Short-Range Fusion (LSR-F)模块去融合，结构论文里说了是MLP结构。最后，还要通过一个Gradient Fusion (Grad-F)模块去融合提取到的梯度的特征和空间特征图，论文没说什么结构，应该是concat+CNN。

损失函数用对GT的canny检测器到的边缘图来监督，使用Annotator-robust Loss[3]以及GAN Loss：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/5.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

#### Structure-Aware StyleGAN Generator (SAG)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/6.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

得到结构特征提取编码器的最后一层特征以后，根据StyleGAN的理论，将提取到的特征通过一个mapping network（全连接层）到隐空间。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/7.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

$P$ 为池化层，为了让特征图与StyleGAN的输入尺寸一致。

### Structure-Guided Enhancement Module

得到建模的结构信息后，进行基于结构信息指导的增强Structure-Guided Enhancement Module。这里提出了一个基于结构信息的特征融合，其中包含一个结构信息指导的卷积Structure Guided Convolutions (SGC)和结构信息指导的归一化Structure Guided Normalizations (SGN)模块。

#### Structure Guided Convolutions (SGC)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/8.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


用得到的结构信息图预测出一个卷积核与原来的特征图做卷积。

#### Structure Guided Normalizations (SGN)

用得到的结构信息图预测出 $\alpha$，$\gamma$，然后原来的特征图乘以 $\alpha$ 再加上 $\gamma$，最后加上一个残差连接：

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/9.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

损失函数为重建损失加上感知损失

## Experiment

实验设定：在sRGB-based数据集 LOL-real，LOL-sys，以及raw-data-based数据集 SID 开展实验。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/10.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 可视化效果

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/11.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/12.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


### 消融实验

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zhihu_tupian_paper_struceture/13.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- **Ours w/o A**: 移除了appearance modeling
- **Ours w/o S**: 移除了structural modeling
- **Ours w/o F**: 使用了GAN prior[3]的架构去提取结构信息图
- **Ours w/o G**: 移除了结构信息指导的增强模块，即结构信息指导的卷积和归一化，将结构信息图concat一起输入
- **Ours w/o S.G**: 使用一个SOTA边缘提取网络代替论文中的structural modeling模块（但是在低光照下很难提取到有效的边缘）
- **Ours w/o GAN**: 移除GAN Loss
- **Ours with noise**: 增加噪声测试模型鲁棒性

消融实验结果思考：appearance modeling（Ours w/o A）和GAN loss的作用有点明显，但是都不是该论文的核心创新点（或者说是比较简单的创新点）。

### 使用额外的边缘检测以及其他结构信息作为引导

论文指出低光照数据集图片比较少，所以结构信息提取模块的训练会受限，因此作者引入其他结构信息提取的数据集做额外训练可以进一步提升performance（Ours with E.D.）

另外作者提出除了边缘结构图作为引导信息，也可以使用其他特征，比如语义图（Ours with SEG.）和深度图（Ours with Dep.），总体表现没有边缘图好，但是也比一些现有的增强方法好。

## 个人思考

- 个人希望看到使用GT的结构信息能使得performance提到多少？
- 如果拿appearance modeling的结果去提取边缘会达到怎么样的效果？
- 暂未开源，无法实验。

## Reference

[1] Xu X, Wang R, Lu J. Low-Light Image Enhancement via Structure Modeling and Guidance[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 9893-9903.

[2] Yang T, Ren P, Xie X, et al. Gan prior embedded network for blind face restoration in the wild[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 672-681.

[3] Liu Y, Cheng M M, Hu X, et al. Richer convolutional features for edge detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 3000-3009.

---
layout: post
title: Dynamic Region-Aware Convolution (DRConv)
date: 2026-03-08 00:00:00
description: 旷世CVPR2021论文Dynamic Region-Aware Convolution介绍及其在底层视觉中的应用
tags: convolution computer-vision paper-reading
categories: notes
---

本文介绍旷世在CVPR2021上发表的论文Dynamic Region-Aware Convolution以及该论文在一些底层视觉上的应用。

该文章提出了一种新的卷积方式，Dynamic Region-Aware Convolution（DRConv），首先回忆几种常用的卷积形式：

1. **Standard Convolution** - 标准卷积
2. **Dilation Convolution** (空洞卷积)
3. **Deformable Convolution** （可变形卷积）
4. **Depth-wised Convolution** （深度可分离卷积）

## Limitations of Existing Convolutions

当前主流的卷积方法采用在空间上参数共享的方式提取特征，为了提取到更多的信息，只能一直使用不同的卷积核重复提取特征（增加特征图的通道数），这些冗余的计算操作会带了额外的计算负担，并且提取特征的能力也有待提高。

## DRConv

本文提出一个动态区域感知卷积，可以针对特征图中不同的区域，使用不同的卷积核进行卷积操作。

### DRConv 卷积公式

对于一个标准的二维卷积，假设输入为 $F_{in}$，$H,W,C$代表高度，宽度以及通道数，$H \times W$ 表示空间维度，输出 $F_{out}$，卷积核 $K$，忽略卷积的跨步，卷积可以表示为:

$$
F_{out}(i,j) = \sum_{m,n} K(m,n) \cdot F_{in}(i+m, j+n)
$$

此时的卷积核为对整张输入的卷积，加入跨步信息，降低卷积核，即我们正常理解的卷积形式，整个输入共享同一个卷积核。

引入一个guide mask $M$ 表示将输入分成多个区域，在每个区域使用不同的卷积核 $K_1, K_2, ..., K_m$，其中 $m$ 是区域数量，$k$ 代表卷积核的尺寸，此时，卷积可以表示为:

$$
F_{out}(u,v) = \sum_{c} \sum_{m,n} K_{M(u,v)}(m,n,c) \cdot F_{in}(u+m, v+n, c)
$$

其中 $(u,v)$ 代表卷积核的中心位置，因为区域的形状是不固定的，所以当 $k>1$ 时，是有可能提取到邻接区域的信息的。

所以实现DRConv有两个问题：
- 如何生成guide mask？
- 如何生成不同的卷积核？

## Learnable guided mask

对于 $k \times k$ 的DRConv有 $m$ 个区域，输入为 $X \in \mathbb{R}^{H \times W \times C}$，首先使用 $k \times k$ 标准卷积生成guided feature $G \in \mathbb{R}^{H \times W \times m}$，guided mask $M$ 可以通过 hardmax() 实现：

$$
M_{i,j} = \arg\max_c G_{i,j,c}
$$

### 简单示例

假设 $m=3$（分成三个区域），得到的guided feature 有3个通道，遍历guided feature，返回最大值所在feature的索引，最终的guided mask 也会分成三个区域。

由于最大值操作argmax不可导，没法反向传播，所以反向过程需要特殊设计。

### Forward (前向传播)

给定 $m$ 个卷积核 $K_1, K_2, ..., K_m$，对于某个区域，采用的卷积核为：

$$
K_{M(u,v)} = \sum_{i=1}^{m} K_i \cdot \mathbb{1}[M(u,v) = i]
$$

### Backward (反向传播)

在guided mask 上进行了hardmax（取最大值）的操作，但是由于这种操作不可导，在反向传播的时候使用softmax近似梯度。

## Filter generator module

为了得到 $m$ 个 $k \times k$ 的卷积核，首先将输入 $X$ 通过adaptive average pooling（AAP）成 $X' \in \mathbb{R}^{1 \times 1 \times C}$，然后通过两个 $1 \times 1$ 卷积层：
- 第一个卷积层使用sigmoid为激活函数
- 第二卷积层使用分组卷积并不带激活层

## 实验结果

在ImageNet上分类达到SOTA的效果，并且计算量相比标准卷积和其他方式有很大提升。

作者可视化了guided mask，确实会具有一些语义信息。

## Applications of DRConv in low-level vision

### 1. Local Color Distributions Prior for Image Enhancement（ECCV 2022）

该论文通过使用局部颜色分布先验去解决同一幅图像中同时存在欠曝以及过曝区域的图像。通过DRConv将局部颜色分布图融入网络之中。

### 2. Learning Hierarchical Dynamics with Spatial Adjacency for Image Enhancement (MM 2022)

该论文发现暗通道的图与DRConv的guided mask有相似的分布，所以使用了暗通道代替guided mask的作用实现 DRConv。

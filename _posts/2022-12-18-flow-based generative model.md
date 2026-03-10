---
layout: post
title: Flow-based Generative Model
date: 2022-12-18 00:00:00
description: 流模型生成模型
tags: deep-learning generative-model flow normalization computer-vision
categories: paper-sharing
---

# 流模型生成模型 (Flow-based Generative Model)

**汇报人：王晨曦**
**2022.12.18**

---

## 1. 引言

生成模型旨在将一个分布生成另外一个分布，这个分布可以跟我们想要的真实的分布非常相近。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image5.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image6.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image7.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image8.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 2. 生成模型概览

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image9.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 2.1 GAN

**GAN** 通过联合训练一个判别器，但是存在难训练的问题（模式崩塌）。

### 2.2 VAE

**VAE** 通过最优化出一个最优下界（ELBO），没有直接优化目标函数，所以能力非常有限。

### 2.3 Flow-based Model

流对目标函数进行模型则直接去优化，虽然这对模型的设计会有很高的要求。

| 模型 | 优点 | 缺点 |
|------|------|------|
| GAN | 生成质量高 | 训练不稳定、模式崩塌 |
| VAE | 训练稳定 | 生成的图像模糊 |
| Flow | 精确的对数似然、潜在变量可解释 | 计算量大、网络结构受限 |

---

## 3. 前置知识

### 3.1 雅可比矩阵 (Jacobian Matrix)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image10.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image11.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

一个函数的雅可比矩阵，与这个函数的逆的雅可比矩阵互逆：

$$J_{f^{-1}}(f(x)) \cdot J_f(x) = I$$

这一性质对于流模型的可逆性至关重要。

### 3.2 矩阵行列式

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image12.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image13.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image14.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image15.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

**行列式就是体积**。这一几何意义在理解概率密度变换时非常重要。

### 3.3 变量代换公式

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image16.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image17.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image18.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image19.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image20.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

变量代换公式：

$$p_X(x) = p_Z(f(x)) \cdot \left|\det \frac{\partial f(x)}{\partial x}\right|$$

或者等价地：

$$\log p_X(x) = \log p_Z(z) + \log \left|\det \frac{\partial f(x)}{\partial x}\right|$$

其中 $z = f(x)$ 是从输入 $x$ 到潜在变量 $z$ 的变换，$\det \frac{\partial f(x)}{\partial x}$ 是雅可比矩阵的行列式。

---

## 4. 流模型核心思想

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image21.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

流模型的核心思想是：学习一个可逆的变换 $f$，将复杂的数据分布 $p_X(x)$ 映射到简单的先验分布 $p_Z(z)$（通常是高斯分布）。

### 4.1 网络结构要求

对生成器有很高的要求：
1. **可逆性**：必须保证函数可逆
2. **行列式可计算**：行列式计算不能太耗时

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image22.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 5. 耦合层设计 (Coupling Layer)

### 5.1 基本思想

如何设计这样一个生成器？相对而言，行列式的计算要比函数求逆要困难，所以我们从行列式计算出发思考。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image23.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image25.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image26.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image27.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image28.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image29.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image30.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image31.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image32.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

**关键设计**：
- 将 $D$ 维的 $x$ 分为两部分 $x_1, x_2$
- 使用仿射变换：$y_1 = x_1$，$y_2 = \exp(s) \cdot x_2 + m$
- 行列式为 $\exp(\sum s_i)$，计算简单

其中 $m$ 与行列式计算无关，所以实际上 $m$ 这个操作可以是任意操作（卷积、全连接等）。

### 5.2 矩阵可逆性解释

关于矩阵可逆这部分，其实没怎么看到有人针对这个问题做研究，有一个解释是说：**矩阵可逆的要求很低，行列式只要不为0就代表矩阵可逆，所以随机sample一个矩阵大概率都是可逆的。**

---

## 6. 多尺度结构 (Multi-scale Structure)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image33.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 6.1 Squeeze操作

为了进一步降低计算量，引入多尺度的结构（SPLIT）。

如图所示，原始输入经过第一步flow运算（"flow运算"指的是多个耦合层的复合）后，输出跟输入的大小一样，这时候将输入对半分开两半 $z_1, z_2$（自然也是沿着通道轴），其中 $z_1$ 直接输出，而只将 $z_2$ 送入到下一步flow运算。

### 6.2 交替结构

对于第一层的设计仍会有部分缺陷，一直复制会导致有一部分的信息始终是输入的分布。可以通过交替或者随机打乱的方式解决。

---

## 7. 1x1可逆卷积

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image35.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image24.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

在分块耦合层的部分，一个很重要的操作就是将维度打乱，防止输入的一部分一直复制到输出。

前面提到两种方法：一个是交错开，一个是随机打乱。但是交错开跟随机打乱其实本质上都是实现了一个置换操作。可以把这个置换的矩阵换成一个可学习的参数，本质上就是一个 **1x1的卷积**。

---

## 8. 多个流模型的组合

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image36.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

对生成器的高要求限制了一个生成器的表达能力，所以可以用多个满足这个条件的生成器组成一个大的生成器，**Flow的名字就由此而来**（流的复合）。

---

## 9. 流模型模块结构

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image37.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 10. 结果展示

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image38.gif" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 11. 流模型在底层视觉中的应用

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image5.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

流模型应用到底层视觉有两个特点：

### 11.1 利用最大似然函数

利用最大似然函数去估计概率密度，将传统的回归损失替换成了最大似然损失。基于的动机是：对于一个低质量的图片，其对应的高质量图片是有很多个版本的，反之亦然。

- **回归损失**：将多个高质量的版本取平均得到输出
- **最大似然**：找到多个高质量图片中概率最大的一个图片作为输出

### 11.2 利用可逆网络

设计了一个可逆的网络，通常会训练两个过程（前向过程和反向过程）。因为网络完全可逆，可以利用前向过程和反向过程的不同特点学习不同的东西。

---

## 12. 相关论文

### 12.1 条件流模型 (Conditional Normalizing Flow)

| 论文 | 年份/会议 |
|------|-----------|
| SRFlow: Learning the Super-Resolution Space with Normalizing Flow | ECCV 2020 |
| DehazeFlow: Multi-scale Conditional Flow Network for Single Image Dehazing | MM 2021 |
| Low-Light Image Enhancement with Normalizing Flow | AAAI 2022 |
| StyleFlow: Attribute-conditioned Exploration of StyleGAN | SIGGRAPH 2022 |
| Learning Diverse Tone Styles for Image Retouching | arxiv 2022 |

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image39.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image40.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 12.2 可逆网络 (Invertible Network)

| 论文 | 年份/会议 |
|------|-----------|
| Invertible Denoising Network: A Light Solution for Real Noise Removal | CVPR 2021 |
| WINNet: Wavelet-Inspired Invertible Network for Image Denoising | TIP 2022 |
| Bijective Mapping Network for Shadow Removal | CVPR 2022 |
| Invertible Image Decolorization | TIP 2021 |
| Invertible Image Rescaling | ECCV 2020 |

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image41.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flow_based_model/image42.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 13. 总结

流模型作为一种独特的生成模型，具有以下特点：

1. **精确的对数似然**：可以直接优化最大似然目标
2. **可逆的网络结构**：潜在空间具有很好的可解释性
3. **双向推理**：既可以从潜在空间生成样本，也可以将样本编码回潜在空间

但在底层视觉任务中，流模型主要发挥其两个特点：
- 最大似然估计
- 可逆网络结构

这为图像超分辨率、去噪、去雾、图像增强等任务提供了新的思路。

---

## 参考资料

[1] 苏剑林. (Aug. 26, 2018). 《细水长flow之RealNVP与Glow：流模型的传承与升华》. Retrieved from https://spaces.ac.cn/archives/5807

[2] 苏剑林. (Aug. 11, 2018). 《细水长flow之NICE：流模型的基本概念与实现》. Retrieved from https://spaces.ac.cn/archives/5776

[3] Lilian Weng. Flow-based deep generative models. Lilian-weng.github.io, 2018

[4] Dinh, L., Krueger, D., and Bengio, Y. (2014). NICE: Non-linear Independent Components Estimation. arXiv preprint arXiv:1410.8516.

[5] Dinh, L., Sohl-Dickstein, J., and Bengio, S. (2016). Density Estimation using Real NVP. arXiv preprint arXiv:1605.08803.

[6] Kingma, D. P., Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. Advances in Neural Information Processing Systems, 31.

---

## 感谢聆听

THANKS

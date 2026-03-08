---
layout: post
title: noise model in ISP
date: 2023-09-10 00:00:00
description: 相机raw的噪声来源和建模
tags: ISP noise-model paper image-processing
categories: paper-sharing
---

最近OPPO实习完对噪声模型有了新的理解，近期关于夜景raw降噪的paper比较多，整理一下

## 1. 相机成像物理流程

相机成像过程本质上是将光子转换为数字信号的过程。下面我们从物理原理出发，推导完整的成像公式。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/noise_pipe.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

上图展示了从 **Photon to raw pixel value**（光子到原始像素值）再到 **Raw RGB to sRGB**（经过ISP处理）的过程。

### 1.1 光子入射 (Photon Arrival)

光线照射到图像传感器上，光子以一定的速率到达传感器表面。假设入射光强为 $I$（单位时间单位面积的光子数），曝光时间为 $T$，像素面积为 $A$，则入射光子数为：

$$Y \sim \text{Poisson}( \lambda = I \cdot T \cdot A \cdot \eta )$$

其中 $\eta$ 是量子效率（Quantum Efficiency, QE），表示光子转换为电子的效率。

### 1.2 光电转换 (Photoelectric Conversion)

光子通过光电效应被传感器转换为电子。这一过程遵循**泊松分布**：

$$K \sim \text{Poisson}(Y)$$

产生的光生电子数为：

$$K = Y + n_{\text{shot}}$$

其中 $n_{\text{shot}}$ 是**散粒噪声（Shot Noise）**，其方差等于均值：

$$\text{Var}(K) = K = Y$$

### 1.3 模拟放大 (Analog Gain)

光生电子通过模拟放大器进行放大，得到模拟电压：

$$V = \frac{K \cdot e}{C_{\text{cap}}}$$

其中 $e$ 是元电荷 ($1.602 \times 10^{-19}$ C)，$C_{\text{cap}}$ 是采样电容。

模拟增益（Analog Gain）记为 $G_{\text{analog}}$，放大后的信号为：

$$V_{\text{out}} = G_{\text{analog}} \cdot V$$

### 1.4 模数转换 (ADC)

模拟电压通过ADC转换为数字信号：

$$D = \text{round}\left( \frac{V_{\text{out}}}{V_{\text{ADC}}} \right)$$

其中 $V_{\text{ADC}}$ 是ADC的量化步长。

---

## 2. 噪声模型

相机成像过程中的噪声主要分为两类：**泊松噪声（Shot Noise）** 和 **高斯噪声（Read Noise）**。

### 2.1 散粒噪声 (Shot Noise)

散粒噪声是由光子到达的随机性引起的，符合泊松分布：

$$n_{\text{shot}} \sim \mathcal{N}(0, K)$$

其中 $K$ 是产生的光生电子数。散粒噪声的标准差为：

$$\sigma_{\text{shot}} = \sqrt{K}$$

### 2.2 读出噪声 (Read Noise)

读出噪声是由传感器读出电路引入的，主要包括：

- 热噪声 (Thermal Noise)
- 1/f噪声 (Flicker Noise)
- 固定模式噪声 (FPN)

读出噪声通常可以用高斯分布建模：

$$n_{\text{read}} \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$$

### 2.3 总噪声模型

综合以上，RAW图像的噪声模型可以表示为：

$$y = x + n_{\text{shot}} + n_{\text{read}}$$

其中：
- $x$ 是无噪声的理想信号
- $n_{\text{shot}} \sim \mathcal{N}(0, x)$ （散粒噪声，方差等于信号均值）
- $n_{\text{read}} \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$ （读出噪声，方差为常数）

### 2.4 噪声模型的数学表示

更精确地，考虑泊松-高斯混合噪声模型：

$$y = x + \sqrt{x} \cdot n_{\text{shot}}' + n_{\text{read}}$$

其中 $n_{\text{shot}}' \sim \mathcal{N}(0, 1)$，$n_{\text{read}} \sim \mathcal{N}(0, \sigma_{\text{read}}^2)$。

实际应用中，常用的简化模型为：

$$\sigma_y^2 = \sigma_{\text{shot}}^2 + \sigma_{\text{read}}^2 = \alpha \cdot x + \beta$$

其中：
- $\alpha$ 是与散粒噪声相关的增益系数
- $\beta$ 是读出噪声的方差

---

## 3. 噪声标定

噪声标定的目的是估计上述噪声模型中的参数 $\alpha$ 和 $\beta$。

### 3.1 标定原理

根据噪声模型：

$$\sigma_y^2 = \alpha \cdot x + \beta$$

如果我们能获取多张不同曝光下的图像，计算每张图像的方差和均值，就能通过线性回归拟合出 $\alpha$ 和 $\beta$。

### 3.2 Shot Noise 标定 (全局增益K)

对于 **Shot Noise**，通过**明场图像**（Bright Field）去拟合全局增益 $K$。


**标定步骤**：
1. 拍摄多张不同曝光时间下的均匀光照图像
2. 计算每张图像的均值 $\mu$ 和方差 $\sigma^2$
3. 绘制 $\sigma^2$ vs $\mu$ 曲线
4. 拟合直线得到斜率 $\alpha$

### 3.3 Read Noise 标定 (方差)

对于 **Read Noise**，通过**暗场图像**（Dark Field）计算样本方差去拟合真实噪声方差。

**标定步骤**：
1. 拍摄多张暗场图像（无光照或极低光照）
2. 计算暗场图像的方差
3. 该方差即为读出噪声方差的估计 $\beta$

### 3.4 标定公式汇总

| 噪声类型 | 标定方法 | 关键公式 |
|---------|---------|---------|
| Shot Noise | 明场图像拟合 | $\sigma^2 = \alpha \cdot \mu + \beta$ |
| Read Noise | 暗场图像方差 | $\beta = \text{Var}(\text{dark})$ |

> [1] Radiometric CCD Camera Calibration and Noise Estimation. TPAMI 1994

---

## 4. 为什么在RAW域做去噪？

我们为什么想脱离sRGB域，转而在RAW域做去噪呢？原因如下：

### 4.1 位宽更高，信息更丰富

- 你在屏幕上看到的图片大多是 **8bit** 的（0~255）
- 手机拍摄的RAW图基本都是 **10bit** 的
- 好一些的摄像头一般是 **12bit** 的
- 相机一般是 **14bit** 的

### 4.2 光强与信号值线性关系

光强和信号值之间的关系基本上是线性的。理论上图像提亮2倍只需要对信号直接×2。

### 4.3 噪声符合物理规律

噪声未经过ISP的各种非线性变化，是符合物理规律的，而这些物理规律部分特性已知。

---

## 5. 最新研究进展

### 5.1 ICCV 2021: Rethinking Noise Synthesis and Modeling in Raw Denoising

**两大贡献**：
1. 提出了标定出来的合成噪声仍然不真实，直接从真实噪声里采样会更好。
2. 作者发现目前基于DNN去噪工作在对比传统标定方法的时候，给错了传统标定算法的标定参数（因为相机本身输出的参数不准确），作者重新复现了标定过程发现基于标定的算法效果远好于基于DNN的方法。

### 5.2 CVPR 2022: Dancing under the stars: video denoising in starlight

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image32.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

使用 **GAN** 来合成噪声。

### 5.3 CVPR 2023: DNF - Decouple and Feedback Network

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image33.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

提出一种raw去噪的模型框架，引入中间未解马赛克的无噪图做监督。

### 5.4 ICCV 2023: ExposureDiffusion

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image34.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

引入扩散模型做raw降噪，将扩散模型加噪降噪的过程和相机过程中的噪声结合起来。

### 5.5 Lighting Every Darkness in Two Pairs (CVPR 2023)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image35.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

让网络学标定，其实就是拿合成噪声做预训练，用真实样本微调。

---

## 6. 总结

噪声标定是RAW域去噪的基础。通过物理建模：

1. **Shot Noise**：$\sigma_{\text{shot}} = \sqrt{K}$，与信号强度的平方根成正比
2. **Read Noise**：$\sigma_{\text{read}}$，为常数，与信号无关

总噪声：$\sigma_y^2 = \alpha \cdot x + \beta$

正常光照下的标定已经非常成熟，近几年有一些论文专门针对了极暗场景下的噪声标定，因为光照不足会导致信噪比非常小从而带来更大的噪声。基于物理模型特征去搭建神经网络做raw去噪也许会更好。


## 附：为什么像素尺寸越大噪声越小？（校招影石面试官问题，当时没答上来）

### 问题描述

为什么像素尺寸越大，噪声越小？

### 物理原理

对于**泊松噪声**（Shot Noise），可以视为**均值等于方差**的分布：

$$K \sim \text{Poisson}(\lambda), \quad \text{Var}(K) = \lambda = K$$

即：**方差等于均值**。

### 公式推导

假设有两个像素：

- **小像素**：尺寸为 $1\times 1$，获得的光子数为 $K_1$，噪声标准差为 $\sigma_1 = \sqrt{K_1}$
- **大像素**：尺寸为 $n \times n$（面积是大像素的 $n^2$ 倍），获得的光子数为 $K_2 = n^2 \cdot K_1$，噪声标准差为 $\sigma_2 = \sqrt{K_2} = \sqrt{n^2 \cdot K_1} = n \cdot \sqrt{K_1}$

### 关键问题：放大比较

如果我们把**小像素的信号放大到跟大像素相同的尺度**（比如通过插值或电子放大），会怎样？

设放大系数为 $n^2$，则：

- **放大后的小像素信号**：$K_1' = n^2 \cdot K_1 = K_2$
- **放大后的小像素噪声**：由于泊松噪声的方差与均值成正比，放大后方差也会乘以 $(n^2)^2$：
  $$\sigma_1' = n^2 \cdot \sqrt{K_1} = n^2 \cdot \frac{\sqrt{K_2}}{n} = n \cdot \sqrt{K_2} = n \cdot \sigma_2$$

### 结论

| 像素类型 | 光子数 | 噪声标准差 |
|---------|--------|-----------|
| 小像素（放大后） | $K_2$ | $n \cdot \sigma_2$ |
| 大像素 | $K_2$ | $\sigma_2$ |

由于 $n > 1$，所以 **放大后的小像素噪声比大像素噪声大 $n$ 倍**。

换句话说：**大像素本身能接收到更多的光子，其信噪比（SNR）更高**，因为：
$$\text{SNR} = \frac{K}{\sigma} = \frac{K}{\sqrt{K}} = \sqrt{K}$$

光子数越多，信噪比越高，噪声相对越小。这就是为什么大像素的相机在低光环境下表现更好的原因。


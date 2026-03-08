---
layout: post
title: 噪声模型在ISP中的应用
date: 2023-09-10 00:00:00
description: 深入理解图像信号处理器中的噪声建模与去噪技术
tags: ISP noise-model paper image-processing
categories: paper-sharing
---

# 噪声模型在ISP中的应用

**汇报人：王晨曦**
**2023.09.10**

---

## 1. 引言

通常来说，光线打到传感器产生像素值的过程中会产生噪声，它们通常符合某种特殊的分布，此时去噪会比较有效。如果通过了ISP（Image Signal Processor，图像信号处理器），噪声的分布会无法建模，使得去噪十分困难。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image4.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image5.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image6.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

上图展示了从 **Photon to raw pixel value**（光子到原始像素值）再到 **Raw RGB to sRGB**（经过ISP处理）的过程。

---

## 2. 为什么在RAW域做去噪？

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image7.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

我们为什么想脱离sRGB域，转而在RAW域做去噪呢？原因如下：

### 2.1 位宽更高，信息更丰富

- 你在屏幕上看到的图片大多是 **8bit** 的（0~255）
- 手机拍摄的RAW图基本都是 **10bit** 的
- 好一些的摄像头一般是 **12bit** 的
- 相机一般是 **14bit** 的

### 2.2 光强与信号值线性关系

光强和信号值之间的关系基本上是线性的。理论上图像提亮2倍只需要对信号直接×2。

### 2.3 噪声符合物理规律

噪声未经过ISP的各种非线性变化，是符合物理规律的，而这些物理规律部分特性已知。

---

## 3. 经典论文：A Physics-based Noise Formation Model (CVPR 2020)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image8.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

这是 **CVPR 2020** 的一篇经典论文，提出了基于物理的噪声形成模型，用于极端低光环境下的RAW图像去噪。

---

## 4. 噪声模型详解

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image9.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image10.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image11.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image12.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 5. 噪声的标定

对于 **shot noise**，通过明场图像去拟合全局增益K。

对于 **read noise**，通过大量暗场图像计算样本方差去拟合真实噪声方差。

> [1] Radiometric CCD Camera Calibration and Noise Estimation. TPAMI 1994

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image22.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image23.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image24.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image25.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image26.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 6. 实验结果

### 6.1 合成噪声

标定好噪声参数后可以合成真实噪声。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image27.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 6.2 训练效果

使用合成噪声训练网络的结果，在 **SID数据集** 上去得到比用真实成对数据训练得到的结果更好。

**Note**：之所以合成数据集能达到比直接使用成对数据训练结果更好就是因为合成的噪声是基于物理模型标定好的，使得网络更容易学习到这个去噪的过程，而对于真实数据对的训练，因为真实噪声分布非常复杂，直接依靠网络端到端去学习会非常困难。

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image28.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 6.3 自建数据集

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image29.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image30.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

## 7. 最新研究进展

### 7.1 ICCV 2021: Rethinking Noise Synthesis and Modeling in Raw Denoising

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image31.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

**两大贡献**：
1. 提出了标定出来的合成噪声仍然不真实，直接从真实噪声里采样会更好。
2. 作者发现目前基于DNN去噪工作在对比传统标定方法的时候，给错了传统标定算法的标定参数（因为相机本身输出的参数不准确），作者重新复现了标定过程发现基于标定的算法效果远好于基于DNN的方法。

### 7.2 CVPR 2022: Dancing under the stars: video denoising in starlight

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image32.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

使用 **GAN** 来合成噪声。

### 7.3 CVPR 2023: DNF - Decouple and Feedback Network

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image33.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

提出一种raw去噪的模型框架，引入中间未解马赛克的无噪图做监督。

### 7.4 ICCV 2023: ExposureDiffusion

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image34.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

引入扩散模型做raw降噪，将扩散模型加噪降噪的过程和相机过程中的噪声结合起来。

### 7.5 Lighting Every Darkness in Two Pairs (CVPR 2023)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image35.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

让网络学标定，其实就是拿合成噪声做预训练，用真实样本微调。

---

## 8. 总结与展望

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/noise_model_isp/image5.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 8.1 主要观点

1. **噪声标定很重要**：正常光照下的标定已经非常成熟，近几年有一些论文专门针对了极暗场景下的噪声标定，因为光照不足会导致信噪比非常小从而带来更大的噪声。

2. **极端环境噪声建模鲜有研究**：极夜场景的噪声标定似乎也做到了饱和，缺乏扎实的计算摄影基础很难再做突破。

3. **从sRGB到RAW的转变**：目前低光任务主流还在做RGB域的，也有一些转到了RAW域去做。但是目前看来很少有工作根据噪声物理模型去做去噪，实际上还是一种端到端的任务，只是换了个域。

4. **基于物理模型的潜力**：基于物理模型特征去搭建神经网络做raw去噪也许会更好。

### 8.2 未来方向

- 正常光照下一般做raw去噪会联合解马赛克一起做
- 在早年联合去噪和解马赛克也是一个比较热门的话题
- 最近极夜场景虽然做raw去噪有一些工作，但是很少和解马赛克一起做，或者直接默认一起做了
- 极夜下的联合去噪和解马赛克是否会有新的不同？


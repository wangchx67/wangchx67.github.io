---
layout: post
title: Diffusion Transformer
date: 2026-01-18 00:00:00
description: DiT、MMDiT、DDT 等
tags: diffusion study-recording transformer
categories: notes
---


## 1. DiT (Diffusion Transformer)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/DiT/dit.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 1.1 背景

DiT 是首次将 Transformer 架构应用于扩散模型的工作（2023年），在此之前，扩散模型主要使用 UNet 架构。

### 1.2 核心思想

DiT 将扩散过程视为在 latent 空间上的操作：

1. **VAE Encoder**：将图像编码为 latent
2. **DiT Block**：latent patchfy一维向量进行 Transformer 处理，处理完之后unpatchfy
3. **VAE Decoder**：将 latent 解码回图像

### 1.3 模型结构

DiT 使用标准的 ViT (Vision Transformer) 架构，处理 patch 化的 latent：

```
输入: [B, C, H, W]
  ↓ Patchify (8x8 patches)
[B, (H/8)*(W/8), C*8*8]
  ↓ DiT Blocks (重复12次)
[B, (H/8)*(W/8), C*8*8]
  ↓ Unpatchify
输出: [B, C, H, W]
```

### 1.4 位置编码与条件注入

DiT 中需要注入时间步 t 和类别 c 的条件，有几种方式：

#### 1.4.1 In-context Conditioning

将 t 和 c 作为特殊的 token 追加到序列中：

```
[t] [c] [patch_1] [patch_2] ... [patch_n]
```

简单但效果一般。

#### 1.4.2 Adaptive Layer Norm (AdaLN)

使用自适应归一化：

$$
\text{AdaLN}(x, \gamma, \beta) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

其中 $\gamma, \beta$ 由 t 和 c 预测得到。

---

## 2. MMDiT (Multi-modal Diffusion Transformer)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/DiT/mmdit.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### 2.1 背景

MMDiT 是 **Stable Diffusion 3** 使用的架构，

### 2.2 核心思想

MMDiT 设计了双流DiT(双流分别处理多模态信息)，提出了 **分离的 QKV 投影** 机制，专门针对图像和文本使用不同的投影矩阵，这个设计后来被Flux扩展为双流+单流，双流处理多模态信息后，单流DiT block进行聚合

### 2.3 数据流

```
输入:
  - Latent: [B, C, H, W] -> [B, (H*W), C]
  - Text: [B, L, D]

  ↓ Patchify (图像) + 位置编码

  ↓ MMDiT Blocks (重复24次)
    - Self-Attention (图像 → 图像)
    - Cross-Attention (图像 ← 文本)
    - MLP

  ↓ Unpatchify

输出: [B, C, H, W]
```



## 3. DDT (Decoupled Diffusion Transformer)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/DiT/mmdit.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

近年来 **Diffusion Transformer (DiT)** 在图像生成任务上表现很好，但仍然存在两个问题：

- 训练收敛较慢
- 推理步骤较多

在传统 diffusion transformer 中，每一个 denoising step 通常同时完成两个任务：

1. **提取语义信息（low-frequency semantic information）**
2. **恢复图像细节（high-frequency details）**

但这两个任务实际上是 **相互冲突的**：

- 语义提取希望 **抑制高频噪声**
- 细节恢复需要 **生成高频信息**

因此让同一个 transformer 同时做这两件事会带来优化困难。

核心思想：

> 将语义提取和细节生成两个任务进行解耦。

模型结构：
 
```
noisy latent
     │
     ▼
Condition Encoder
     │
     ▼
Velocity Decoder
     │
     ▼
predicted velocity
```

具体来说：

| 模块 | 作用 |
|-----|-----|
| Encoder | 提取语义信息 |
| Decoder | 生成细节 |

---

特点：

1. 增加 encoder 的容量可以明显提高性能，因为语义建模是 diffusion 过程的核心。Decoder 只负责预测 velocity / noise 主要恢复高频细节，因此可以设计得更小。

2. 跨 timestep 共享 encoder 特征

观察：

在 diffusion 过程中，相邻时间步之间的输入变化很小：

```
x_t
x_{t-1}
```

因此：

```
Encoder(x_t) ≈ Encoder(x_{t-1})
```

DDT 利用这一点，在多个 timestep 之间 **共享 encoder feature**。

这样可以减少 encoder 的重复计算，从而加速推理。

---

代码：

https://github.com/MCG-NJU/DDT/blob/main/src/models/denoiser/decoupled_improved_dit.py

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/DiT/ddt_code.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

主要的变化就是在讲所有的dit blocks分成两部分，一部分是encoder一部分是decoder，同时做repa loss的时候对encoder语义建模的结果做。

---
layout: post
title: Diffusion Transformer
date: 2026-01-18 00:00:00
description: DiT、MMDiT、DDT 等
tags: diffusion study-recording transformer
categories: notes
---

本文介绍 Diffusion Transformer (DiT) 架构的演进历程，从最初的 DiT 到 SD3 使用的 MMDiT，再到解耦的 DDT。

## 1. DiT (Diffusion Transformer)

### 1.1 背景

DiT 是首次将 Transformer 架构应用于扩散模型的工作（2023年），在此之前，扩散模型主要使用 UNet 架构。

### 1.2 核心思想

DiT 将扩散过程视为在 latent 空间上的操作：

1. **VAE Encoder**：将图像编码为 latent
2. **DiT Block**：在 latent 上进行 Transformer 处理
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

#### 1.4.3 AdaLN-Zero（最优）

在 AdaLN 基础上引入可学习的缩放因子：

$$
\text{AdaLN-Zero}(x, \gamma, \beta, \alpha) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta + \alpha \cdot x
$$

DiT 最终采用 **AdaLN-Zero**，并用 MLP 从 t 和 c 预测所有参数。


---

## 2. MMDiT (Multi-modal Diffusion Transformer)

### 2.1 背景

MMDiT 是 Stable Diffusion 3 使用的架构

### 2.2 核心思想

MMDiT 提出了 **分离的 QKV 投影** 和 **空间注意力** 机制：

1. **图像 token**：保持空间结构
2. **文本 token**：通过交叉注意力与图像交互

### 2.3 模型结构

```
输入:
  - Latent: [B, C, H, W] -> [B, (H*W), C]
  - Text: [B, L, D]

  ↓ Patchify (图像) + 位置编码

  ↓ DiT Blocks (重复24次)
    - Self-Attention (图像 → 图像)
    - Cross-Attention (图像 ← 文本)
    - MLP

  ↓ Unpatchify

输出: [B, C, H, W]
```


---

## 3. DDT (Decoupled Diffusion Transformer)

### 3.1 背景

DDT（Decoupled Diffusion Transformer）提出解耦空间信息的思想，认为图像 token 之间的空间关系应该与 token 内容本身分离处理。

### 3.2 核心思想

DDT 的核心观点：

1. **Token 内容**：需要全局交互（理解语义）
2. **空间位置**：只需要局部交互（保持结构）

因此将 Transformer 分为两个路径：
- **内容路径**：处理 token 的语义内容
- **空间路径**：处理 token 的位置信息

### 3.3 模型结构

```
输入: [B, N, D]  (N = H*W 个 tokens)

  ↓ Split
  /    \
内容分支   空间分支
[BN, D]   [BN, D]

  ↓          ↓
Content    Spatial
Attention  Attention

  ↓          ↓
[BN, D]   [BN, D]

  \    /
  Combine
[BN, D]

  ↓ Output
[BN, D]
```
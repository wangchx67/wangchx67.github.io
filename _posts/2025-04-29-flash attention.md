---
layout: post
title: flash attention
date: 2025-04-29 00:00:00
description: 学习Flash Attention
tags: study-recording paper transformer optimization
categories: notes
---

标准Attention的计算复杂度和内存复杂度都是O(N²)，其中N是序列长度。当处理长序列时，Attention机制会占用大量显存。

Flash Attention通过IO-Aware的算法设计，利用GPU显存层次结构，显著减少了HBM（High Bandwidth Memory）的访问次数，将显存复杂度从O(N²)降低到O(N)，同时保持数值精确性。

## 2. 标准Attention

标准Self-Attention的计算公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{N \times d}$：Query矩阵
- $K \in \mathbb{R}^{N \times d}$：Key矩阵
- $V \in \mathbb{R}^{N \times d}$：Value矩阵
- $d$：head dimension
- $N$：序列长度

标准实现的计算过程：
1. 计算 $S = QK^T$，需要O(N²)显存存储注意力分数矩阵
2. 计算 $P = \text{softmax}(S)$，同样需要O(N²)显存
3. 计算 $O = PV$，得到最终输出

## 3. Flash Attention核心—— online Softmax

Flash Attention的核心是**在线softmax算法**。传统softmax需要先计算所有exp值再归一化，这需要两次遍历数据。在线softmax可以单次遍历完成：

```python
def online_softmax(x):
    """在线softmax：单次遍历计算softmax"""
    m = float('-inf')  # 当前最大值
    d = 0              # 当前分母（exp求和）
    result = []

    for xi in x:
        m_new = max(m, xi)           # 更新最大值
        d = d * exp(m - m_new) + exp(xi - m_new)  # 更新分母
        m = m_new

    for xi in x:
        result.append(exp(xi - m) / d)  # 计算softmax值

    return result
```

### 分块计算

Flash Attention将输入分割成小块，逐块计算attention并累积结果。这种方法：
- 避免了存储完整的注意力矩阵
- 利用了GPU的SRAM进行快速计算
- 减少了HBM读写次数

### 数值稳定性

通过维护最大值m和缩放因子d，确保数值稳定性，避免softmax溢出。

## 4. 官方源码示例

Flash Attention的官方实现托管在[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)。以下是核心kernel的简化版本：

### 4.1 Flash Attention 2 前向传播

```python
# 官方实现核心逻辑（简化版）
# 原始代码: https://github.com/Dao-AILab/flash-attention

import math
 
def flash_attention_fwd(q, k, v):
    """
    Flash Attention前向传播简化实现
    Q, K, V: [batch, num_heads, seq_len, head_dim]
    """
    N, d = q.shape

    # 初始化输出和辅助变量
    output = torch.zeros_like(q)

    # 块大小（实际实现中由GPU硬件决定）
    BLOCK_SIZE = 128

    # 遍历Q的块
    for i in range(0, N, BLOCK_SIZE):
        q_block = q[i:i+BLOCK_SIZE]

        # 初始化该块的输出和统计量
        m_i = torch.full((q_block.shape[0],), -float('inf'), device=q.device)
        d_i = torch.zeros(q_block.shape[0], device=q.device)
        o_i = torch.zeros_like(q_block)

        # 遍历K,V的块
        for j in range(0, N, BLOCK_SIZE):
            k_block = k[j:j+BLOCK_SIZE]
            v_block = v[j:j+BLOCK_SIZE]

           K^T / # 计算Q sqrt(d)
            s = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(d)

            # 在线softmax计算
            m_new = torch.maximum(m_i, torch.max(s, dim=-1))

            # 数值稳定处理
            s_minus_m = torch.exp(s - m_new.unsqueeze(-1))

            d_new = d_i * torch.exp(m_i - m_new) + torch.sum(s_minus_m, dim=-1)

            # 更新输出
            p = s_minus_m / s_minus_m.sum(dim=-1, keepdim=True)
            o_i = (o_i * (d_i * torch.exp(m_i - m_new)).unsqueeze(-1) +
                   torch.matmul(p, v_block)) / d_new.unsqueeze(-1)

            m_i = m_new
            d_i = d_new

        output[i:i+BLOCK_SIZE] = o_i

    return output
```
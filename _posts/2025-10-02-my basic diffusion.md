---
layout: post
title: My Basic Diffusion
date: 2025-10-02 00:00:00
description: 主要介绍我的diffusion模型学习记录
tags: diffusion deep-learning
categories: notes
---

## My basic diffusion

主要介绍我的diffusion模型学习记录

### Basic theory

#### forward（加噪）

对于一个真实数据域 $\mathbf{x}_0 \sim q(\mathbf{x})$ ，加噪 $T$ 步得到 noise sample sequences  
$\mathbf{x}_1, \dots, \mathbf{x}_T$ ，通过 $beta$ 控制步数，其中 $\{\beta_t \in (0, 1)\}_{t=1}^T$ ：

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$

$$
\begin{aligned}
\mathbf{x}_t
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

这里构建起直接的 $x_{0},x_{t}$ 之间的联系。

#### backward（降噪）

去噪就是在求 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ ，然而 $ q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}$ ，其中$q(\mathbf{x}_{t} )$ 是需要依赖 $q(x_{0})$ 的，我们当前无法获得真实的数据域分布，基于 $ x_{0},x_{t} $ 之间的直接联系，并借助贝叶斯公式可以求得：

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; {\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), {\tilde{\beta}_t} \mathbf{I})
$$

$$
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\

\end{aligned}
$$

$$
... (略过推导) \\
\begin{aligned}
\tilde{\beta}_t
&
= {\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t

&= {\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$

#### training objective

生成器为 $p_{\theta}()$, 可以通过类似VAE的变分下界去最大似然优化（optimize the negative log-likelihood）:

$$
\begin{aligned}
- \log p_\theta(\mathbf{x}_0)
&\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) & \small{\text{; KL is non-negative}}\\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
\text{Let }L_\text{VLB}
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
\end{aligned}
$$

$$
\begin{aligned}
L_\text{VLB}
&= (略过推导) \\

&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$

主要优化目标为$L_{t-1}$（对于t+1步则为$L_{t}$）

生成器在扩散模型中的形式是去噪，以$x_{T}$为输入，预测$x_{t-1}$，相当于 $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$，其中${\mu}_\theta$为作为预测的网络，根据前面的推导：

$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= {\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$

最终目标则为：

$$
\begin{aligned}
L_t
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| {\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - {\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| {\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - {\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$

简化后为：

$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$

与DDPM论文一致。

### DDIM



### 条件生成



### 不同预测方式以及loss方式

#### $x_{0}$ （直接预测图像）

#### $\epsilon$  （预测噪声）

#### $v$ （预测速度）

### diffusion中常用设计与结构

#### DiT, Diffusion Transformer

#### ROPE, 旋转位置编码

[LLM学习记录（五）--超简单的RoPE理解方式 - 知乎](https://zhuanlan.zhihu.com/p/642289220)

对于每 2 个相邻的维度 $(x_{2i},x_{2i+1})$，定义旋转矩阵：

$$
\mathbf{R}_m^{(i)} = \begin{pmatrix} 
\cos(m\theta_i) & -\sin(m\theta_i) \\ 
\sin(m\theta_i) & \cos(m\theta_i) 
\end{pmatrix}
$$

其中旋转角度 $θ_{i}$ 按频率衰减(由于正余弦函数的周期性，保证尾部与头部的距离不比中间任意两个位置的距离短，theta一般取无法达到的数字，一般采用10000)：

$$
\theta_i = \text{theta}^{-\frac{2i}{d}}, \quad i \in \{0, 1, \dots, d/2-1\}
$$

应用旋转矩阵后，每个维度对的变换为：

$$
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = 
\begin{pmatrix} 
\cos(m\theta_i) & -\sin(m\theta_i) \\ 
\sin(m\theta_i) & \cos(m\theta_i) 
\end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

展开得到高效计算公式：

$$
\boxed{
\begin{aligned}
x_{2i}' &= x_{2i} \cdot \cos(m\theta_i) - x_{2i+1} \cdot \sin(m\theta_i) \\
x_{2i+1}' &= x_{2i} \cdot \sin(m\theta_i) + x_{2i+1} \cdot \cos(m\theta_i)
\end{aligned}
}
$$

Attention 分数计算时，RoPE 展现出美妙的数学性质(注意力分数只依赖于**相对位置** n−m*n*−*m*)：

$$
\begin{aligned}
\langle \mathbf{q}_m', \mathbf{k}_n' \rangle 
&= \langle \mathbf{R}_m \mathbf{q}_m, \mathbf{R}_n \mathbf{k}_n \rangle \\
&= \langle \mathbf{q}_m, \mathbf{R}_m^\top \mathbf{R}_n \mathbf{k}_n \rangle \\
&= \langle \mathbf{q}_m, \mathbf{R}_{n-m} \mathbf{k}_n \rangle
\end{aligned}
$$

RoPE 可以用复数乘法优雅地表示：

$$
z' = z \cdot e^{i m\theta} = (x_{\text{real}} + i x_{\text{imag}}) \cdot (\cos(m\theta) + i\sin(m\theta))
$$

diffusers中的实现方式，截取flux对应的方式（diffusers/models/embeddings.py）：

```python
def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin
   
def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis  # [S, D]
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

    cos, sin = cos.to(x.device), sin.to(x.device)

    if use_real_unbind_dim == -1:
        # Used for flux, cogvideox, hunyuan-dit
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    elif use_real_unbind_dim == -2:
        # Used for Stable Audio, OmniGen, CogView4 and Cosmos
        x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
        x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
```

代码中的旋转操作等价于矩阵乘法：

$$
\begin{aligned}
\text{x\_rotated} &= [-x_{\text{imag}}, x_{\text{real}}] \quad \text{(旋转90°)} \\
\text{out} &= x \cdot \cos\theta + \text{x\_rotated} \cdot \sin\theta \\
&= \begin{pmatrix} x_{\text{real}} \\ x_{\text{imag}} \end{pmatrix} \cos\theta + 
   \begin{pmatrix} -x_{\text{imag}} \\ x_{\text{real}} \end{pmatrix} \sin\theta \\
&= \begin{pmatrix} 
   x_{\text{real}}\cos\theta - x_{\text{imag}}\sin\theta \\
   x_{\text{imag}}\cos\theta + x_{\text{real}}\sin\theta
   \end{pmatrix}
\end{aligned}
$$

$$
\phi_{m,i} = m \cdot \theta_i, \quad \text{freqs} = \text{outer}(\text{pos}, \theta) \in \mathbb{R}^{S \times d/2}
$$



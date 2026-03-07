---
layout: post
title: Architecture design of diffusion model
date: 2025-10-18 00:00:00
description: 学习下目前扩散模型中的架构设计
tags: study-recording
categories: notes
---

学习下目前扩散模型中的架构设计

### diffusion中常用设计与结构

#### DiT, Diffusion Transformer

#### ROPE, 旋转位置编码


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



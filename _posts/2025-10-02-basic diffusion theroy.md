---
layout: post
title: Baisc diffusion theroy
date: 2025-10-02 00:00:00
description: 推导一下扩散模型
tags: diffusion study-recording
categories: notes
---

推导一下扩散模型

### Basic theory

#### forward（加噪）

对于一个真实数据域 $x_{0} \sim q(x)$ ，加噪 $T$ 步得到 noise sample sequences $ x_{1}, \dots, x_{T} $ ，通过 $ \beta $ 控制步数，其中 $ \{\beta_{t} \in (0, 1)\}_{t=1}^T$ ：

$$
q(\mathbf{x}_{t} \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_{t}; \sqrt{1 - \beta_{t}} \mathbf{x}_{t-1}, \beta_{t}\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0}) = \prod^T_{t=1} q(\mathbf{x}_{t} \vert \mathbf{x}_{t-1})
$$

$$
\begin{aligned}
\mathbf{x}_{t}
&= \sqrt{\alpha_{t}}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_{t}}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_{t} \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t} \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t}}\boldsymbol{\epsilon} \\
q(\mathbf{x}_{t} \vert \mathbf{x}_{0}) &= \mathcal{N}(\mathbf{x}_{t}; \sqrt{\bar{\alpha}_{t}} \mathbf{x}_{0}, (1 - \bar{\alpha}_{t})\mathbf{I})
\end{aligned}
$$

这里构建起直接的 $x_{0},x_{t}$ 之间的联系。

#### backward（降噪）

去噪就是在求 $q(x_{t-1} \vert x_{t})$ ，然而 $ q(x_{t-1} \mid x_{t}) = \frac{q(x_{t} \mid x_{t-1}) \cdot q(x_{t-1})}{q(x_{t})}$ ，其中$q(x_{t} )$ 是需要依赖 $q(x_{0})$ 的，我们当前无法获得真实的数据域分布，基于 $ x_{0},x_{t} $ 之间的直接联系，并借助贝叶斯公式可以求得：

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t}, \mathbf{x}_{0}) = \mathcal{N}(\mathbf{x}_{t-1}; {\tilde{\boldsymbol{\mu}}}(\mathbf{x}_{t}, \mathbf{x}_{0}), {\tilde{\beta}_{t}} \mathbf{I})
$$

$$
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t}, \mathbf{x}_{0})
&= q(\mathbf{x}_{t} \vert \mathbf{x}_{t-1}, \mathbf{x}_{0}) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_{0}) }{ q(\mathbf{x}_{t} \vert \mathbf{x}_{0}) } \\

\end{aligned}
$$

$$
... (略过推导) \\
\begin{aligned}
\tilde{\beta}_{t}
&
= {\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \cdot \beta_{t}} \\
\tilde{\boldsymbol{\mu}}_{t}

&= {\frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x}_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}} \boldsymbol{\epsilon}_{t} \Big)}
\end{aligned}
$$

#### training objective

生成器为 $p_{\theta}()$, 可以通过类似VAE的变分下界去最大似然优化（optimize the negative log-likelihood）:

$$
\begin{aligned}
- \log p_\theta(\mathbf{x}_{0})
&\leq - \log p_\theta(\mathbf{x}_{0}) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_{0}) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_{0}) ) & \small{\text{; KL is non-negative}}\\
&= - \log p_\theta(\mathbf{x}_{0}) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_{0})} \Big] \\
&= - \log p_\theta(\mathbf{x}_{0}) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_{0}) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
\text{Let }L_\text{VLB}
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_{0})} \log p_\theta(\mathbf{x}_{0})
\end{aligned}
$$

$$
\begin{aligned}
L_\text{VLB}
&= (略过推导) \\

&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_{T} \vert \mathbf{x}_{0}) \parallel p_\theta(\mathbf{x}_{T}))}_{L_{T}} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t}, \mathbf{x}_{0}) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_{t}))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_{0} \vert \mathbf{x}_{1})}_{L_{0}} ]
\end{aligned}
$$

T和0都是常数，所以主要优化目标为$L_{t-1}$（对于t+1步则为$L_{t}$）

生成器在扩散模型中的形式是去噪，以$x_{t}$为输入，预测$x_{t-1}$，相当于 $p_\theta(x_{t-1} \vert x_{t}) = \mathcal{N}(x_{t-1}; \mu_\theta(x_{t}, t), \Sigma_\theta(x_{t}, t))$，其中$\theta$是网络参数，根据前面的推导：

$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_{t}, t) &= {\frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x}_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_{t}, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x}_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_{t}, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_{t}, t))
\end{aligned}
$$

最终目标则为：

$$
\begin{aligned}
L_{t}
&= \mathbb{E}_{\mathbf{x}_{0}, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_{t}, t) \|^2_2} \| {\tilde{\boldsymbol{\mu}}_{t}(\mathbf{x}_{t}, \mathbf{x}_{0})} - {\boldsymbol{\mu}_\theta(\mathbf{x}_{t}, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0}, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| {\frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x}_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}} \boldsymbol{\epsilon}_{t} \Big)} - {\frac{1}{\sqrt{\alpha_{t}}} \Big( \mathbf{x}_{t} - \frac{1 - \alpha_{t}}{\sqrt{1 - \bar{\alpha}_{t}}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_{t}, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0}, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_{t})^2 }{2 \alpha_{t} (1 - \bar{\alpha}_{t}) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_{t} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_{t}, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0}, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_{t})^2 }{2 \alpha_{t} (1 - \bar{\alpha}_{t}) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_{t} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t}}\boldsymbol{\epsilon}_{t}, t)\|^2 \Big]
\end{aligned}
$$

简化后为：

$$
\begin{aligned}
L_{t}^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_{0}, \boldsymbol{\epsilon}_{t}} \Big[\|\boldsymbol{\epsilon}_{t} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_{t}, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_{0}, \boldsymbol{\epsilon}_{t}} \Big[\|\boldsymbol{\epsilon}_{t} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_{t}}\boldsymbol{\epsilon}_{t}, t)\|^2 \Big]
\end{aligned}
$$

与DDPM论文一致。

### DDIM

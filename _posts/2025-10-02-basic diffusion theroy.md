---
layout: post
title: Baisc diffusion theroy
date: 2025-10-02 00:00:00
description: 推导一下扩散模型
tags: diffusion study-recording
categories: notes
---

推导一下扩散模型，主要还是基于[Lil' log What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)这篇博客进行学习并自行推导。

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

去噪就是在求 $q(x_{t-1} \vert x_{t})$ ，然而 $ q(x_{t-1} \mid x_{t}) = \frac{q(x_{t} \mid x_{t-1}) \cdot q(x_{t-1})}{q(x_{t})}$ ，其中$q(x_{t} )$ 是需要依赖 $q(x_{0})$ 的，基于 $ x_{0},x_{t} $ 之间的直接联系，并借助贝叶斯公式可以求得：

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

与DDPM论文一致。实际训练过程为，sample一个噪声$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ，对于某特随机选择的timestep $t$ ，计算 $x_{t}$ ，并将其与时间条件一起作为网络的输入预测噪声，计算预测结果与sample出来噪声的MSE来优化网络。

### DDIM

DDIM本质上在DDPM的基础上在反向生成过程中去掉了随机性，首先引入参数 $\sigma_t$ 对生成过程重参数化：

$$
\begin{aligned}
\mathbf{x}_{t-1} 
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 +  \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} & \\
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} & \\
&= \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t) + \sigma_t\boldsymbol{\epsilon} \\
q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
\end{aligned}
$$

可以发现网络参数固定后，仍然有随机性在，这就导致无法跨多步，因此只要去掉这个随机性，将$\sigma_t$置零，就可以在采样的时候跨多步预测噪声（deterministic generation），这个性质也非常好地帮助对预训练diffusion进行蒸馏。


### 条件生成

#### Classifier Guided Genration

基于预训练的生成模型，在生成每一步用一个分类器的梯度去引导往某个特定条件去生成，结合score-function进行推导，无条件DDPM的score-function为：
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$
这个梯度在score-matching的时候跟diffusion做backward生成去掉的噪声是一直的，在此基础上加入条件$y$以及分类器$f_\phi(.)$:
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y)
&= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\
&\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t))
\end{aligned}
$$
新的梯度信息（噪声）$\nabla_{x_{t}}\log q(\mathbf{x}_t, y)$代替原本的$\nabla_{x_{t}}\log q(\mathbf{x}_t)$参与生成，随着timestep慢慢引导生成目标图像。

#### Classifier-Free Guidance

不基于分类器的生成需要从头训练，基于原本的优化目标建立基于条件$y$的联合分布，即$q(x_{t-1} \vert x_{t})$ 变成$q(x_{t-1} \vert x_{t},y)$，相当于除了时间条件t加了一个其他条件$y$，由于新增了一个条件，可以通过计算condition与uncondtion的结果做差，使得生成更往条件的方向去靠，仍旧基于score-function推导：

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t)
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big) \\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$

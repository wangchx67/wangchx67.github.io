---
layout: post
title: Different diffusion prediction and loss
date: 2026-02-10 00:00:00
description: 学习噪声/图像/速度预测以及对应的优化方式
tags: diffusion study-recording
categories: notes
---
 
### DDPM范畴下的不同预测方式

最早的DDPM即采用噪声预测的方式，加噪过程为

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

$$
\epsilon \sim \mathcal{N}(0,I)
$$

Loss直接基于预测的噪声和对应的target噪声算，形式为：

$$
L_t^{simple}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\|
\epsilon -
\epsilon_\theta(x_t,t)
\|^2
\right]
$$

从$\epsilon$恢复到$x_{0}$：

$$
x_0 =
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
$$

基于 $\epsilon_\theta$ 为噪声网络预测器的话，

$$
x_{0,\theta}
=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}
$$

$$
L_t
=
\mathbb{E}
\left[
\|
x_0 - x_{0,\theta}(x_t,t)
\|^2
\right]
$$

假如 $x_{0,\theta}$ 不通过 $\epsilon_\theta$ 推导，直接预测 $x_{0}$ ，那么从 $x_{0}$ 推 $\epsilon$ 的话：

$$
\epsilon
=
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}
$$

$$
\epsilon_\theta
=
\frac{x_t-\sqrt{\bar{\alpha}_t}x_{0,\theta}}{\sqrt{1-\bar{\alpha}_t}}
$$

定义速度 $v$ :

$$
v =
\sqrt{\bar{\alpha}_t}\epsilon
-
\sqrt{1-\bar{\alpha}_t}x_0
$$

基于速度预测网络 $v_\theta(x_t,t)$ ， 优化目标为：

$$
L_t
=
\mathbb{E}
\left[
\|
v - v_\theta(x_t,t)
\|^2
\right]
$$

从 $v$ 恢复到 $x_0$ ：

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

$$
v =
\sqrt{\bar{\alpha}_t}\epsilon -
\sqrt{1-\bar{\alpha}_t}x_0
$$

可以得到：

$$
x_0 =
\sqrt{\bar{\alpha}_t}x_t -
\sqrt{1-\bar{\alpha}_t}v
$$

同理得到 $\epsilon$ :

$$
\epsilon =
\sqrt{1-\bar{\alpha}_t}x_t +
\sqrt{\bar{\alpha}_t}v
$$

总结来说，三种prediction可以两两互相转换：

$$
x_0 =
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
$$

$$
\epsilon =
\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}
$$

$$
x_0 =
\sqrt{\bar{\alpha}_t}x_t -
\sqrt{1-\bar{\alpha}_t}v
$$

$$
\epsilon =
\sqrt{1-\bar{\alpha}_t}x_t +
\sqrt{\bar{\alpha}_t}v
$$

三种预测方式在理论上都一致，但是在不同的timestep下，使用不同的方式去做loss会有比较大区别，回顾加噪方式：

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

以 $\epsilon-predicton$ 为base，

$$
L_\epsilon
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\|
\epsilon -
\epsilon_\theta(x_t,t)
\|^2
\right]
$$

该方式下在高噪声下好学，低噪声下不好学(输出结果噪声的权重 {% raw %}$\sqrt{1-\bar{\alpha}_t}${% endraw %} 会趋于0)。

对于 $x_0-prediction$ ， 因为 

$$
x_0 =\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}
$$ 

{% raw %}
$$
L_{x_0}
=\frac{{1-\bar{\alpha}_t}}{{\bar{\alpha}_t}}L_\epsilon
$$
{% endraw %}

此时当timestep较小时，Loss趋于正无穷，很难学；当较大时是与 $\epsilon-predicton$ 类似；

对于 $v-prediction$ , 由于：

$$
v =
\sqrt{\bar{\alpha}_t}\epsilon -
\sqrt{1-\bar{\alpha}_t}x_0
$$

$$
L_{v}
=\bar{\alpha}_tL_{x_0} + （1-\bar{\alpha}_t） L_\epsilon
$$

权衡了上述两种的优势，timestep较小时随 $L_\epsilon$ , 较大时随 $L_{x_0}$ 。

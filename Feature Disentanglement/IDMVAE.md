# IDMVAE: Disentanglement of Variations with Multimodal Generative Modeling

- 发表在：International Conference on Learning Representations 2026 (ICLR 2026)
- 作者：Yijie Zhang、Yiyang Shen、Weiran Wang **Department of Computer Science University of Iowa Iowa City, IA 52242, USA**
- [论文原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/581e1a06fa20f2c079dc5fb2db236335-Abstract-Conference.html)
- [官方代码](https://github.com/InterstellarJ2049/IDMVAE)

## 问题分析
现有工作将多模态信息分为共有与特有，但二者容易出现信息泄漏和冗余。

共有信息进入特有表示，特有信息也可能留在共有表示中，会影响跨模态生成一致性。

多模态VAE常使用简单高斯分布作为潜变量先验，实际潜变量分布可能更加复杂，简单先验不一定能充分拟合。

## 方法

### 1. 编码&重建
以MMVAE+为基础，使用单模态VAE的共有和私有编码器将图文表示分成四路近似后验分布。

$$
x_m
\longrightarrow
\begin{cases}
q(z\mid x_m) & \text{共有编码器}\\
q(w_m\mid x_m) & \text{私有编码器}
\end{cases}
$$

$$
z_m\sim q(z\mid x_m),\qquad
w_m\sim q(w_m\mid x_m).
$$

#### 同模态重建

通过解码器重建原始输入

$$
x_{\mathrm{m}}
\rightarrow
(z_{\mathrm{m}},w_{\mathrm{m}})
\rightarrow
\hat{x}_{\mathrm{m}}.
$$

#### 跨模态重建

从辅助先验概率分布中采样$\tilde w_n$替换原本模态内的私有信息进行重建。

$$
z_m\sim q(z\mid x_m),\qquad
\tilde w_n\sim r(\tilde w_n),
$$

$$
(z_m,\tilde w_n)
\hat{x}_n,\quad m\ne n.
$$


使用MoE（Mixture of Experts，专家混合）聚合各模态共有后验：

$$
q(z\mid X)=\frac{1}{M}\sum_{m=1}^{M}q(z\mid x_m),\qquad X=\{x_1,\ldots,x_M\}.
$$

MMVAE+目标：

$$
\mathcal L_{\mathrm{MMVAE+}}=-\frac{1}{M}\sum_{m=1}^{M}
\mathbb E_{\substack{z\sim q(z\mid x_m),\;w_m\sim q(w_m\mid x_m)\\
\{\tilde w_n\sim r(\tilde w_n)\}_{n\ne m}}}
\left[\log\left(
\frac{p(x_m\mid z,w_m)p(z)p(w_m)}{q(z\mid X)q(w_m\mid x_m)}
\prod_{n\ne m}p(x_n\mid z,\tilde w_n)
\right)\right].
$$


简单、无结构的先验可能不适合表示学习：学到的潜变量可能具有复杂结构，例如不同类别的表示形成彼此分离的聚类，而不是服从单峰高斯分布。使用扩散模型建模先验 $p(z)$，通过从纯噪声开始的去噪过程生成潜变量，以提高先验的表达能力。

为了将扩散模型引入损失，将 KL 散度展开为：

$$
D_{\mathrm{KL}}\bigl(q(z\mid x)\Vert p(z)\bigr)=\mathbb E_{q(z\mid x)}[\log q(z\mid x)]+\mathbb E_{q(z\mid x)}[-\log p(z)].\tag{3}
$$


### 2. 跨视图互信息

使用对比学习匹配同一样本不同模态的共有表示。

$$
I(z_m; z_n) ≈\operatorname{Contrast}(z_m,z_n)=\mathbb E
\log
\frac{\phi(z_m,z_n)}
{\phi(z_m,z_n)+\sum_{j=1}^{k}\phi(z_m,\bar z_n^j)}.
\tag{1}
$$

$$
\phi(z_m,z_n)=
\exp\left(
\frac{z_m^\top z_n}{\|z_m\|\|z_n\|}
\right).
$$




跨视图互信息损失：

$$
\mathcal L_{\mathrm{CrossMI}}=-\frac{2}{M(M-1)}\sum_{m<n}\operatorname{Contrast}(z_m,z_n).
$$


### 3. 生成式增强

减少共有和私有内的信息冗余

对模态m取两条样本

$$
x_m,\qquad x'_m.
$$

编码采样

$$
(z_m,w_m)\leftarrow x_m,
\qquad
(z'_m,w'_m)\leftarrow x'_m.
$$

交换组合，生成新样本：

$$
x_m^+\sim p(x_m\mid z_m,w'_m).
$$

再编码

$$
z_m^{\prime\prime}\sim q(z\mid x_m^+),\qquad
w_m^{\prime\prime}\sim q(w_m\mid x_m^+).
$$

目标

$$
z_m^{\prime\prime}\leftrightarrow z_m,
\qquad
w_m^{\prime\prime}\leftrightarrow w'_m.
$$

共有与私有表示损失

$$
\mathcal L_{\mathrm{GenAug},z_m}=-\operatorname{Contrast}(z_m^{\prime\prime},z_m),
\qquad
\mathcal L_{\mathrm{GenAug},w_m}=-\operatorname{Contrast}(w_m^{\prime\prime},w'_m).
$$

生成式增强损失

$$
\mathcal L_{\mathrm{GenAug}}=\frac{1}{2M}
\sum_{m=1}^{M}
\left(
\mathcal L_{\mathrm{GenAug},z_m}+\mathcal L_{\mathrm{GenAug},w_m}
\right).
$$





### 4. 联合训练目标



$$
\min\mathcal L_{\mathrm{IDMVAE}}=\mathcal L_{\mathrm{MMVAE+}}+\lambda_1\mathcal L_{\mathrm{CrossMI}}+\lambda_2\mathcal L_{\mathrm{GenAug}}.\tag{2}
$$



## 实验

- PolyMNIST-Quadrant：五个视图共享数字类别，各视图的象限位置为私有标签。
- CUB-HQ：鸟类图片与文本描述；图片先由预训练VAE转换为$4\times32\times32$张量，文本使用one-hot表示。图像使用ResNet，文字使用CNN。
- TCGA：mRNA与miRNA两个组学视图，使用MLP编码和解码，预测一年死亡指标。

### 结果

PolyMNIST-Quadrant上，使用线性分类器评价潜变量：

| 模型 | $z\rightarrow$数字 ↑ | $z\rightarrow$象限 ↓ | $w\rightarrow$象限 ↑ | $w\rightarrow$数字 ↓ |
|---|---:|---:|---:|---:|
| MMVAE+ | 0.382 | 0.355 | 0.999 | 0.341 |
| IDMVAE | 0.983 | 0.271 | 0.999 | 0.162 |
| IDMVAE + 扩散先验 | 0.982 | 0.267 | 0.999 | 0.143 |



CUB-HQ上，文本生成图片的FID：MMVAE+为70.157，IDMVAE为64.435，加入扩散先验后为60.549，越低越好。

TCGA上，使用 $z+w$ 的预测准确率：MMVAE+为 $0.690\pm0.011$，IDMVAE为 $0.718\pm0.017$，加入扩散先验后为 $0.731\pm0.019$。



# D3ER: Supporting Multi-Modal Recommendation via Disentangle  and Distillation-based Dynamic Ensemble

- 发表在：*Proceedings of the 34th ACM  International Conference on Multimedia (MM ’26),*  [论文链接](https://arxiv.org/pdf/2608.25737) [项目代码]()

- 作者 Bingnan Wang, Yi Li, Xiongxin Tang, Fanjiang Xu, 和 Jiangmeng Li. **中国科学院软件研究所**



## 问题分析
与DHMRec解决的问题类似但：
特征解耦之后的模块进行朴素融合往往会削弱各自的独立有效性。


## 方法

### FCD（feature component disentanglement）模块

FCD主要做三件事：
1、实例级对齐：使用对比学习让同一件商品的图文共有表示相互匹配。
2、分布级对齐：解决对比学习中存在的假负样本问题。
3、分离共有表示与私有表示：使用欧氏距离衡量不相似性。

对于预提取的模态特征 $`x_m`$，使用MLP将其分为共有特征与私有特征。

$$
z_m^c=f_{\theta_m^c}(x_m),\qquad z_m^s=f_{\theta_m^s}(x_m)\tag{1}
$$

MLP有四个： $`f_{\theta_v^c}`$、 $`f_{\theta_v^s}`$、 $`f_{\theta_t^c}`$、 $`f_{\theta_t^s}`$，对应得到： $`z_v^c`$、 $`z_v^s`$、 $`z_t^c`$、 $`z_t^s`$。

更新MLP参数的联合约束

$$
\mathcal{L}_{\mathrm{FCD}}=\alpha_c\mathcal{L}_{cl}+\alpha_w\mathcal{L}_{wd}+\mathcal{L}_{sep}\tag{8}
$$

<details>

<summary>损失函数</summary>

实例级对齐损失

$$
\mathcal{L}_{cl}=\sum_{m_1,m_2\in\mathcal{M}}^{m_1\ne m_2}\sum_{i\in\mathcal{I}}\frac{\exp\left(z_{m_1,i}^c (z_{m_2,i}^c)^\top / \tau\right)}{\displaystyle\sum_{j\in\mathcal{I}}\sum_{m\in\mathcal{M}}\exp\left(z_{m_1,i}^c (z_{m,j}^c)^\top / \tau\right)}.\tag{2}
$$

仅使用实例级对齐容易出现假负样本问题，例如两件商品款式和颜色都很相近，只有品牌不一样，虽然不是同一件商品但也共享了很多有用的语义。会产生把相似语义也分开的问题。


分布级对齐损失

$$
\mathcal{L}_{wd}=\sum_{m_1,m_2\in\mathcal{M}}^{m_1\ne m_2}\mathcal{W}_p\left(\mathcal{P}(z_{m_1}^c),\mathcal{P}(z_{m_2}^c)\right).\tag{5}
$$

Wasserstein距离：把概率分布 $`P_1`$ 的概率质量搬运成 $`P_2`$ 的分布，最少需要付出多少代价。

$$
W_p(\mathcal{P}_1,\mathcal{P}_2)=\left[\inf_{\mu\in\Gamma(x_1,x_2)}\int d(x_1,x_2)^p\,d\mu(x_1,x_2)\right]^{\frac{1}{p}}.\tag{3}
$$

由于计算成本过高，原文采用Radon变换的线性积分方法计算Wasserstein距离衡量图文分布的差异


$$
\mathcal{W}_p(\mathcal{P}_1,\mathcal{P}_2)=\left[\int_{\mathbb{S}^{d_\theta-1}}W_p\left(\mathcal{R}\mathcal{P}_1(\cdot,\theta),\mathcal{R}\mathcal{P}_2(\cdot,\theta)\right)\,d\theta\right]^{\frac{1}{p}}.\tag{4}
$$


共有表示融合

Wasserstein重心，找到一种分布使它到图片分布及文本分布的距离尽可能小。

$$
z^c=\min_z\sum_{m\in\mathcal{M}}\mathcal{W}_p\left(\mathcal{P}(z_m^c),\mathcal{P}(z)\right).\tag{6}
$$


### 模态内分离

使用L2距离衡量不相似性。

$$
\mathcal{L}_{sep}=-\min\left(\sum_{m\in\mathcal{M}}\sum_{i\in\mathcal{I}}\left\|z_{m,i}^c-z_{m,i}^s\right\|_2^2,
d_m\right).\tag{7}
$$

</details>




## KDBoost（knowledge distillation-enhanced gradient boosting）模块

建立三个推荐模型，模型骨干网络在三次试验中分别采用MGCN、DiffMM，PGL。

给每一路建立一个推荐模型，合起来称为累计模型：

$$
\mathcal{G}_\Theta=\lbrace g_{\vartheta^c},g_{\vartheta_v^s},g_{\vartheta_t^s}\rbrace
$$

三个分支的分数相加

$$
S(u,i)=\mathcal{G}_\Theta(x,u,i)=\sum_{g_\vartheta\in\mathcal{G}_\Theta}g_\vartheta(x,u,i)\tag{9}
$$

BPR损失

$$
\begin{aligned}
\mathcal{L}_{\mathrm{BPR}}
&=-\sum_{(u,i,j)\in\mathcal{O}}
\log\left(\sigma\left(\Delta S(u,i,j)\right)\right), \\
\Delta S(u,i,j)&=S(u,i)-S(u,j).
\end{aligned}
\tag{10}
$$


让选中分支拟合 BPR 的负梯度。

$$
\begin{aligned}
\mathcal{L}_{\mathrm{boost}}^{\prime}
&=\sum_{(u,i,j)\in\mathcal{O}}
\left(
\Delta S_{k}^{\ast}
-\left(-\frac{
\partial\mathcal{L}_{\mathrm{BPR}}\left(\mathrm{sg}\left(\Delta S_{k-1}\right)\right)
}{\partial\Delta S_{k-1}}\right)
\right)^{2} \\
&=\sum_{(u,i,j)\in\mathcal{O}}
\left(
\Delta S_{k}^{\ast}
-\left(1-\sigma\left(\mathrm{sg}\left(\Delta S_{k-1}\right)\right)\right)
\right)^{2}.
\end{aligned}
\tag{11}
$$

$`S_1`$ 使用普通BPR，后续应用式11。

$$
\mathcal{L}_{\mathrm{boost}}=
\begin{cases}
-\displaystyle\sum_{(u,i,j)\in\mathcal{O}}
\log\left(\sigma\left(\Delta S_{k}^{\ast}\right)\right), & k=1, \\[6pt]
\mathcal{L}_{\mathrm{boost}}^{\prime}, & k>1.
\end{cases}
\tag{12}
$$

用“旧整体模型＋本轮补充模型”作为老师，训练一个仍然由三个分支组成的学生系统。学生由旧整体模型的参数初始化，然后接受蒸馏训练。

学生拟合两位老师的分数差之和。

$$
\mathcal{L}_{\mathrm{distill}}=\sum_{(u,i,j)\in\mathcal{O}}
\left(\Delta S_k-\mathrm{sg}\left(\Delta S_{k-1}+\Delta S_k^{\ast}\right)\right)^2.\tag{13}
$$

老师的预测也可能不够好，加入全局校正。

$$
\mathcal{L}_{\mathrm{gc}}=-\sum_{(u,i,j)\in\mathcal{O}}
\log\left(\sigma(\Delta S_k)\right).\tag{14}
$$

蒸馏阶段的总目标

$$
\mathcal{L}_d=\mathcal{L}_{\mathrm{gc}}+\mathcal{L}_{\mathrm{distill}}.\tag{15}
$$


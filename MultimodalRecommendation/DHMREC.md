# DHMRec: Collaboration-Guided Multimodal Disentanglement and Hierarchical Fusion for Recommendation

- 发表在：*Proceedings of the AAAI Conference on Artificial Intelligence* 2026-03-14 [论文链接](https://doi.org/10.1609/aaai.v40i19.38661) [项目代码：只包含pdf，目前没有代码](https://github.com/cheer-io/DHMRec)

- 作者 Xiaohan Zhan, Yuliang Shi, Jihu Wang, Shijun Liu, Fanyu Kong和Zhiyong Chen. **山东大学软件学院**、**山大地纬软件  股份有限公司**和**香山东大学控制科学与工程学院**


# 问题分析
主要在做特征结偶和层次融合
商品的图片和文字可能有三类信息

- 两边都能体现的信息：颜色、款式。
- 各自模态正擅长表达的信息：图像中衣服的纹理、图案；文字中的材质类型。
- 对推荐没有帮助的信息：模特的发型。

有用的信息和噪声混在一起，不同的信息对推荐的作用不同，简单的平均融合不合适。


# 方法
## 模态解耦模块
商品的文本和图像特征已由sentence-transformer及CNN预先提取，个经过一次线性变换得到纬度相同的$Z_m$。

设置两个编码器：公共编码器 $E_{C_m}$，私有编码器 $E_{P_m}$ 将每种模态分成两条支路。两种编码器采用线性网络。

$$
C_{i,m}=E_{C_m}(Z_m),
P_{i,m}=E_{P_m}(Z_m)\qquad \text{(2)}
$$

经过编码器之后对于一份图像特征和一份文字特征被转换成了四路表示：
|  模态 |  共有支路  |  私有支路  |
| -  |  -  |  -  |
|  图像 |  $C_{i,v}$  | $P_{i,v}$   |
| 文字  |  $C_{i,t}$  | $P_{i,t}$   |

### 协作信号引导去噪
由于噪声往往出现在私有特征中，于是将私有私有表示 $P_{i,m}$输入到门控网络，用包含丰富协作信息的ID嵌入去调节推荐任务相关的模态特定特征表示。

$$
S_{i,m}=P_{i,m}\odot\sigma(W_2^mE_{i,id}+b_2^m)\qquad \text{(3)}
$$

### 模态公共表示对齐

采用JSD损失对齐图片共有表示 $C_{i,v}$ 和文字共有表示 $C_{i,t}$ 让他们彼此一致。

$$
\mathcal L_{\mathrm{JSD}}=\frac12\left[KL(P\|M)+KL(Q\|M)\right]\qquad \text{(4)}
$$


$$
P=\sigma(C_{i,t}),\qquad Q=\sigma(C_{i,v}),\qquad M=\frac{P+Q}{2}
$$

把图片共有表示和文字共有表示合起来。

$$
C_i=\sigma^{-1}\left(\frac{\sigma(C_{i,t})+\sigma(C_{i,v})}{2}\right)\qquad \text{(5)}
$$

### 解耦约束

使用正交损失让共有表示与私有表示保持正交，减少重复的表示，增加差异化。

$$
\mathcal L_{\mathrm{orth}_m}=\frac{|\langle C_{i,m},S_{i,m}\rangle|}{\|C_{i,m}\|_2\|S_{i,m}\|_2}\qquad \text{(6)}
$$

该模块最终得到三个表示：
$C_i$:融合后的共有表示。
$S_{i,v}$:图片特有表示。
$S_{i,t}$:文字特有表示。

## 多视图图学习模块

### 用户-物品图
创建四张用户-物品交互图，将ID表示、文字特有表示、图片特有表示、公共表示分别在四张图上进行LightGCN传播。

$$
E_{\hat m}^{(\ell)}=\left(D^{-\frac12}AD^{-\frac12}\right)E_{\hat m}^{(\ell-1)}\qquad \text{(7)}
$$

$$
E_{\hat m}=\frac{1}{L+1}\sum_{\ell=0}^{L}E_{\hat m}^{(\ell)}\qquad \text{(8)}
$$

### 物品-物品图
使用KNN算法构建三张物品-物品图，利用内容相似的商品补充信息，增强物品表示。

$$
I_{i,i'}^m=\frac{(x_i^m)^\top x_{i'}^m}{\|x_i^m\|\|x_{i'}^m\|},\qquad m\in\{t,v\}\qquad \text{(9)}
$$


使用top-k保留最相关的k个邻居。

$$
\hat I_{i,i'}^m=
\begin{cases}
1, & i'\text{ 属于 }i\text{ 最相似的 }k\text{ 个邻居} \\
0, & \text{其他情况}
\end{cases}
\qquad \text{(10)}
$$

归一化

$$
\tilde I^m=D^{-\frac12}\hat I^mD^{-\frac12}\qquad \text{(11)}
$$

组合图片图和文字图得到共有图。

$$
\tilde I^c=\beta_v\tilde I^v+\beta_t\tilde I^t\qquad \text{(12)}
$$

三张图准备好之后，分别传播三路表示

$$
\hat E_{i,\tilde m}^{(\ell)}=\tilde I^{\tilde m}\hat E_{i,\tilde m}^{(\ell-1)},\qquad \tilde m\in\{t,v,c\}\qquad \text{(13)}
$$

用更新后的物品向量重新计算用户表示。

$$
\hat e_{u,\tilde m}=\sum_{i\in\mathcal N(u)}\frac{1}{\sqrt{|\mathcal N(u)|}\sqrt{|\mathcal N(i)|}}\hat e_{i,\tilde m}\qquad \text{(14)}
$$

该模块最终得到四个用户-商品向量矩阵：

$$
\hat E_c,\qquad \hat E_t,\qquad \hat E_v,\qquad\hat E_{id}
$$


## 多模态层次融合模块
该模块主要做两件事情
1、先融合共有、图片特有、文字特有三路内容信息。

2、再把融合后的内容表示与 ID 表示结合起来。
### 解耦模态特征融合
先将三路表示拼接

$$
\hat E=[\hat E_c;\hat E_t;\hat E_v]
$$

经过线性变换压回d维

$$
\hat E_{\mathrm{fus}}=W_3\hat E
$$

为每一路准备一个Q,K,V矩阵，以图片为例：

$$
Q_v=W_q^v\hat E_{\mathrm{fus}}
$$

$$
K_v=W_k^v\hat E_v,\qquad V_v=W_v^v\hat E_v
$$

计算正分支

$$
O_{\tilde m}^{\mathrm{pos}}=\mathrm{Softmax}\left(\frac{Q_{\tilde m}K_{\tilde m}^{\top}}{\sqrt N}\right)V_{\tilde m}\qquad \text{(15)}
$$

计算负分支

$$
O_{\tilde m}^{\mathrm{neg}}=\left(1-\mathrm{Softmax}\left(\frac{Q_{\tilde m}K_{\tilde m}^{\top}}{\sqrt{N}}\right)\right)V_{\tilde m}\qquad \text{(16)}
$$

合并正负分支

$$
O_{\tilde m}=\alpha_{\mathrm{pos}}O_{\tilde m}^{\mathrm{pos}}+\alpha_{\mathrm{neg}}O_{\tilde m}^{\mathrm{neg}}\qquad \text{(17)}
$$

做线性变换、层归一化，并加回原来的表示

$$
\tilde E_{\tilde m}=\mathrm{LN}\left(\mathrm{LN}\left(W_o^{\tilde m}O_{\tilde m}\right)+\hat E_{\tilde m}\right)\qquad \text{(18)}
$$

把三路处理结果相加

$$
\tilde E_{\mathrm{mm}}=\tilde E_c+\tilde E_v+\tilde E_t\qquad \text{(19)}
$$

### 融合ID表示

为了确保多模态融合与协同信号之间的一致性，通过计算用户和物品嵌入的内积来获得交互分布,从而对齐多模态和ID嵌入的偏好分布

$$
P_{\mathrm{mm}}=\mathrm{Softmax}\left(\tilde E_{u,\mathrm{mm}}^\top\tilde E_{i,\mathrm{mm}}\right)\qquad \text{(20)}
$$

使用KL散度对齐两个分布$P_{id}$由 $E_{id}$ 推导而来

$$
\mathcal L_{\mathrm{KL}}=KL(P_{\mathrm{mm}}\|P_{\mathrm{id}})\qquad \text{(21)}
$$

最终的用户和物品嵌入由多模态融合嵌入和ID嵌入加权得到

$$
\dot E=E_{\mathrm{id}}+\alpha_{\mathrm{mm}}\tilde E_{\mathrm{mm}}\qquad \text{(22)}
$$


## 预测& 优化
通过计算用户和物品最终嵌入的内积来获得用户的预测分数。

$$
\hat y_{ui}=(\dot e_u)^\top\dot e_i
$$

采用贝叶斯个性化排序(BPR)作为推荐优化损失。

$$
\mathcal L_{\mathrm{BPR}}=\sum_{(u,i,j)\in D_{\mathrm{train}}}-\ln\sigma(\hat y_{ui}-\hat y_{uj})
$$

总体目标整合了BPR损失、解耦损失、KL对齐损失和 L2 正 则化。

$$
\begin{aligned}
\mathcal L
={}&\mathcal L_{\mathrm{BPR}}\\
&+\gamma_{\mathrm{dis}}
\left(
\mathcal L_{\mathrm{JSD}}
+\mathcal L_{\mathrm{orth}_v}
+\mathcal L_{\mathrm{orth}_t}
\right)\\
&+\gamma_{\mathrm{kl}}\mathcal L_{\mathrm{KL}}\\
&+\gamma_{\mathrm{reg}}\|\theta\|_2^2
\end{aligned}
$$

## 结果
在三个亚马逊数据集Baby、Sports 和 Clothing上与最佳基线相比, DHMRec在三个数据集的R@20上分别取得了2.63%、 1.09%和2.44%的性能提升。
## 存在的不足
去掉解耦模块后baby数据集的NDCG@20反而上升了，
BPR损失希望相关商品尽量排在前面，解耦模块希望共有表示彼此对齐，共有表示与特有表示尽量正交，是否发生了冲突？
如果一个商品的某一属性能够帮助判断用户偏好，但它同时存在于共有表示和特有表示中，那么正交约束会尽量减少这种重合。


## 想法尝试
第一个解耦模块可以替换成MPD的做法。将两个模态的私有表示分为MP、AO、UP，之后合并两个模态的MP当作共有表示，将两个模态的AO、UP合并为私有表示交给第二个模块。

预提取的模态特征 $X_m$。

$$
x_{i,m}=X_m[:,i]\in\mathbb R^{D_m},
$$

对特征进行逐维标准化

$$
z_{i,m,k}=\frac{x_{i,m,k}-\mu_{m,k}}{\sigma_{m,k}+\epsilon}.
$$

得到：

$$
z_{i,v}\in\mathbb R^{D_v},\qquad z_{i,t}\in\mathbb R^{D_t}.
$$

对每个模态设置两种门，输入门 $s_m$ 和输出门 $h_m$对应STG输入门与HCG输出门。
每个模态都有两组可训练参数：

$$
\alpha_m,\beta_m\in\mathbb R^{D_m}.
$$

通过 Sigmoid 得到：

$$
s_m=\mathrm{Sigmoid}(\alpha_m),
$$

$$
h_m=\mathrm{Sigmoid}(\beta_m).
$$

$s_v$	图像 $\rightarrow$ 文本 输入门

$h_t$	图像 $\rightarrow$ 文本 输出门

$s_t$	文本 $\rightarrow$ 图像 输入门

$h_v$	文本 $\rightarrow$ 图像 输出门

拿图像举例

$$
\widetilde z_{i,v}=s_v\odot z_{i,v}
$$

$$
\boxed{\widehat z_{i,t}=f_{v\rightarrow t}(\widetilde z_{i,v})}
$$

反向进行同样的操作，预测网络可以使用MLP。

双向预测误差

$v\rightarrow t$

$$
\boxed{\mathcal L_{v\rightarrow t}=\frac1{D_t}\sum_j h_{t,j}e_{v\rightarrow t,j}+\lambda_s\frac1{D_v}\sum_k s_{v,k}-\lambda_h\frac1{D_t}\sum_j h_{t,j}}
$$

$t\rightarrow v$

$$
\boxed{\mathcal L_{t\rightarrow v}=\frac1{D_v}\sum_k h_{v,k}e_{t\rightarrow v,k}+\lambda_s\frac1{D_t}\sum_j s_{t,j}-\lambda_h\frac1{D_v}\sum_k h_{v,k}}
$$

总损失

$$
\boxed{\mathcal L_{\mathrm{dec}}=\mathcal L_{v\rightarrow t}+\mathcal L_{t\rightarrow v}}
$$

门控训练

1、将输出门全部固定设为1，更新输入门和预测器。

2、固定输入门与预测器，更新输出门。

3、固定第二步更新后带有阈值的输出门，更新输入门与预测器。

$$
\bar s_m=\mathbf1[s_m>\tau_s],
$$

$$
\bar h_m=\mathbf1[h_m>\tau_h].
$$

结合门控将每个模态划分MP、AO、UP。

以图像为例

$$
\boxed{M_{i,v}=\bar h_v\odot z_{i,v}}
$$

$$
\boxed{A_{i,v}=(1-\bar h_v)\odot\bar s_v\odot z_{i,v}}
$$

$$
\boxed{U_{i,v}=(1-\bar h_v)\odot(1-\bar s_v)\odot z_{i,v}}
$$

$M_{i,v}$：自身被选为可预测目标的图像维度；

$A_{i,v}$：自身未被选为可预测目标，但被选作预测输入的维度；

$U_{i,v}$：两种选择都没有选中的维度。

$$
M_{i,v}+A_{i,v}+U_{i,v}=z_{i,v}.
$$

文本特征处理逻辑相同。

调整合并特征得到共有表示、特有表示

$$
r_{i,m}^{M}=H_m^{M}M_{i,m},
$$

$$
r_{i,m}^{A}=H_m^{A}A_{i,m},
$$

$$
r_{i,m}^{U}=H_m^{U}U_{i,m}.
$$

合并MP得到 $c_i^{\mathrm{new}}$

$$
\boxed{c_i^{\mathrm{new}}=W_M[r_{i,v}^{M};r_{i,t}^{M}]+b_M}
$$

对AO、UP进行ID引导门控

$$
g_{A,i}^m=\mathrm{Sigmoid}\left(W_{g,A}^m e_{i,\mathrm{id}}+b_{g,A}^m\right),
$$

$$
g_{U,i}^m=\mathrm{Sigmoid}\left(W_{g,U}^m e_{i,\mathrm{id}}+b_{g,U}^m\right).
$$

$$
\widetilde r_{i,m}^{A}=r_{i,m}^{A}\odot g_{A,i}^m,
$$

$$
\widetilde r_{i,m}^{U}=r_{i,m}^{U}\odot g_{U,i}^m.
$$

将 AO、UP 融合为每个模态的输出表示

图像：

$$
\boxed{s_{i,v}^{\mathrm{new}}=W_F^v[\widetilde r_{i,v}^{A};\widetilde r_{i,v}^{U}]+b_F^v}
$$

文本：

$$
\boxed{s_{i,t}^{\mathrm{new}}=W_F^t[\widetilde r_{i,t}^{A};\widetilde r_{i,t}^{U}]+b_F^t}
$$

最终得到：

$$
\boxed{c_i^{\mathrm{new}}\in\mathbb R^d,\quad s_{i,v}^{\mathrm{new}}\in\mathbb R^d,\quad s_{i,t}^{\mathrm{new}}\in\mathbb R^d}
$$
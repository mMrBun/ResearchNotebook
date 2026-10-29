# Sign-Aware Multimodal Graph Recommendation

- 发表在：Proceedings of the AAAI Conference on Artificial Intelligence 2026年3月14日 [论文链接](https://doi.org/10.1609/aaai.v40i18.38547)

- 作者：Yahong Lian、Haotian Tian、Chunyao Song、Tingjian Ge。前三位来自南开大学，Tingjian Ge 来自马萨诸塞大学洛厄尔分校。

- [作者代码](https://github.com/DRec4AI/SiMGR2025)；[本地复现记录](../CodeReproduction/SiMGR/README.md)。

# Question

结论是用户买过一个产品不代表他喜欢这个产品。

比如用户买了一双跑鞋 A，评价 5 星，买了一个头盔 B，评价 1 星。虽然用户与头盔发生了交互，但不能直接把这条记录当作喜欢的证据。差评仍然有用，它可以帮助模型学习用户不满意什么。

如果系统只记录“买过还是没买过”，跑鞋 A 和头盔 B 就都会被记成“买过”。这种做法没有充分考虑高评分交互与低评分交互之间的语义差异，高评分表示用户偏好，而低评分明确表达不满意。忽视负反馈的含义，可能限制推荐性能。

本文做的是多模态 Top-K 商品推荐：根据用户的历史交互、评分和商品图文信息，对候选商品打分排序。

这些是作者提出本项工作的出发点，并且做了对比实验。
![figure 1](../assets/SiMGR/%20figure1.png)

绿色柱子：将所有用户评价1-5星都当作正向交互

蓝色柱子：只保留4-5星当作正向交互

紫色柱子：LSGRec(Liu 等 2025a)纳入负反馈后的得分

图 1 的纵轴是 NDCG@20，越高表示前 20 个推荐结果的排序质量越好。比较同一个模型的蓝色和绿色柱子，可以看出把低评分混进正反馈可能损害效果；LSGRec 的紫色柱子则说明，保留并区分负反馈可以比直接删除更有效。

通过对比实验，作者认为区分正负反馈有帮助，使用商品的图文信息也有帮助，将两种方法结合起来时会面临下面三种挑战。

### 商品的图文信息与符号交互结构怎么融合

图片和文本描述的是商品内容，正负交互表达的是用户对商品的满意度，简单的拼接或相加不一定有效，反而可能降低推荐性能。

### 统一评分边界不适用所有用户

有些用户习惯打高分，三星对他们而言可能就是差评；而对一些评分严格的用户来说，三星已经算相对满意。因此需要根据用户的评分习惯，分别确定正负反馈的划分界限。

### 把交互图拆分成正负子图之后会加剧稀疏性

进一步拆分后，部分节点在某个子图中可能只剩很少的邻居，甚至没有邻居。作者考虑补充一些推测出来的连接，改善信息传播条件。这里减少的是每个子图单独可用的边，并不是原始交互记录被删除了。

# Idea

针对这些挑战，作者提出了三个模块：SGA 负责利用正负反馈，PST 负责确定正负反馈的划分界限，APFI 负责补充稀疏的反馈关系。

### SGA（Signed Graph Augmentation）

先固定评分阈值 $`\delta = 3`$

4、5 分进入正反馈图 $`\mathcal{G}^+`$

1、2、3 分进入负反馈图 $`\mathcal{G}^-`$

#### 在正图上进行LightGCN传播

```math
\dot{E}^{+(l+1)}
=
(D^+)^{-\frac{1}{2}}
A^+
(D^+)^{-\frac{1}{2}}
\dot{E}^{+(l)}
\qquad \text{(5)}
```

传播完成后对各层取平均

```math
\ddot{E}^+
=
\frac{1}{L+1}
\sum_{l=0}^{L}\dot{E}^{+(l)}
\qquad \text{(6)}
```

$`A^+`$ 是正图的邻接矩阵，$`D^+`$ 是正图的度矩阵。初始表示来自可学习的用户、商品 ID 向量；传播两层时，把第 0、1、2 层一起平均，得到 $`\ddot{E}^+`$。公式（7）、（8）是把矩阵计算展开到单个用户和商品。

将公式5拆分成用户和商品的表示如下：用户从与之相连的商品节点收集信息，商品也以同样方式从相连用户那里收集信息

```math
\dot{\mathbf{e}}_u^{+(l+1)}
=
\sum_{i\in\mathcal{N}_u^+}
\frac{1}{\Gamma_{ui}}
\dot{\mathbf{e}}_i^{+(l)}
```

```math
\dot{\mathbf{e}}_i^{+(l+1)}
=
\sum_{u\in\mathcal{N}_i^+}
\frac{1}{\Gamma_{ui}}
\dot{\mathbf{e}}_u^{+(l)}
```

```math
\Gamma_{ui}
=
\sqrt{|\mathcal{N}_u^+|\,|\mathcal{N}_i^+|}
```

对称度归一化，用来避免用户和商品连接的边数差距过大。

#### 将正反馈表示沿负图传播，再进行变换。

使用刚得到的 $`\ddot{E}^+`$ 作为输入，节点先获得正反馈信息，再通过负反馈关系收集其他节点的这些表示。

比如用户1喜欢A不喜欢C
负反馈图在传播时用户1可以拿到自己不喜欢的C的正反馈信息。作用是通过MLP将这个信息变为有助于推荐的特征。作者把这种负反馈关系称作异配性。

> 同配性：连接的节点倾向于拥有相似的特征；异配性相反。

通过符号余弦损失约束MLP做非线性特征转换。

```math
\mathcal{L}_{\mathrm{sign}}
=
\begin{cases}
1-\cos(\mathbf{e}_u^*,\mathbf{e}_i^*),
& \text{if } (u,i)\in\mathcal{E}^+\\
\gamma\cdot\max\left(0,\cos(\mathbf{e}_u^*,\mathbf{e}_i^*)-\psi\right),
& \text{if } (u,i)\in\mathcal{E}^-
\end{cases}
\qquad \text{(4)}
```

```math
\ddot{E}^-
=
\mathrm{MLP}\left(
(D^-)^{-\frac{1}{2}}
A^-
(D^-)^{-\frac{1}{2}}
\ddot{E}^+
\right)
\qquad \text{(9)}
```

```math
\text{正反馈表示}
\longrightarrow
\text{负图传播}
\longrightarrow
\text{MLP}
\longrightarrow
\text{负反馈表示}
```

#### 让正负反馈增强原有交互表示
 $`E_{\mathrm{ig}}^{\mathrm{id}}`$ 是原始交互图，保留了所有交互图分支。通过上面得到的 $\ddot{E}^+$ 和 $\ddot{E}^-$ 来增强原有的表示。

```math
E_{\mathrm{ig}}^{\mathrm{id}}(\mathrm{new})
=
E_{\mathrm{ig}}^{\mathrm{id}}
+
\mathrm{ReLU}(w^+ + w^-)
\qquad \text{(10)}
```

```math
w^+=\mathbf{a}[0]\odot\ddot{E}^+,\qquad
w^-=\mathbf{a}[1]\odot\ddot{E}^-,\qquad
\mathbf{a}=\mathcal{F}_s([\mathbf{a}^+,\mathbf{a}^-])
```

```math
\mathbf{a}^+
=
\frac{\ddot{E}^+\ddot{W}^+}
{\ddot{E}^+\ddot{W}^+ + \ddot{E}^-\ddot{W}^-},
\qquad
\mathbf{a}^-
=
\frac{\ddot{E}^-\ddot{W}^-}
{\ddot{E}^+\ddot{W}^+ + \ddot{E}^-\ddot{W}^-}
\qquad \text{(11)}
```

得到增强后的交互表示后，再与图文分支的表示融合：

```math
E^*=E_{\mathrm{ig}}^{\mathrm{id}}(\mathrm{new})+\widetilde{E}
```

### PST（Personalized Sign Threshold）

PST 根据每个用户已有的评分计算阈值，用来替代所有用户共用的 $`\delta=3`$。

用bool掩码矩阵 $`M_{ui}=\mathbb{I}(R_{ui}>0)`$ 标记哪些位置有评分，再计算该用户的评分均值和标准差：

```math
\mu_u=\frac{\sum_iR_{ui}M_{ui}}{\sum_iM_{ui}},\qquad
\varepsilon_u=\sqrt{\frac{\sum_i(R_{ui}-\mu_u)^2M_{ui}}{\sum_iM_{ui}}}
\qquad \text{(13)}
```

没有交互的位置在矩阵里记为 0。

用户的最终阈值是：

```math
\delta_u=\nu\mu_u-(1-\nu)\varepsilon_u
\qquad \text{(14)}
```

$`\nu`$ 是控制均值和标准差影响的超参数。
这样对于同样的三分评价，对于习惯打高分的人可能是差评，对于习惯打低分的人可能是好评。

### APFI（Adaptive Pseudo Feedback Injection）

APFI 利用商品的图文相似度，为正负图补充伪反馈边，缓解拆图后的稀疏性。

先按商品的交互次数从少到多排序，按比例 $`\xi=0.1`$ 划为长尾商品，头部商品：

```math
N_{\mathrm{tail}}=\lfloor\xi N\rfloor
```

随后，分别计算商品图片向量和文本向量之间的余弦相似度，再逐元素相乘：

```math
s_{ij}^m=\frac{(h_i^m)^\top h_j^m}{\|h_i^m\|\|h_j^m\|},\quad m\in\{v,t\}
```
```math
\qquad S^f=S^v\odot S^t\qquad \text{(16)}
```
如果图片很相似但文本不相似，最终的相似度分数也会很低。
最终得到一个商品-商品的相似度分数矩阵$S^f $。
> 如果图片和文本的相似度都是负数，相乘变为正数了岂不是变成噪声了？


对于每个用户，取出其历史交互商品对应的相似度行，与未交互商品比较。

```math
S_u^f=\{S_{[q,:]},\,q\in\mathcal{N}_u\},\qquad
S_u^f\in\mathbb{R}^{|\mathcal{N}_u|\times N}
\qquad \text{(17)}
```

利用上面构造的长尾掩码，分别得到：

```math
TaC_u=S_u^f\cdot b_i^{\mathrm{tail}}
```

```math
PoC_u=S_u^f\cdot\neg b_i^{\mathrm{tail}}
```

展平操作，方便 Top-Q 排序。

```math
TaC_u=\ {vec}(TaC_u)
```

```math
PoC_u=\ {vec}(PoC_u)
```

```math
PoPI(u)_w,\qquad
PoPI(u)=\ {top}_Q(TaC_u)
\qquad \text{(18)}
```

$PoPI(u)_w$ 为对应相似度分数，$`Q=20`$

伪正边的添加判断

```math
\widetilde{R}^+_{ui}
=
\begin{cases}
1, & i\in PoPI(u)\ \land\ PoPI(u)_w>\zeta^+\\
0, & \text{otherwise}
\end{cases}
\qquad \text{(19)}
```

与伪正反馈相反，这里选择最不相似的头部商品。

```math
NePI(u)_w,\qquad
NePI(u)=\ {top}_Q(-PoC_u)
\qquad \text{(20)}
```

```math
\widetilde{R}^-_{ui}
=
\begin{cases}
1, & i\in NePI(u)\ \land\ NePI(u)_w<\zeta^-\\
0, & \text{otherwise}
\end{cases}
\qquad \text{(21)}
```

选出的伪边分别加到原来的正负交互矩阵：

```math
R_b^+\leftarrow R_b^++\widetilde{R}^+,\qquad
R_b^-\leftarrow R_b^-+\widetilde{R}^-
```

### Optimization

作者担心融合后的最终表示与增强交互表示对同一个用户或商品所表达的特征不够一致，于是引入对比学习，让他们保持较好的对其关系。

用户侧的InfoNCE损失
```math
\mathcal{L}_{cl}^{u}
=
\sum_{u\in\mathcal{U}}
-\log
\frac{
\exp(\mathbf{e}_u^{id}\cdot\mathbf{e}_u^*/\tau)
}{
\sum_{z\in\mathcal{U}}
\exp(\mathbf{e}_z^{id}\cdot\mathbf{e}_z^*/\tau)
}
\qquad \text{(22)}
```
商品侧使用同样的对比学习损失。


总损失：

```math
\mathcal{L}
=
\mathcal{L}_{BPR}
+
\lambda_s\mathcal{L}_{sign}
+
\beta_{cl}(\mathcal{L}_{cl}^{u}+\mathcal{L}_{cl}^{i})
+
\lambda_r\|\Theta\|
\qquad \text{(23)}
```
 $\mathcal{L}_{BPR}$：帮助用户对不同商品的推荐分数排序。希望交互过商品的分数大于为交互过的商品分数。

 $\mathcal{L}_{cl}$ 让不同来源的表示保持对齐。

 
 $\lambda_r\|\Theta\|$ 正则化项，约束模型参数，降低过拟合风险。


# Experiment

### 数据集和实验设置

作者使用 Amazon 的 Baby、Sports、Electronics 三个数据集。

| 数据集      | 用户数 | 商品数 | 交互数  | 密度   |
| ----------- | ------:| ------:| -------:| ------:|
| Baby        | 19,445 | 7,050  | 160,792 | 0.117% |
| Sports      | 35,598 | 18,357 | 296,337 | 0.045% |
| Electronics | 20,247 | 11,589 | 347,393 | 0.148% |

模型基于 MMRec 框架实现，向量维度 64，batch size 为 2048，学习率 0.001，用户—物品图传播 2 层，使用 Xavier 初始化和 Adam 优化。验证集 Recall@20 连续 20 轮没有提升就提前停止，论文实验设备为 NVIDIA RTX 3090。

### 消融实验

图 3 比较 Base、Base+SGA、去掉 APFI、去掉 PST，以及完整 SiMGR。加入 SGA 后指标整体提高，完整模型表现最好。

# Result

### 论文结果

| 方法     | Baby R@20  | Baby N@20  | Sports R@20 | Sports N@20 | Electronics R@20 | Electronics N@20 |
| -------- | ----------:| ----------:| -----------:| -----------:| ----------------:| ----------------:|
| MF-BPR   | 0.0557     | 0.0240     | 0.0634      | 0.0288      | 0.0662           | 0.0306           |
| LightGCN | 0.0750     | 0.0322     | 0.0846      | 0.0386      | 0.0791           | 0.0374           |
| SiReN    | 0.0678     | 0.0296     | 0.0750      | 0.0334      | 0.0548           | 0.0326           |
| SiGRec   | 0.0770     | 0.0329     | 0.0781      | 0.0350      | 0.0538           | 0.0308           |
| LSGRec   | 0.0715     | 0.0323     | 0.0849      | 0.0388      | 0.0536           | 0.0315           |
| MGCN     | 0.0984     | 0.0436     | 0.1114      | 0.0504      | 0.0879           | 0.0421           |
| BM3      | 0.0845     | 0.0369     | 0.0968      | 0.0441      | 0.0792           | 0.0376           |
| FREEDOM  | 0.1001     | 0.0438     | 0.1096      | 0.0490      | 0.0881           | 0.0409           |
| LGMRec   | 0.0967     | 0.0441     | 0.1064      | 0.0484      | 0.0866           | 0.0416           |
| DiffMM   | 0.0796     | 0.0349     | 0.1014      | 0.0454      | 0.0871           | 0.0394           |
| TMLP     | 0.0973     | 0.0438     | 0.1111      | 0.0500      | 0.0854           | 0.0403           |
| SiMGR    | **0.1052** | **0.0471** | **0.1160**  | **0.0526**  | **0.0904**       | **0.0428**       |

### 本地复现结果

本地已经跑通 Sports，以下为运行日志，时间： **2026-10-21 16:07:09**，seed 为 **999**。记录 BEST 部分。

| 数据划分    | R@10   | R@20   | N@10   | N@20   |
| ----------- | ------:| ------:| ------:| ------:|
| Valid       | 0.0661 | 0.0982 | 0.0364 | 0.0445 |
| Test        | 0.0671 | 0.1001 | 0.0377 | 0.0462 |
| 论文 Sports | 0.0782 | 0.1160 | 0.0428 | 0.0526 |

| K   | Recall@K | NDCG@K | Precision@K | MAP@K  |
| --- | --------:| ------:| -----------:| ------:|
| 5   | 0.0445   | 0.0303 | 0.0098      | 0.0249 |
| 10  | 0.0671   | 0.0377 | 0.0074      | 0.0279 |
| 20  | 0.1001   | 0.0462 | 0.0055      | 0.0301 |
| 50  | 0.1590   | 0.0581 | 0.0035      | 0.0320 |

本地跑了3次，记录的是最高值，但还是没有达到论文中的分数，现在对原理理解还不够，想不到怎么优化。

# Limitation

1. 该工作依赖用户的真实评价，现在电商系统存在泛滥且普遍的刷好评操作，另外相当一部分用户群体只有在不喜欢商品的时候才会写评价而且是差评，如果觉得商品没啥问题甚至连确认收货都忘记点，系统默认好评。所以有价值的真实评价怎么界定？

2. **伪反馈可能引入错误。** 我有点无法理解构造伪边的操作，用户对头部曝光产品长期没有交互就构造伪负边我认为不太合理，没有交互不代表可能不喜欢，现在没交互，未来可能有交互，只是现在不需要。对于相似的同类物品没有交互作者认为是缺少曝光，这种考虑有点片面，拿羽绒服举例，单纯的因为款式过时了这种情况就会造成没有交互。

这篇文章给我的主要启发是，交互记录不能一律当成喜欢，低评分也可以成为有用的信息。如果要在这篇工作的基础上继续优化，我觉得要考虑如何正确使用这些评价的正负反馈，而不是单纯的增加图的复杂度。

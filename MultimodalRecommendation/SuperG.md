# Joint Similar User Exploration and Informative Behavior Guidance for Multi-Modal New Item Recommendation

- 发表在：ACM Web Conference 2026（WWW ’26）会议论文集 [论文链接](https://doi.org/10.1145/3774904.3792302) 

- 作者 Jianye Xie、Lianyong Qi、Weiming Liu、Anqi Wang、Xiaolong Xu、Haolong Xiang、Xuyun Zhang、Wenwen Gong、Yang Zhang、Amin Beheshti、Wanchun Dou **中国石油大学（华东）** **山东省智能油气工业软件重点实验室** **字节跳动** **南京信息工程大学** **麦考瑞大学** **中国农业大学** **北德克萨斯大学** **南京大学**

1. 问题分析

新物品缺少交互记录，现有方法通常根据共同交互物品寻找相似用户，但接触过同一热门物品的用户，整体兴趣未必相近；兴趣相近的用户也可能没有共同交互物品。



### 2. Similar User Exploration：相似用户探索

#### 2.1 把物品的多模态内容变成向量

| 模态          | 使用模型                                | 表示                                    |
| ------------- | --------------------------------------- | --------------------------------------- |
| 文本 Text     | RoBERTa-base：最终层的 `[CLS]` 隐藏状态 | $\mathbf x_j^{(txt)}\in\mathbb R^{768}$ |
| 图像 Visual   | BLIP 图像编码器的全局表示               | $\mathbf x_j^{(vis)}\in\mathbb R^{768}$ |
| 音频 Acoustic | BEATs 最后一层帧级特征做均值池化        | $\mathbf x_j^{(aud)}\in\mathbb R^{768}$ |

有三种模态时，商品统一向量为：

$$
\mathbf z_j=\frac{\mathbf x_j^{(vis)}+\mathbf x_j^{(aud)}+\mathbf x_j^{(txt)}}{3}\in\mathbb R^{768}.
$$

- $j$：商品编号；$\mathbf z_j$：商品最终向量。
- $\mathbf x_j^{(vis)},\mathbf x_j^{(aud)},\mathbf x_j^{(txt)}$：该商品的视觉、音频、文本向量。
- $3$：三种可用模态的数量；**只有图文时除以 $2$**。Baby、Clothes 使用图文；TikTok 使用图文音频。

#### 2.2 ：由历史物品序列形成用户表示

$$
\boxed{\mathbf e_u=\mathrm{MEAN}\!\left(\mathrm{Trm}(\widetilde{\mathbf X}_u)\right)\in\mathbb R^d}\tag{1}
$$



$$
\mathbf X_u=[\mathbf z_1;\mathbf z_2;\ldots;\mathbf z_N]\in\mathbb R^{N\times d},
\qquad
\widetilde{\mathbf X}_u=\mathbf X_u+\mathbf P.
$$

从内容和先后循序中概括用户兴趣，之后用该向量检索相似用户。





#### 2.3 对比学习

公式（1）生成的向量不一定天然具备很好的用户区分能力。作者希望同一用户历史被轻微扰动后，仍具有稳定的兴趣表示；不同用户更容易区分。

论文为历史 $\mathcal H_u$ 设计了四种增强方式：

1. **Attention Masking**：随机挑选一部分历史物品，在 Transformer 注意力计算中屏蔽。
2. **Subsequence Sampling**：随机截取一段连续历史。
3. **Subsequence Reordering**：随机选取一个连续子序列，打乱其内部顺序。
4. **Segment Shuffling**：把历史拆成多个连续段，打乱段之间的顺序，但保持每段内部顺序。



#### 2.4 ：双方向用户对比损失

$$
\begin{aligned}
L_{CL}=-\Bigg[&\log\frac{
\exp\!\left(\cos(\mathbf e'_u,\mathbf e''_u)/\tau_1\right)}{
\displaystyle\sum_{u^-\in\mathcal U_{neg}}
\exp\!\left(\cos(\mathbf e'_u,\mathbf e''_{u^-})/\tau_1\right)}\\
+&\log\frac{
\exp\!\left(\cos(\mathbf e'_u,\mathbf e''_u)/\tau_1\right)}{
\displaystyle\sum_{u^-\in\mathcal U_{neg}}
\exp\!\left(\cos(\mathbf e'_{u^-},\mathbf e''_u)/\tau_1\right)}\Bigg].
\end{aligned}\tag{2}
$$

**输入**：同一用户的两种增强视图及其他用户的负样本视图。**输出**：一个用于训练的标量 $L_{CL}$。

### 3. Informative Behavior Mining：信息性行为挖掘



#### 3.1 ：从目标用户与相似用户的历史构造候选集

$$
\boxed{\mathcal B^{retrieved}=\{b_{l,r}\mid l\in\{1,\ldots,m+1\},\ r\in\{1,\ldots,k\}\}}\tag{3}
$$

**输入**：目标用户的历史 + $m$ 个相似用户的历史。**输出**：最多 $(m+1)k$ 条候选行为。

#### 3.2 ：基础语义检索分数

$$
\boxed{S_{b_{u,j},b_{l,r}}^{retriever}
=\cos(\mathbf e_u,\mathbf e_l)+\cos(\mathbf z_j,\mathbf z_r)}\tag{4}
$$

**输入**：两个用户向量 + 两个商品向量。**输出**：一个相关性分数。

#### 3.3 ：目标用户的个性化偏好分数

$$
\boxed{S_{u,r}^{retriever}=\cos(\mathrm{MLP}(\mathbf e_u),\mathbf z_r)}\tag{5}
$$

**输入**：目标用户向量、候选历史商品向量。**输出**：一个标量个性化分数。

#### 3.4 ：合并基础相似度和个人偏好

$$
\boxed{S_{u,j,l,r}^{retriever}
=(1-\lambda)S_{b_{u,j},b_{l,r}}^{retriever}
+\lambda S_{u,r}^{retriever}}\tag{6}
$$

**输入**：两个评分与权重 $\lambda$。**输出**：用于排序的单个检索分数。

#### 3.5 ：逐条候选行为送入 LLM 做反馈评估

$$
\boxed{O_{b_{u,j},b_{l,r}}=\mathrm{LLM}(b_{u,j},b_{l,r})},\quad
l\in\{1,\ldots,m+1\},\ r\in\{1,\ldots,k\}.
\tag{7}
$$

**输入**：一个结构化 Prompt，包含目标用户与新商品信息，以及一条候选行为。**输出**：LLM 对目标用户是否会交互的预测反馈。

#### 3.6 ：正确答案的对数概率就是行为反馈分数



$$
\boxed{\mathrm{eval}(y,O_{b_{u,j},b_{l,r}})
=\log p_{\mathrm{LLM}}(y\mid b_{u,j},b_{l,r})}\tag{9}
$$

**输入**：LLM 输出与真实标签。**输出**：一个标量行为反馈分数。

#### 3.7 ：把所有 LLM 反馈转成一个重要性分布

$$
\boxed{
 p_{\mathrm{LLM}}(b_{l,r}\mid b_{u,j},y)
 =\frac{\exp\left(\mathrm{eval}(y,O_{b_{u,j},b_{l,r}})\right)}{
 \displaystyle\sum_{a=1}^{m+1}\sum_{q=1}^{k}
 \exp\left(\mathrm{eval}(y,O_{b_{u,j},b_{a,q}})\right)}
}\tag{8}
$$

**输入**：所有候选行为的 LLM 反馈分数。**输出**：所有候选行为上的概率分布，总和为 $1$。**作用**：把“哪条行为更能帮助 LLM 预测正确答案”变成可用于监督检索器的相对权重。


#### 3.8 ：检索器也生成自己的行为重要性分布

$$
\boxed{
 p_{retriever}(b_{l,r}\mid b_{u,j})
 =\frac{\exp(S_{u,j,l,r}^{retriever})}{
 \displaystyle\sum_{a=1}^{m+1}\sum_{q=1}^{k}
 \exp(S_{u,j,a,q}^{retriever})}
}\tag{10}
$$

**输入**：公式（6）输出的全部候选行为分数。**输出**：总和为 $1$ 的检索器行为分布。

**（8）与（10）最大的区别**：前者由 LLM 对**真实答案的支持程度**决定；后者由**可学习的向量检索评分**决定。两种分布对应同一组候选行为，但可能不同。作者希望检索器向前者学习。

#### 3.9 公式（11）：KL 散度把 LLM 反馈传给检索器

$$
\boxed{L_{retriever}
=D_{\mathrm{KL}}\!\left(p_{retriever}\parallel p_{\mathrm{LLM}}\right)}\tag{11}
$$

展开后，若候选行为索引为 $c$：

$$
D_{\mathrm{KL}}(P\parallel Q)
=\sum_c P_c\log\frac{P_c}{Q_c},
\qquad P=p_{retriever},\quad Q=p_{\mathrm{LLM}}.
$$

- $P_c$：候选行为 $c$ 在检索器分布中的权重；$Q_c$：在 LLM 反馈分布中的权重。
- 例如 LLM 认为三条行为的重要性是 $[0.7,0.2,0.1]$，检索器却给出 $[0.2,0.3,0.5]$，两者分布不一致，需要训练检索器。
- KL 散度一般**不对称**：$D_{\mathrm{KL}}(P\parallel Q)\neq D_{\mathrm{KL}}(Q\parallel P)$；公式实现时不能随意互换顺序。

**输入**：两组候选行为上的分布。**输出**：一个可用于优化的损失标量。**目的**：让检索器以后能更准确地选出真正帮助下游 LLM 正确判断的行为，而不只选语义最相似的物品。

这种机制可以类比为用 LLM 的预测反馈“教”检索器，但不要简单说成“LLM 取代了检索器”：**最终实际排序仍使用检索器的公式（6）评分。**

#### 3.10 本模块最终输出：Top-$N$ 行为

根据公式（6）的检索分数，综合排序初步得到的 $\mathcal B^{retrieved}$，再取前 $N$ 条：

$$
\mathcal B^{ranked}=\{b_1,b_2,\ldots,b_N\},\qquad N=5\ \text{（论文默认值）}.
$$

- $\mathcal B^{ranked}$：最终用于推荐 Prompt 的行为集合；$b_i$：排名第 $i$ 的行为；$N$：最终保留的行为数量。
- 不要求从每个相似用户那里平均分配行为：某人可以贡献多条，也可以一条不入选。
- **训练中**利用 LLM 反馈优化打分；**推理中**根据训练后的打分函数直接做筛选，不需要用目标样本的真实答案评估行为。


### 4. Behavior-Guided Hybrid Recommendation：行为引导的混合推荐

#### 4.1 文本与嵌入向量

- **自然语言文本**可以说明“哪个用户与哪个物品交互过”，包含商品标题、类别和其他可用属性，方便 LLM 处理明确的语义关系。
- **向量表示**保存了之前内容编码器和用户序列模型学到的稠密特征。直接把向量打印成 `[0.1,0.3,...]` 的普通文本。

使用 Hybrid Prompt，包含任务指令、目标用户、候选新商品、筛选出的历史行为，以及相应的用户和物品嵌入。
#### 4.2 ：共享 Adapter 把外部向量映射进 LLM 输入空间

$$
\boxed{\widetilde{\mathbf e}_u=f(\mathbf e_u),\qquad
\widetilde{\mathbf z}_j=f(\mathbf z_j)}\tag{12}
$$

论文的适配器是两层瓶颈结构 `Linear–ReLU–Linear`。

$$
f(\mathbf x)=\mathbf W_2\,\mathrm{ReLU}(\mathbf W_1\mathbf x+\mathbf b_1)+\mathbf b_2.
$$

其中 $\mathbf x$ 是待转换向量，$\mathbf W_1,\mathbf W_2$ 是两个线性层权重，$\mathbf b_1,\mathbf b_2$ 是偏置，$\mathrm{ReLU}$ 是非线性激活函数。Adapter 的输出要与 LLM 的输入嵌入空间兼容。
**输入**：用户或商品的原始向量。**输出**：适合注入 LLM 输入嵌入序列的向量。**目的**：推荐模型向量与 LLM Token Embedding 即使维度相同也不一定处于相同的语义/数值空间，需要可学习映射。

#### 4.3 ：把适配后的向量注入特殊嵌入位置

$$
\boxed{[\mathrm{USER\_EMBED}]=\widetilde{\mathbf e}_u,\qquad
[\mathrm{ITEM\_EMBED}]=\widetilde{\mathbf z}_j}\tag{13}
$$

把转换后的连续向量注入相应的嵌入位置，使 LLM 可以在处理文本之外接收外部推荐特征。

**输入**：结构化 Prompt + 适配后的用户/物品向量。**输出**：包含自然语言和外部嵌入的 LLM 输入。**目的**：融合可读的行为描述与推荐模型学习到的稠密特征。

Prompt 示意：

```text
Instruction: 判断目标用户是否会与新商品交互，只回答 Yes 或 No。
Target user: 用户 A；属性……；[USER_EMBED]
New item: 新上架防晒霜；标题/类别……；[ITEM_EMBED]
Informative behaviors:
  Behavior 1: 用户 B（含其向量）与防晒喷雾（含其向量）交互：Yes
  Behavior 2: 用户 A（含其向量）与补水面膜（含其向量）交互：Yes
  ……
Output: Yes / No
```


#### 4.4 ：用二元交叉熵训练 Adapter

$$
\boxed{L_{ad}=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)\right]}\tag{14}
$$

假如真实标签是 $y_i=1$，则该条样本损失为 $-\log(\hat y_i)$：$\hat y_i=0.9$ 时约 $0.105$；$\hat y_i=0.1$ 时约 $2.303$。模型越不支持正确答案，损失越高。若真实标签为 $0$，损失变成 $-\log(1-\hat y_i)$。

**输入**：LLM 的 Yes 概率和真实标签。**输出**：平均 BCE 损失。**训练目的**：让 Adapter 学会把外部用户/商品向量转成对 LLM 有用的输入表示。

**LLM 主体参数被冻结**，只优化输入侧共享 Adapter。

#### 4.5 最终预测输出

给定混合 Prompt，冻结的 LLM 受指令约束生成一个标签：`Yes` / `No`，表示用户 $u$ 是否可能与新物品 $j$ 交互。评估预测得分计算 Recall@K、NDCG@K 等排名指标。

第二个模块已经把大量历史筛到少数有用行为。额外将用户和商品向量经过共享 Adapter 映射到 LLM 输入嵌入空间，使自然语言描述与稠密特征共同指导 LLM。LLM 主体保持冻结，利用推荐标签训练 Adapter，完成 Yes/No 预测。

### 5. ：三个损失共同优化

$$
\boxed{L_{total}=L_{CL}+\alpha L_{retriever}+\beta L_{ad}}\tag{15}
$$

代入论文默认值：

$$
L_{total}=L_{CL}+0.5L_{retriever}+0.3L_{ad}.
$$

好的用户表示帮助找到靠谱近邻；好的行为检索帮助 LLM 获得有效证据；LLM 输出又反过来提供检索监督；Adapter 保证外部嵌入能真正进入 LLM 的表征空间。这些目标存在联系，因此作者把对应损失组合起来。





## 结果

在 Amazon Baby、Amazon Clothing/Shoes/Jewelry、TikTok 三个数据集的模拟新物品设置中，SuperG 的 Recall@10 相对最强对照 MLLM-MSR 分别提高 14.8%、16.9%、16.4%。Baby 上 Recall@10 为 0.0986，NDCG@20 为 0.0720。

消融实验：去掉相似用户探索、信息性行为挖掘或行为引导模块后，性能均下降。

## 存在的不足

- 对用户历史依赖较强。 SuperG 需要根据历史交互寻找相似用户，再筛选有用的行为。对于刚注册或交互较少的用户，可能难以找到可靠的近邻，也缺少足够的历史行为。论文主要解决新物品的冷启动问题，这类用户的推荐效果还需要验证。

- 实际推荐速度有待验证。 模型需要经过相似用户检索、历史行为筛选和大语言模型预测，处理步骤较多。在电商、短视频等需要频繁推荐的场景中，能否同时满足推荐速度和计算成本的要求，还需要测试。

- 代码仓库找不到。
  目前想不到其他的。
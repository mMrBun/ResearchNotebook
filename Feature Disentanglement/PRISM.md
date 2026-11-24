# PRISM: Personalized Recommendation via Information Synergy  Module

- 发表在：*Accepted as a Full Paper at WWW 2026*  [论文链接](https://doi.org/10.48550/arXiv.2601.10944) [代码仓库](https://github.com/YutongLi2024/PRISM)

- 作者 Zhang, Xinyi, Yutong Li, Peijie Sun, Letian Sha和Zhongxuan Han. **Imperial College London  London, UK** , **Telecommunications  Nanjing, China** , **Zhejiang University Hangzhou, China**

## 问题分析
1、现有方法多关注图片特有、文本特有以及图文共有信息，却漏掉只有图文结合后才有的信息。
2、对于不同用户对不同模态的偏好程度现有工作多采用按模态分配权重，但粒度太粗不能显著区分用户当前更关注图片？文本？

## 方法
作者采用两层设计分别解决上述两种局限。
1、交互专家层
2、自适应融合层

### 交互专家层

#### 专家输出与训练策略

设置四个专家网络（MLP）将图文表示分成
1、图像特有
2、文本特有
3、协同信息（只有图文结合后才涌现的信息，去掉其中任何一个将无法预测）
4、共享冗余（两侧都存在的信息，拿掉其中任何一测都可由另一测补充）

$$
\mathbf e^j=E_j(\mathbf e^{img},\mathbf e^{txt}),\quad
j\in\{\text{uni-i, uni-t, syn, rdn}\}.
\qquad (1)
$$

改变输入条件得到图文全部输入的完整预测、遮蔽文本输入的图像预测以及遮蔽图像的文本预测

由(1)得到的融合嵌入 $E_j$ 结合商品id嵌入交给序列编码器 $S_j
$ 最后由预测层 $P_j$得到推荐预测 $y_j$


$$
\mathbf y_j=P_j(S_j(\mathbf e^{id},E_j(\mathbf e^{img},\mathbf e^{txt}))),\quad
j\in\{\text{uni-i, uni-t, syn, rdn}\}.
\qquad (2)
$$

$$
\mathbf y_j^{img}=P_j\bigl(S_j\bigl(E_j(\mathbf e^{img},\mathbf r),\mathbf e^{id}\bigr)\bigr),\quad
j\in\{\text{uni-i, uni-t, syn, rdn}\},
\qquad (3)
$$

$$
\mathbf y_j^{txt}=P_j\bigl(S_j\bigl(E_j(\mathbf e^{txt},\mathbf r),\mathbf e^{id}\bigr)\bigr),\quad
j\in\{\text{uni-i, uni-t, syn, rdn}\},
\qquad (4)
$$

#### 交互损失

训练图像独有专家
训练文本独有专家
使用三元组损失让正样本预测尽可能靠近完整的图文预测

$$
\mathcal L_{\text{uni-i}}=\mathcal{TripletLoss}(\mathbf y,\mathbf y^{img},\mathbf y^{txt}).
\qquad (5)
$$

$$
\mathcal L_{\text{uni-t}}=\mathcal{TripletLoss}(\mathbf y,\mathbf y^{txt},\mathbf y^{img}).
\qquad (6)
$$

训练协同专家，鼓励完整预测远离两种遮蔽预测。



$$
\mathcal L_{\text{syn}}=\frac{1}{2}
\left[\mathcal{CosSim}(\mathbf y,\mathbf y^{img})
+\mathcal{CosSim}(\mathbf y,\mathbf y^{txt})\right],
\qquad (7)
$$

训练共享冗余专家，这里遮蔽预测被当作正样本，希望无论遮蔽哪种模态信息都能靠近完整模态预测。


$$
\mathcal L_{\text{rdn}}=1-\frac{1}{2}
\left[\mathcal{CosSim}(\mathbf y,\mathbf y^{img})
+\mathcal{CosSim}(\mathbf y,\mathbf y^{txt})\right].
\qquad (8)
$$

联合损失

$$
\mathcal L_{\text{exp}}
=\lambda_{\text{uni-i}}\cdot\mathcal L_{\text{uni-i}}
+\lambda_{\text{uni-t}}\cdot\mathcal L_{\text{uni-t}}
+\lambda_{\text{syn}}\cdot\mathcal L_{\text{syn}}
+\lambda_{\text{rdn}}\cdot\mathcal L_{\text{rdn}},
\qquad (9)
$$

### 自适应融合层

用历史交互序列中的id嵌入作为偏好基础协助计算每类信息的重要性，这里W也使用MLP

$$
w^j=\mathrm{W}\bigl(
\mathbf e^{\text{uni-i}},\mathbf e^{\text{uni-t}},
\mathbf e^{\text{syn}},\mathbf e^{\text{rdn}},\mathbf e^{id}
\bigr),\quad
j\in\{\text{uni-i, uni-t, syn, rdn}\},
\qquad (10)
$$

按照各自的权重融合各多模态表示

$$
\mathbf e^m=\sum_j w^j\mathbf e^j,\quad
j\in\{\text{uni-i, uni-t, syn, rdn}\}.
\qquad (11)
$$

### 预测与优化

PRISM的设计是可插拔的，$`L_{rec}`$ 可以是任何骨干模型的损失函数，再加上公式9的exp损失用于模型训练。

$$
\mathcal L=\mathcal L_{\mathrm{rec}}+\mathcal L_{\mathrm{exp}}.
\qquad (12)
$$

### 结果
论文在 Home、Beauty、Sports 和 Yelp 四个数据集上测试了 PRISM。把它分别接入 SASRec、STOSA 和 InDiRec 后，三种骨干模型报告的 Recall 和 NDCG 都比各自的原版高。在 Yelp 上，完整的 PRISM+InDiRec 的 Recall@20 为 0.0622，去掉协同专家后降到 0.0571。去掉其余三个专家中的任意一个，指标也会下降。这说明四个分支在这组实验设置下都有贡献。
| 模型（Yelp） | Recall@20 | NDCG@20 |
|---|---:|---:|
| 原版 InDiRec | 0.0551 | 0.0301 |
| **PRISM + InDiRec** | **0.0622** | **0.0352** |
| 去掉图像独有专家 | 0.0582 | 0.0320 |
| 去掉文本独有专家 | 0.0570 | 0.0300 |
| 去掉协同专家 | 0.0571 | 0.0301 |
| 去掉共享冗余专家 | 0.0594 | 0.0325 |

### 本地复现结果
### Beauty 数据集 PRISM + STOSA 测试结果

| 结果 | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| 本地复现（5 个随机种子均值） | 0.0727 | 0.0993 | 0.0451 | 0.0518 |
| 论文表 2（5 个随机种子均值） | 0.0762 | 0.1083 | 0.0481 | 0.0549 |
| 差值 | −0.0035 | −0.0090 | −0.0030 | −0.0031 |

### Beauty 数据集 PRISM + InDiRec 测试结果

| 结果 | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| 本地复现（5 个随机种子均值） | 0.0913 | 0.1250 | 0.0558 | 0.0642 |
| 论文表 2（5 个随机种子均值） | 0.1012 | 0.1388 | 0.0618 | 0.0711 |
| 差值 | −0.0099 | −0.0138 | −0.0060 | −0.0069 |


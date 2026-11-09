## 已阅读文献

<details>
<summary>多模态推荐</summary>

1. Zhang, Shengzhe, Liyi Chen, Dazhong Shen, Chao Wang, and Hui Xiong. “Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation.” *Proceedings of the ACM Web Conference 2025 (WWW ’25)*, 2025. [https://doi.org/10.1145/3696410.3714676](https://doi.org/10.1145/3696410.3714676)

2. Dang, Yuzhuo, Xin Zhang, Zhiqiang Pan, Yuxiao Duan, Wanyu Chen, Fei Cai, and Honghui Chen. “MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation.” *arXiv preprint* arXiv:2508.15304v2, 2025. [https://arxiv.org/abs/2508.15304v2](https://arxiv.org/abs/2508.15304v2)

3. Kim, Sunwoo, Hyunjin Hwang, and Kijung Shin. “Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation.” *Proceedings of the ACM Web Conference 2026 (WWW ’26)*, 2026. [https://doi.org/10.1145/3774904.3792248](https://doi.org/10.1145/3774904.3792248)

4. Xie, Jianye, Lianyong Qi, Weiming Liu, Anqi Wang, Xiaolong Xu, Haolong Xiang, Xuyun Zhang, Wenwen Gong, Yang Zhang, Amin Beheshti, and Wanchun Dou. “Joint Similar User Exploration and Informative Behavior Guidance for Multi-Modal New Item Recommendation.” *Proceedings of the ACM Web Conference 2026 (WWW ’26)*, 2026. [https://doi.org/10.1145/3774904.3792302](https://doi.org/10.1145/3774904.3792302)

5. Lian, Yahong, Haotian Tian, Chunyao Song和Tingjian Ge. Sign-Aware Multimodal Graph Recommendation. 2026年3月14日. *Proceedings of the AAAI Conference on Artificial Intelligence*
 [https://doi.org/10.1609/aaai.v40i18.38547](https://doi.org/10.1609/aaai.v40i18.38547)

 6. Zhan, Xiaohan, Yuliang Shi, Jihu Wang, Shijun Liu, Fanyu Kong和Zhiyong Chen. DHMRec: Collaboration-Guided Multimodal Disentanglement and Hierarchical Fusion for Recommendation. 2026-03-14 *Proceedings of the AAAI Conference on Artificial Intelligence*
 [ https://doi.org/10.1609/aaai.v40i19.38661]( https://doi.org/10.1609/aaai.v40i19.38661)





</details>



<details>
<summary>备选文章</summary>

**1. D3ER — ACM Multimedia 2026，主会长文，已录用**

完整题名：**D3ER: Supporting Multi-Modal Recommendation via Disentangle and Distillation-based Dynamic Ensemble**

这是与你读过的 **DHMRec 最接近**的一篇。

- **分解什么：** 图文共有信息、图像特有信息、文本特有信息。
- **怎样分解：** 共有部分使用实例级对比对齐和 Wasserstein 分布对齐；特有部分通过带阈值的差异约束与共有部分分离。
- **值得关注：** 分解之后，它用蒸馏增强的梯度提升方法交替训练不同信息分支，重点讨论“共有和特有信息一起训练，是否会削弱各自的作用”。

作者稿首页注明主会 **10 页论文**，会议官方日程也已收录。截至今天，状态为已录用，会议将于 11 月 10—14 日举行。[论文全文](https://arxiv.org/html/2608.25737v1) · [PDF](https://arxiv.org/pdf/2608.25737) · [官方日程](https://2026.acmmm.org/site/ACMMM2026_Full_Programme.pdf)

**我的阅读建议：先看 §3.2 Feature Component Disentanglement，与 DHMRec 的 JSD＋正交损失逐项比较。**

---

**2. PRISM — WWW 2026，Research Track Full Paper**

完整题名：**PRISM: Personalized Recommendation via Information Synergy Module**

这篇直接做**多模态序列推荐**，与我们前面讨论的“二分之外还有什么信息”非常相关。

- **分解什么：** 图像独有、文本独有、冗余、协同信息。
- **怎样分解：** 设置四个专家网络，通过随机替换一个模态，比较完整输入与扰动输入下的推荐输出，用不同损失引导专家分工。
- **值得关注：** 根据用户偏好，对不同信息类型进行动态融合；重点是“两种模态结合后才产生的协同作用”。

已核对 WWW 官方 Research Tracks 录用列表，条目为 **rfp1121**。[官方录用列表](https://www2026.thewebconf.org/accepted/research-tracks.html) · [论文全文](https://arxiv.org/html/2601.10944v1) · [作者提供的代码](https://github.com/YutongLi2024/PRISM)

**我的阅读建议：重点看 §2.3 的专家训练方式。注意，它的“独有／冗余／协同”与 MPD 的 MP／AO／UP 是不同定义。**

---

**3. TSD — CVPR 2026，主会论文，8791–8800 页**

完整题名：**Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis**

虽然任务是多模态情感分析，但它的**三类子空间分解**很值得借鉴。

- **分解什么：** 所有模态共同的信息、仅在某一对模态之间共享的信息、单个模态私有的信息。
- **怎样分解：** 采用不同子空间编码器，结合一致性、依赖度和正交等约束，再进行融合。
- **值得关注：** 三个及以上模态中，一条信息可能只被其中两个模态共享，不能简单归为“全局共有”或“完全私有”。

已核对 CVF 正式论文目录与 CVPR 主会日程。[官方目录](https://openaccess.thecvf.com/CVPR2026?day=2026-06-05) · [作者全文](https://arxiv.org/html/2602.19585v1) · [PDF](https://arxiv.org/pdf/2602.19585)

**我的阅读建议：如果以后想加入评论、品牌等更多信息源，这篇的分组方式尤其值得看；仅有图文两个模态时，不能直接照搬其三模态结构。**

---

**4. PLUM-Net — AAAI 2026，主会论文，18611–18619 页**

完整题名：**PLUM-Net: Prototype-Induced Label Structuring for Disentangled Multimodal Representation Network**

- **分解什么：** 多模态的共有表示与私有表示。
- **怎样分解：** 利用原型产生单模态硬、软标签，再通过任务条件特征分流器引导共有／私有通路。
- **值得关注：** 它提供了“利用标签和任务信息指导分解”的路线，可与 DHMRec 的协同信号门控进行比较。

这篇主要用于情感、幽默等多模态理解任务，并非推荐论文。[AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/38928) · [官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38928/42890)

---

**5. IDMVAE — ICLR 2026，正式会议论文**

完整题名：**Disentanglement of Variations with Multimodal Generative Modeling**

- **分解什么：** 多模态共享潜变量与模态私有潜变量。
- **怎样分解：** 通过跨视图互信息最大化提取共享信息，利用生成增强和循环一致性式损失减少冗余，并引入扩散模型增强潜变量先验。
- **值得关注：** 它重点处理“虽然设置了共有和私有两个分支，但信息仍可能相互混杂”的问题。

这篇适合深入理解**怎样约束分解真正发生**，但需要一些 VAE、互信息基础，阅读门槛比前几篇高。[ICLR 官方论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/581e1a06fa20f2c079dc5fb2db236335-Abstract-Conference.html) · [作者稿](https://arxiv.org/abs/2509.23548)

---

**6. FD-CMKD — AAAI 2026，主会论文，23739–23747 页**

完整题名：**Distilling Cross-Modal Knowledge via Feature Disentanglement**

- **分解什么：** 特征中的低频与高频成分。
- **怎样分解：** 作者根据其任务中的跨模态一致性差异，对低频信息施加强对齐，对高频信息采用较宽松的对齐，并加入尺度一致性约束。
- **值得关注：** 它展示了一种与“多个编码器分别提取共有／特有”不同的路线：**从频域分解入手，再区别处理不同成分**。

任务是跨模态知识蒸馏，不能直接把它的低频／高频结论当作所有推荐特征都成立的规律。[AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/39548) · [作者 PDF](https://yangrenyu.github.io/paper/aaai2026.pdf) · [作者代码](https://github.com/Johumliu/FD-CMKD)

---

**7. DANTE — AAAI 2026，主会论文，13052–13060 页**

完整题名：**Disentangling for Transfer: Boosting Limited Modalities via Information-Theoretic Regularization and Cross-Modal Reconstruction**

- **分解什么：** 可跨模态对齐的信息，以及特权模态独有的信息。
- **怎样分解：** 结合跨模态重构预训练、双向对齐和特权独有信息压缩。
- **值得关注：** 研究训练时拥有丰富模态、测试时只能使用有限模态时，哪些知识应当迁移。

这篇主要面向医疗场景。如果你以后关注商品缺图、缺文本等模态缺失问题，可以进一步阅读。[AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/38305) · [官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38305/42267)

**结合你目前的基础，我建议先读 D3ER → PRISM → TSD。** D3ER 方便与你熟悉的 DHMRec 对照；PRISM 帮你理解“共有／特有之外的协同信息”；TSD 则拓展到多于两个模态的分解。以上是基于方法相关性的阅读顺序，不是论文质量排名。

</details>
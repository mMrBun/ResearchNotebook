# Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation

- 发表在：ACM Web Conference 2026（WWW ’26）会议论文集 [论文链接](https://doi.org/10.1145/3774904.3792248) [项目代码](https://github.com/kswoo97/PerPEFT)

- 作者 Sunwoo Kim、Hyunjin Hwang、Kijung Shin **韩国科学技术院（KAIST）**

## 解决的问题

现有方法使用CLIP 等多模态基础模型对物品图像和文本编码，再用PEFT适配推荐任务，但通常所有用户共享一套微调参数，使同一物品得到与用户兴趣无关的表示。不同用户可能关注同一物品的不同属性，仅靠后续用户嵌入难以让编码器捕捉这些细粒度差异。

## 使用的方法

先训练共享 PEFT 的 GlobalPEFT，以 SASRec 的用户预测向量表示兴趣，并用 K-Means 把用户分组；然后为每组分配独立的 PEFT 模块和投影器，使 CLIP 为同一物品生成组条件化的图文表示。物品 ID 嵌入和 SASRec 主干在组间共享。训练时从同组用户购买过的物品池抽负样本，让模型区分组内更难的候选；同组用户组成小批次以复用物品编码。可接 LoRA、(IA)³、IISAN 等 PEFT 模块。


## 结果
在 Amazon 2023 的 Sports & Outdoors、Toys & Games、Beauty & Personal Care、Arts/Crafts/Sewing 四个领域上，PerPEFT 在 48 项设置中的 44 项优于共享微调 GlobalPEFT； NDCG@20 相对最强基线最高提升 15.3%。

在 LoRA、(IA)³ 和 IISAN 三种 PEFT 下都有相同的增强；消融实验结果：按兴趣分组、组内负采样，以及同时使用文本与图像均有增强。

## 存在的不足（目前能想到的）
- 作者只在Amazon商品数据集上进行测试，缺少跨平台的对照。
- 用户兴趣是随时间变化的，尤其是在短视频带货平台上，用户兴趣变化速度快，作者的方法需要先训练全局的PEFT模块，再训练组PEFT，分组模块的动态更新怎么做？
- 每个人的兴趣都不一样，组内的PEFT只能代表一类人有某一个共同兴趣爱好，不能代表组内每个人兴趣都一模一样，这会掩盖掉同组用户的差异。



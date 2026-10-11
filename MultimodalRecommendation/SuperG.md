# Joint Similar User Exploration and Informative Behavior Guidance for Multi-Modal New Item Recommendation

- 发表在：ACM Web Conference 2026（WWW ’26）会议论文集 [论文链接](https://doi.org/10.1145/3774904.3792302) 

- 作者 Jianye Xie、Lianyong Qi、Weiming Liu、Anqi Wang、Xiaolong Xu、Haolong Xiang、Xuyun Zhang、Wenwen Gong、Yang Zhang、Amin Beheshti、Wanchun Dou **中国石油大学（华东）** **山东省智能油气工业软件重点实验室** **字节跳动** **南京信息工程大学** **麦考瑞大学** **中国农业大学** **北德克萨斯大学** **南京大学**


## 解决的问题

- 在缺少新物品交互数据的条件下，找出可靠的相似用户，并从其与目标用户的历史中提取有效证据。

新上架的物品有图像、文本等内容，却没有交互记录。现有方法通常仅凭物品共同交互判断用户相似性，把偶然接触热门物品的用户当成近邻；同时，用户历史行为有噪声，难以挑出真正有助于推荐新物品的行为。

## 使用了什么方法
SuperG由**相似用户探索**、**信息性行为挖掘**和**行为引导的混合推荐**三部分构成。
相似用户探索：将历史交互序列送入Transformer，利用序列增强与对比学习得到更有区分度的用户表示，再检索相似用户。
信息性行为挖掘：从目标用户及其相似用户的历史中检索并重排与候选新物品相关的行为，结合大语言模型反馈筛选有用证据。
行为引导的混合推荐：把行为证据与多模态物品表示结合，引导大语言模型（Llama-3-8B-Instruct）给出推荐。

## 结果
在 Amazon Baby、Amazon Clothing/Shoes/Jewelry、TikTok 三个数据集的模拟新物品设置中，SuperG 的 Recall@10 相对最强对照 MLLM-MSR 分别提高 14.8%、16.9%、16.4%。Baby 上 Recall@10 为 0.0986，NDCG@20 为 0.0720。

消融实验：去掉相似用户探索、信息性行为挖掘或行为引导模块后，性能均下降。

## 存在的不足
- 代码仓库找不到。
目前想不到其他的。
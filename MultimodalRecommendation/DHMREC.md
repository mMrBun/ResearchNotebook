# DHMRec: Collaboration-Guided Multimodal Disentanglement and Hierarchical Fusion for Recommendation



# 问题分析
主要在做特征结偶和层次融合
商品的图片和文字可能有三类信息

- 两边都能体现的信息：颜色、款式。
- 各自模态正擅长表达的信息：图像中衣服的纹理、图案；文字中的材质类型。
- 对推荐没有帮助的信息：模特的发型。

有用的信息和噪声混在一起，不同的信息对推荐的作用不同，简单的平均融合不合适。


# 方法
### 模态解耦模块
商品的文本和图像特征已由sentence-transformer及CNN预先提取，个经过一次线性变换得到纬度相同的$Z_m$。

设置两个编码器：公共编码器 $E_{C_m}$，私有编码器 $E_{P_m}$ 将每种模态分成两条支路。两种编码器采用线性网络。

$$
C_{i,m}=E_{C_m}(Z_m)
$$









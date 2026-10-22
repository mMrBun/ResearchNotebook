## 声明

本代码是Lian, Yahong, Haotian Tian, Chunyao Song和Tingjian Ge. Sign-Aware Multimodal Graph Recommendation. 2026年3月14日. *Proceedings of the AAAI Conference on Artificial Intelligence*
的代码复现目录

源仓库地址https://github.com/DRec4AI/SiMGR2025

源仓库中没有requirements.txt文件，无法确定完整的依赖版本。

源码中存在无用代码导致报错等情况。

本次复现引入uv作为python包管理器，方便后续在新的环境中一键复现完整环境。

在跑代码之前请在data目录中完成数据集的下载并解压。
数据集地址：[Baby/Sports/Elec](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) 

## sport数据集复现结果
```log
Wed 21 Oct 2026 17:37:05 INFO 	Parameters: ['seed']=(999,),
Valid: recall@5: 0.0442    recall@10: 0.0673    recall@20: 0.0993    recall@50: 0.1560    ndcg@5: 0.0299    ndcg@10: 0.0374    ndcg@20: 0.0455    ndcg@50: 0.0569    precision@5: 0.0093    precision@10: 0.0071    precision@20: 0.0052    precision@50: 0.0033    map@5: 0.0248    map@10: 0.0279    map@20: 0.0301    map@50: 0.0319    ,
Test: recall@5: 0.0461    recall@10: 0.0695    recall@20: 0.1025    recall@50: 0.1586    ndcg@5: 0.0310    ndcg@10: 0.0386    ndcg@20: 0.0471    ndcg@50: 0.0585    precision@5: 0.0101    precision@10: 0.0076    precision@20: 0.0056    precision@50: 0.0035    map@5: 0.0253    map@10: 0.0284    map@20: 0.0306    map@50: 0.0324    
```



## 复现步骤
### 强烈推荐使用uv进行包管理
[安装uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

进入项目文件夹

```bash
cd CodeReproduction/SiMGR2025
```

安装项目依赖

```bash
uv sync
```

源码运行需要进入到src目录作为workspace，可以在调试配置文件中修改cwd实现，或者直接cd到src

```bash
cd src
```

运行项目(baby dataset),其他数据集只需修改数据集名称
```bash
uv run python main.py -m SiMGR -d baby
```

### conda
创建虚拟环境
```bash
conda create -n SiMGR python=3.10.22
```
激活虚拟环境
```bash
conda activate SiMGR
```
安装依赖
```bash
pip install -r requirements.snapshot.txt
```
源码运行需要进入到src目录作为workspace，可以在调试配置文件中修改cwd实现，或者直接cd到src

```bash
cd src
```
运行代码(baby dataset),其他数据集只需修改数据集名称
```bash
python main.py -m SiMGR -d baby
```
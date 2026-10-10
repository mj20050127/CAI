# 当代人工智能课程实验

实验一研究基于TF-IDF的10类文本分类，比较MLP与线性SVM，并包含网络结构、正则化、学习率、特征维度、数据清洗及多随机种子的对照实验。

详细实验指导、16个脚本的作用、运行顺序和输出位置见 [实验一复现指南](data-Project1/README.md)。

## 开始运行

在Anaconda Prompt中进入克隆后的仓库目录，执行：

```bat
conda create -n project1 python=3.10
conda activate project1
python -m pip install -r requirements.txt
cd data-Project1
```

将老师提供的 `train_data.csv` 和 `test_data_unlabeled.csv` 放到 `data-Project1` 目录。这两个原始数据文件不包含在仓库中。

按详细指南逐项运行实验；如果仅需按已确定的配置全量重训并生成预测，执行：

```bat
python final_predict_mlp.py
```

预测写入 `data-Project1/predictions.csv`，为无表头的单列整数标签，共2457行。仓库同时保留已有预测、各阶段汇总与配置，以及P1损失图重绘所需的两份训练记录。完整实验中的模型和其他训练记录由脚本重新生成。

首次运行M2正则扩展需要先运行原M2生成其CSV及joblib；两组维度扩展需要对应原阶段CSV。具体依赖详见实验指南。

# 实验一 文本分类实验复现指南

本项目使用TF-IDF、MLP和线性SVM完成10类新闻邮件分类，比较网络结构、正则化、学习率、特征维度和邮件清洗的影响。新增实验包括15000/25000维对照、三随机种子重复验证，以及mini-batch损失图重绘。最终预测配置为P1清洗、20000维TF-IDF和单隐藏层MLP。

## 1. 环境和目录

使用Anaconda Prompt。首次配置环境时执行：

```bat
conda create -n project1 python=3.10
conda activate project1
python -m pip install numpy==2.2.5 pandas==2.3.3 scikit-learn==1.7.1 matplotlib==3.10.9 joblib==1.5.3
```

本次实验实际使用Python 3.10.21及上述依赖版本。环境已配置好时，只需执行 `conda activate project1`。

进入解压后的 `data-Project1` 目录再运行脚本。下面是占位示例，请替换成自己的路径：

```bat
cd /d "D:\你的解压目录\data-Project1"
```

多数实验使用相对路径读取数据、保存结果，因此当前工作目录必须是 `data-Project1`。所有Python文件保留在同一目录，避免扩展脚本和清洗脚本导入失败。实验使用CPU和scikit-learn，无需CUDA；运行时间受硬件、线程库和其他任务影响。

## 2. 数据与统一实验规则

将老师提供的以下文件放在脚本同目录，保持原文件名：

| 文件 | 内容 | 字段 |
|---|---|---|
| `train_data.csv` | 7368条有标签邮件 | `text`、`target` |
| `test_data_unlabeled.csv` | 2457条官方无标签邮件 | `text` |

类别编号为0至9。常规验证实验按8:2分层划分有标签数据，得到5894条训练样本、1474条验证样本，划分与模型随机种子为42；MLP批次打乱种子也是42。多随机种子实验另见第5节。

- 先划分，再仅在训练子集上拟合TF-IDF词表和IDF，验证集只调用 `transform`。
- 使用 `lowercase=True`、`ngram_range=(1, 1)`；维度实验改变 `max_features`，实际维度记录在 `actual_feature_count`。
- 以验证Macro-F1为选模主指标，Accuracy为辅助指标，结合分类报告和混淆矩阵分析错误。
- MLP调参和清洗脚本用mini-batch调用 `partial_fit`，跑满设定轮数，并保存验证Macro-F1最优轮次的模型，没有在最佳轮次自动停止。
- MLP记录交叉熵，SVM记录不含正则项的多分类hinge loss；两种损失定义不同，不直接比较数值大小。
- 部分阶段使用0.001 Macro-F1作为实用持平范围，再按代码中的规则推荐配置。这是选模规则，不是统计显著性检验。

初始 `draft_main.py` 和 `svm_experiment.py` 会读取并转换无标签测试文本以检查特征形状；测试数据不参与拟合或参数选择。后续调参、清洗及重复验证脚本只加载有标签数据。最终预测在方案固定后处理官方测试集。

## 3. 每个实验脚本的作用

### 3.1 初始模型对照

| 脚本 | 研究内容和作用 | 输出目录 |
|---|---|---|
| `draft_main.py` | 加载并检查数据，完成分层划分和5000维TF-IDF；比较隐藏层 `(50,)`、`(100,)`、`(100, 50)` 与alpha=0.0001/0.001的6组组合，每组50轮。用于分析网络宽度、增加隐藏层和正则化的影响。 | `outputs/mlp_comparison/` |
| `svm_experiment.py` | 在相同划分、原始文本和5000维TF-IDF上比较线性SVM，候选C=0.1、1、10、100，建立初始SVM对照并观察小C下的欠拟合。报告中的C=0.1和1结果来自该脚本。 | `outputs/svm_comparison/` |

`draft_main.py` 已包含双隐藏层深度对照，无需另找深度实验脚本。

### 3.2 MLP分阶段调参

以下阶段均使用原始邮件P0，batch大小为128。后续阶段的固定参数已经写入代码，通常不读取上一阶段模型；扩展脚本有额外文件依赖，见第4节。

| 脚本 | 改变的因素和作用 | 固定条件及轮数 | 输出目录（相对 `outputs/`） |
|---|---|---|---|
| `mlp_tuning_round2_stage1.py` | M1宽度：比较单隐藏层50、75、100、150、200个神经元，判断继续加宽是否有效。 | 5000维、alpha=0.0001、学习率0.001、60轮 | `mlp_tuning_round2/stage1_width/` |
| `mlp_tuning_round2_stage2.py` | M2正则：比较alpha=0.00001、0.0001、0.0003、0.001、0.003、0.01，寻找拟合能力与泛化的折中。 | 隐藏层100、5000维、学习率0.001、60轮 | `mlp_tuning_round2/stage2_alpha/` |
| `mlp_tuning_round2_stage2_extension.py` | M2扩展：新增alpha=0.03、0.1，与原M2的alpha=0.01基线合并，检验继续增大正则是否出现欠拟合。 | 沿用原M2条件、复用alpha=0.01基线、60轮 | `mlp_tuning_round2/stage2_alpha_extension/` |
| `mlp_tuning_round2_stage3.py` | M3学习率：比较0.0003、0.0005、0.001、0.002，分析收敛速度和验证性能。 | 隐藏层100、alpha=0.01、5000维、80轮 | `mlp_tuning_round2/stage3_learning_rate/` |
| `mlp_tuning_round2_stage4.py` | M4维度：比较5000、10000、20000维TF-IDF，分析词表覆盖的收益。 | 隐藏层100、alpha=0.01、学习率0.0005、80轮 | `mlp_tuning_round2/stage4_tfidf_dimension/` |
| `mlp_tuning_round2_stage4_extension.py` | M4扩展：只新增15000、25000维训练，再合并原M4三档结果形成五档对照，检验高维收益是否饱和。 | 完全沿用原M4配置、80轮 | `mlp_tuning_round2/stage4_tfidf_dimension_extension/` |

50、60、80是不同阶段预设的完整训练预算；`best_epoch` 是该预算内验证Macro-F1最优的轮次。跨阶段比较需同时查看固定配置和轮数，不能把所有差异都归因于一个参数。

### 3.3 SVM分阶段调参

| 脚本 | 研究内容和作用 | 固定条件 | 输出目录（相对 `outputs/`） |
|---|---|---|---|
| `svm_tuning_round2_stage1.py` | S1扩大C搜索：比较10、30、50、100、200、500，观察初始对照后继续增大C的收益。 | P0、5000维、线性核 | `svm_tuning_round2/stage1_c/` |
| `svm_tuning_round2_stage2.py` | S2维度：比较5000、10000、20000维，分析特征数量的影响。 | P0、线性核、C=500 | `svm_tuning_round2/stage2_tfidf_dimension/` |
| `svm_tuning_round2_stage2_extension.py` | S2扩展：新增15000、25000维并合并原S2结果，比较五档维度的F1、hinge loss和支持向量数。 | 完全沿用原S2配置 | `svm_tuning_round2/stage2_tfidf_dimension_extension/` |

SVM使用 `SVC(kernel="linear")`，不是 `LinearSVC`。其迭代次数由求解器决定，与MLP的epochs不同。

### 3.4 数据清洗对照

| 脚本 | 研究内容和作用 | 输出目录 |
|---|---|---|
| `mlp_preprocessing_experiment.py` | 固定20000维、隐藏层100、alpha=0.01、学习率0.0005、batch=128和80轮，对比P0至P3。保存清洗前后样例、文本统计、训练记录和验证最优模型，用于判断清洗是否删除有用信息。 | `outputs/mlp_preprocessing_comparison/` |
| `svm_preprocessing_p1_experiment.py` | 固定线性核、C=500和20000维，对比P0与P1，检验邮件头清洗是否也改善SVM。导入MLP清洗脚本的 `extract_subject_and_body`，保证P1规则一致。 | `outputs/svm_preprocessing_p1/` |

清洗采用累计对照：

| 配置 | 处理方法 |
|---|---|
| P0 | 保留原始邮件 |
| P1 | 仅保留Subject和正文，去掉其他邮件头 |
| P2 | P1基础上删除引用行和标准签名 |
| P3 | P2基础上通过TF-IDF删除英语停用词 |

本次选择P1。15000/25000维扩展使用P0，用于补充原维度实验；这些扩展没有重新搜索P1下的维度，也不会自动改变最终预测配置。

### 3.5 稳定性验证、绘图和最终预测

| 脚本 | 作用 | 输出 |
|---|---|---|
| `repeated_seed_experiment.py` | 固定P1和最终模型参数，用42、52、62重新划分、拟合TF-IDF、训练两种模型，报告均值与标准差。MLP每次固定39轮，不重新挑选最佳轮次。 | `outputs/repeated_seeds/` |
| `replot_p1_minibatch_loss.py` | 读取P1已有epoch和mini-batch CSV重新绘图，不训练模型。主图使用对数纵轴和10个batch移动平均，局部放大第35至40轮，并标记最佳验证轮次。 | `outputs/mlp_preprocessing_comparison/P1_subject_body_minibatch_loss_improved.png` |
| `final_predict_mlp.py` | 在全部7368条有标签数据上执行P1清洗、重新拟合20000维TF-IDF、训练MLP 39轮，再预测2457条无标签邮件。独立运行，不需要旧joblib模型。 | 项目目录 `predictions.csv` |

## 4. 完整复现顺序和前置依赖

在正确环境和工作目录中依次执行，每条命令完成后再运行下一条：

```bat
python draft_main.py
python svm_experiment.py

python mlp_tuning_round2_stage1.py
python mlp_tuning_round2_stage2.py
python mlp_tuning_round2_stage2_extension.py
python mlp_tuning_round2_stage3.py
python mlp_tuning_round2_stage4.py
python mlp_tuning_round2_stage4_extension.py

python svm_tuning_round2_stage1.py
python svm_tuning_round2_stage2.py
python svm_tuning_round2_stage2_extension.py

python mlp_preprocessing_experiment.py
python svm_preprocessing_p1_experiment.py
python repeated_seed_experiment.py
python replot_p1_minibatch_loss.py

python final_predict_mlp.py
```

必须满足的依赖：

| 要运行的脚本 | 前置条件 |
|---|---|
| M2正则扩展 | 先运行原M2，生成 `outputs/mlp_tuning_round2/stage2_alpha/mlp_stage2_alpha_results.csv` 和同目录 `mlp_stage2_best.joblib`，两者都需要。 |
| MLP维度扩展 | 保留同目录的原M4脚本，并先生成 `outputs/mlp_tuning_round2/stage4_tfidf_dimension/mlp_stage4_dimension_results.csv`。 |
| SVM维度扩展 | 保留同目录的原S2脚本，并先生成 `outputs/svm_tuning_round2/stage2_tfidf_dimension/svm_stage2_dimension_results.csv`。 |
| SVM清洗对照 | 保留同目录 `mlp_preprocessing_experiment.py` 供导入函数；不要求已完成MLP清洗训练。 |
| 损失图重绘 | 先完成MLP清洗实验，或备齐 `outputs/mlp_preprocessing_comparison/P1_subject_body_history.csv` 和 `P1_subject_body_minibatch_history.csv`。 |

只想生成最终预测时，在方案已确定后直接运行 `python final_predict_mlp.py`，无需重跑全部调参。只想重绘报告图时，备齐两个P1记录CSV后运行绘图脚本即可。

## 5. 三随机种子实验的解释

固定P1、20000维TF-IDF、MLP隐藏层100、alpha=0.01、学习率0.0005、batch=128、39轮，以及SVM线性核、C=500。

每个种子重新进行8:2分层划分，TF-IDF只在该次训练子集上拟合；同一次种子下MLP和SVM使用相同的特征与验证子集。种子同时影响划分、MLP初始化及批次打乱，因此该实验检验整个流程的随机敏感性，不能把波动仅归因于划分。实验不读取官方测试集。

运行 `python repeated_seed_experiment.py` 后，在 `outputs/repeated_seeds/` 查看：

- `repeated_seed_results.csv`：每个模型、每个种子的训练/验证Accuracy、Macro-F1及训练时间。
- `repeated_seed_summary.csv`：验证Accuracy、Macro-F1的均值和样本标准差，以及训练时间的均值。

## 6. 最终预测和格式核对

最终配置：P1、20000维unigram TF-IDF、隐藏层 `(100,)`、ReLU、Adam、alpha=0.01、学习率0.0005、batch=128、随机种子42、39轮。

39轮来自P1验证实验的最佳轮次，是全量重训前确定的预算；全量重训没有验证集，不会重新搜索“这次最佳轮次”。三随机种子验证也固定39轮。

```bat
python final_predict_mlp.py
```

脚本在全部有标签文本上拟合TF-IDF并训练，对测试文本执行同样的P1处理后仅调用 `transform` 和预测。`predictions.csv` 为单列、无表头、2457行，每行一个0至9整数，顺序与测试CSV一致。官方测试集没有真实标签，无法计算Accuracy或Macro-F1，报告中的指标来自有标签数据的验证集。

如需另存预测：

```bat
python final_predict_mlp.py --output predictions_reproduced.csv
```

## 7. 结果查阅和提交

结果写入 `outputs/` 下各实验目录；重跑同一个脚本会重新写入其同名输出。绘图使用非交互后端，图片保存为文件，不弹出窗口。

| 实验 | 汇总文件（相对 `outputs/`） |
|---|---|
| 初始MLP | `mlp_comparison/mlp_comparison_results.csv` |
| 初始SVM | `svm_comparison/svm_comparison_results.csv` |
| MLP宽度 | `mlp_tuning_round2/stage1_width/mlp_stage1_width_results.csv` |
| MLP原6档正则对照 | `mlp_tuning_round2/stage2_alpha/mlp_stage2_alpha_results.csv` |
| MLP正则补充（0.01/0.03/0.1） | `mlp_tuning_round2/stage2_alpha_extension/mlp_stage2_extended_results.csv` |
| MLP学习率 | `mlp_tuning_round2/stage3_learning_rate/mlp_stage3_learning_rate_results.csv` |
| MLP五档维度 | `mlp_tuning_round2/stage4_tfidf_dimension_extension/mlp_stage4_all_dimension_results.csv` |
| SVM扩大C搜索 | `svm_tuning_round2/stage1_c/svm_stage1_c_results.csv` |
| SVM五档维度 | `svm_tuning_round2/stage2_tfidf_dimension_extension/svm_stage2_all_dimension_results.csv` |
| MLP清洗 | `mlp_preprocessing_comparison/mlp_preprocessing_results.csv` |
| SVM清洗 | `svm_preprocessing_p1/svm_p1_preprocessing_results.csv` |
| 三随机种子 | `repeated_seeds/repeated_seed_results.csv`、同目录 `repeated_seed_summary.csv` |

MLP记录中的 `best_epoch` 按验证Macro-F1选择，`minimum_val_loss_epoch` 按验证交叉熵选择，二者不一定相同。mini-batch CSV记录每次更新后当前batch的交叉熵，不含L2正则项；训练子集每轮47个batch，46个含128条样本，末批含6条样本，因此样本内容和批次大小都可能影响波动。

提交代码时保留本README和第3节列出的全部16个Python脚本，覆盖报告中现有实验。数据按第2节放置，同时提交最终报告和 `predictions.csv`。可附汇总CSV、配置JSON及P1绘图记录，便于核对结果。

`__pycache__`、`.pyc`、旧版报告和版式检查临时文件无需提交。joblib可按顺序运行重新生成；若仅附CSV而不附模型，M2扩展仍需先运行原M2。`TUNING.md`是原始调参参考资料，不是本项目运行入口。

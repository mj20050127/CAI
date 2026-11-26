# 实验三 CIFAR-10 图像分类运行说明

本项目从六层 CNN 开始，比较 CIFAR ResNet-18、数据增强、训练策略、ImageNet 迁移和 CBAM。最终模型是 **E11_stem**，即采用首层适配和 CutMix 的 32×32 CIFAR ResNet-18。

已经有 `best.pt` 时，可以直接准备数据并运行第 2 节的评估命令。需要从头训练时，按第 3、4 节的顺序执行。

## 1 环境与数据准备

以下命令在 **Windows PowerShell** 中执行，工作目录是包含 `README.md`、`train.py` 和 `configs/` 的项目根目录。在克隆后的仓库根目录打开 PowerShell。

```powershell
$python = '.\.venv\Scripts\python.exe'
```

### 安装依赖

本实验的环境为 Python 3.11.9、PyTorch 2.8.0+cu126、torchvision 0.23.0+cu126 和 RTX 4060 Laptop GPU。使用项目已配好的 `.venv` 时，直接运行下面的环境检查。首次搭建环境可以用 Python 3.11 和 [依赖锁定文件](requirements-lock.txt)。

```powershell
py -3.11 -m venv .venv
& $python -m pip --isolated install -r requirements-lock.txt
& $python -m pip check
& $python check_environment.py
```

GPU 检查通过时会显示 `CUDA available: True`、显卡名称和模型输出形状 `(2, 10)`。CPU 环境使用 `check_environment.py --cpu`，训练或评估命令则使用 `--device cpu`。正式训练的配置和 GPU 环境记录保存在各运行目录下。

### 准备 CIFAR-10

```powershell
& $python prepare_data.py --config configs/e0.json --download
```

首次运行由 torchvision 下载官方 CIFAR-10 Python 版数据并校验。若下载较慢，可从 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html) 下载 `cifar-10-python.tar.gz`，放进 `data/` 后再执行该命令。数据已经解压时，可以省略 `--download`。

| 数据 | 样本数 | 每类样本数 | 用途 |
|---|---:|---:|---|
| 训练集 | 45,000 | 4,500 | 更新模型参数 |
| 验证集 | 5,000 | 500 | 比较方案和选最佳轮次 |
| 训练监测子集 | 从训练集固定抽 5,000 | 500 | 用原始标签和无随机增强的图像观察拟合 |
| 官方测试集 | 10,000 | 1,000 | 评估预先选定的权重 |

终端应显示 `train=45000`、`validation=5000`、`monitor=5000`。索引、类别顺序、数据哈希和训练图的通道均值、标准差保存在 `data/split_manifest.json`。划分种子为 2026，监测抽样种子为 2027。复验只改变训练种子，所有模型继续用同一批验证图。

32×32 组默认用补边 4 像素的随机裁剪和水平翻转。E2 关闭这两项增强。E9、E10 使用 224×224 标准 ResNet-18，训练时缩放增强后的图片，验证和测试时先缩放至 256，再中心裁剪至 224，并使用 ImageNet 归一化。

## 2 使用现成权重复评

### 快速复核主模型

仓库内提供已归档的评估输出。运行复评命令前，按第 4 节训练模型，或准备相同的最佳权重。

使用训练时保存的 `best.pt`。这个文件按验证准确率选择，准确率并列时取验证交叉熵更低的一轮。

```powershell
$recheck = 'results/recheck/' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$stem = 'runs/E11_stem_lr0.01/seed42'

& $python -u evaluate.py --checkpoint "$stem/best.pt" --split validation --data-dir data --output-dir "$recheck/E11_stem_seed42_validation"
& $python -u evaluate.py --checkpoint "$stem/best.pt" --split test --data-dir data --output-dir "$recheck/E11_stem_seed42_test"
```

同样的数据和权重下，验证集应输出 `n=5000`、`accuracy=96.70%`，官方测试集应输出 `n=10000`、`accuracy=96.23%`。`--data-dir data` 指向当前机器的数据，适合评估拷贝来的权重。脚本会核对数据清单哈希。CPU 复评可在对应命令中加 `--device cpu`。

显式指定的 `--output-dir` 应为空或尚未创建。省略该参数时，评估会写入权重旁边的 `evaluation_validation_epoch轮次/` 或 `evaluation_test_epoch轮次/`。重复评估同一权重和划分时，默认目录中的同名结果会更新。

### 复核报告中的四组官方测试

```powershell
$testRuns = @{
    E0_seed42       = 'runs/E0/seed42'
    E11_stem_seed42 = 'runs/E11_stem_lr0.01/seed42'
    E11_stem_seed43 = 'runs/E11_stem_lr0.01/seed43'
    E11_stem_seed44 = 'runs/E11_stem_lr0.01/seed44'
}
foreach ($label in $testRuns.Keys | Sort-Object) {
    $run = $testRuns[$label]
    & $python -u evaluate.py --checkpoint "$run/best.pt" --split test --data-dir data --output-dir "$recheck/$label"
    if ($LASTEXITCODE -ne 0) { throw "$label 评估失败" }
}
```

| 模型 | 种子 | 最佳轮次 | 验证准确率 | 官方测试准确率 | 归档结果 |
|---|---:|---:|---:|---:|---|
| E0 | 42 | 100 | 93.46% | 92.84% | [测试指标](results/official_test/E0_seed42/metrics.json) |
| E11_stem | 42 | 90 | 96.70% | 96.23% | [测试指标](results/official_test/E11_stem_seed42/metrics.json) |
| E11_stem | 43 | 80 | 96.76% | 96.46% | [测试指标](results/official_test/E11_stem_seed43/metrics.json) |
| E11_stem | 44 | 94 | 96.84% | 96.33% | [测试指标](results/official_test/E11_stem_seed44/metrics.json) |

E11_stem 三个种子的平均最佳验证准确率为 **96.77%**，样本标准差 **0.07 个百分点**。官方测试均值为 **96.34%**，样本标准差 **0.12 个百分点**。报告以种子 42 展示逐类结果和混淆矩阵，和 E0 在同种子下比较。

| 评估输出 | 用途 |
|---|---|
| `metrics.json` | 样本数、总体准确率、Macro-F1、交叉熵、最佳权重轮次 |
| `per_class.csv` | 每类样本数、判对数、准确率、精确率、召回率和 F1 |
| `predictions.csv` | 每张图的索引、真实类别、预测类别和置信度 |
| `confusion_matrix.csv`、`confusion_counts.png`、`confusion_normalized.png` | 真实类别为行，预测类别为列 |
| `errors.png` | 按置信度选取的错分样例 |

官方测试预测文件应有 10,000 条数据，混淆矩阵的计数之和也应为 10,000。

## 3 训练、续训与查看曲线

### 小规模流程检查

```powershell
& $python -u train.py --config configs/e0.json --smoke --run-name smoke_readme
& $python plot_history.py --run-dir runs/E0/smoke_readme
```

`--smoke` 默认跑 2 轮，抽取 1,000 张训练图、500 张验证图和 500 张监测图。它用于检查前后向、日志、保存和绘图。若 `smoke_readme` 已存在，给这次运行另取一个名字，并同步修改绘图路径。

### 正式数据分段运行

下面用 E0 演示。`--stop-after 3` 表示本次只跑三轮，仍使用配置里的 100 轮余弦学习率计划。`--resume` 使用同目录的 `last.pt` 从下一轮继续。

```powershell
& $python -u train.py --config configs/e0.json --run-name e0_trial --stop-after 3
& $python plot_history.py --run-dir runs/E0/e0_trial
& $python -u train.py --config configs/e0.json --run-name e0_trial --resume
```

正式运行默认目录名由配置里的训练种子决定，如 `seed42`。另开运行用 `--run-name`。续训时沿用原配置、运行名、设备和 DataLoader 参数。`--epochs` 会修改总轮数和调度长度，想先跑几轮看趋势时用 `--stop-after`。完成整轮后会保存 `last.pt`，中断在一轮中途时从上一个完整断点恢复。

| 运行产物 | 内容 |
|---|---|
| `initial.pt` | 第 0 轮初始状态，供配对实验复用 |
| `best.pt`、`last.pt` | 验证最佳权重与最近完整轮的续训状态 |
| `history.csv`、`summary.json` | 逐轮指标、耗时、完成状态和最佳轮次 |
| `config.json`、`data_manifest.json`、`environment.json` | 实际配置、数据划分和运行环境 |
| `model.txt`、`source_snapshot/` | 模型结构、参数量和代码快照 |
| `mixup_batches.csv`、`cutmix_batches.csv` | 启用对应增强时记录的逐批参数 |
| `curves.png` | 运行 `plot_history.py` 后生成的损失、准确率与学习率曲线 |

`history.csv` 中的准确率用 0 到 1 表示。`train_objective` 来自实际训练输入和目标，MixUp、CutMix 使用混合标签。比较不同增强方法时，优先看使用真实标签计算的 `val_loss` 和 `val_accuracy`。`epoch_seconds` 包含每轮训练、监测和验证时间，保存权重的时间另计。

## 4 从头复现各组实验

下面的训练命令适用于 **一份新的工作副本**。本目录已有报告对应的运行结果，查看或复评可直接用现有权重。重建各组时按顺序逐条执行，确认依赖组已写出 `initial.pt`。运行名决定初始权重的路径，尤其要保留 E4 的 `seed42_retry1`。

### 种子 42 的完整对照

```powershell
& $python -u train.py --config configs/e0.json
& $python -u train.py --config configs/e1.json

& $python -u train.py --config configs/e2.json
& $python -u train.py --config configs/e3.json
& $python -u train.py --config configs/e4.json --run-name seed42_retry1
& $python -u train.py --config configs/e5.json
& $python -u train.py --config configs/e6.json --run-name seed42_retry1
& $python -u train.py --config configs/e7.json
& $python -u train.py --config configs/e8.json

& $python -u train.py --config configs/e9.json
& $python -u train.py --config configs/e10.json

& $python -u train.py --config configs/e11.json --run-name seed42_retry1
& $python -u train.py --config configs/e11_stem.json
& $python -u train.py --config configs/e11_stem_cbam.json
```

| 实验 | 主要设置 | 正式目录 | 最佳验证准确率 |
|---|---|---|---:|
| E0 | 六层 CNN | `runs/E0/seed42` | 93.46% |
| E1 | CIFAR ResNet-18 基线 | `runs/E1/seed42` | 95.30% |
| E2 | 关闭裁剪和翻转 | `runs/E2/seed42` | 89.52% |
| E3 | MixUp，`alpha=0.2` | `runs/E3/seed42` | 95.62% |
| E4 | CutMix，`alpha=1.0` | `runs/E4/seed42_retry1` | 96.30% |
| E5 | E1 加 8 个 CBAM 模块 | `runs/E5/seed42` | 95.38% |
| E6 | 固定学习率 `0.1` | `runs/E6/seed42_retry1` | 87.34% |
| E7 | 余弦衰减与验证集早停 | `runs/E7/seed42` | 95.30% |
| E8 | 交叉熵加标签平滑 `0.1` | `runs/E8/seed42` | 95.40% |
| E9 | 224×224 标准 ResNet-18，随机初始化 | `runs/E9/seed42` | 92.00% |
| E10 | 224×224 标准 ResNet-18，ImageNet 骨干初始化 | `runs/E10/seed42` | 96.12% |
| E11 | 32×32，仅迁移 `layer1` 至 `layer4`，学习率 `0.1` | `runs/E11/seed42_retry1` | 95.50% |
| E11_stem | 额外迁移入口，学习率 `0.01` | `runs/E11_stem_lr0.01/seed42` | 96.70% |
| E11_stem_cbam | E11_stem 加 8 个 CBAM 模块 | `runs/E11_stem_cbam_lr0.01/seed42` | 96.64% |

E2–E8 读取 E1 第 0 轮共有层权重。E10 复用 E9 第 0 轮分类头。E11 和 E11_stem 读取 `runs/E4/seed42_retry1/initial.pt`，E11_stem_cbam 读取 E11_stem 的第 0 轮权重。预训练模型使用 `ResNet18_Weights.IMAGENET1K_V1`，首次训练时会检查 `data/weights/resnet18-f37072fd.pth`，缺少缓存时从官方地址下载。

32×32 分支一般为 100 轮、批大小 128、SGD 和余弦衰减。E6 固定学习率，E11_stem 与其 CBAM 组从 `0.01` 起步。E9、E10 使用 30 轮、批大小 32 和 AdamW。完整设置见 [配置目录](configs)。

### E11_stem 种子 43 和 44

复验配置位于 [多种子配置目录](results/multiseed_43_44/configs)。训练种子写在 JSON 的 `seed` 字段中。每个种子按 **E1 初始权重 → E4 初始权重 → E11_stem 正式训练** 的顺序准备。E1、E4 已有同种子的 `initial.pt` 时可直接复用。新副本需要生成它们时，按以下顺序执行。

```powershell
$seedConfigs = 'results/multiseed_43_44/configs'

& $python -u train.py --config "$seedConfigs/e1_seed43.json" --stop-after 1
& $python -u train.py --config "$seedConfigs/e4_seed43.json" --stop-after 1
& $python -u train.py --config "$seedConfigs/e11_stem_seed43.json"

& $python -u train.py --config "$seedConfigs/e1_seed44.json" --stop-after 1
& $python -u train.py --config "$seedConfigs/e4_seed44.json" --stop-after 1
& $python -u train.py --config "$seedConfigs/e11_stem_seed44.json"
```

E1、E4 的一轮准备命令保留 100 轮计划，正式成绩取 E11_stem 完整训练的 `summary.json`。种子 43、44 的 E11_stem 已各跑满 100 轮，见 [种子 43 训练摘要](results/multiseed_43_44/E11_stem_seed43_summary.json) 和 [种子 44 训练摘要](results/multiseed_43_44/E11_stem_seed44_summary.json)。

## 5 生成对照图和推理计时

`compare_runs.py` 从两个运行目录读取完成的训练记录与最佳轮的验证预测。如果新训练的运行还没有验证预测，先对两组执行默认目录的验证评估。下面以 E1/E4 为例。

```powershell
& $python -u evaluate.py --checkpoint runs/E1/seed42/best.pt --split validation --data-dir data
& $python -u evaluate.py --checkpoint runs/E4/seed42_retry1/best.pt --split validation --data-dir data
& $python compare_runs.py --run-dirs runs/E1/seed42 runs/E4/seed42_retry1 --output-dir "$recheck/E1_vs_E4"
```

输出包括 `comparison.json`、逐类与逐图对照 CSV、`curves.png`。输入顺序为基线在前，实验组在后。已经有两组默认验证子目录时，可以直接运行第三条。采用第 2 节的独立输出目录复核时，结果保存在指定位置，正式对照读取的仍是运行目录下的验证子目录。

| 对照 | 归档结果 |
|---|---|
| E0/E1 网络结构 | [分析](results/E0_vs_E1/analysis.md) |
| E1/E2、E1/E3、E1/E4 数据增强 | [基础增强](results/E1_vs_E2/analysis.md) · [MixUp](results/E1_vs_E3/analysis.md) · [CutMix](results/E1_vs_E4/analysis.md) |
| E1/E5、E1/E6、E1/E7、E1/E8 | [CBAM](results/E1_vs_E5/analysis.md) · [学习率](results/E1_vs_E6/analysis.md) · [早停](results/E1_vs_E7/analysis.md) · [标签平滑](results/E1_vs_E8/analysis.md) |
| E9/E10、E4/E11 迁移 | [224×224 完整迁移](results/E9_vs_E10/comparison.json) · [32×32 部分迁移](results/E4_vs_E11/comparison.json) |
| E11_stem 与 CBAM | [预训练加 CutMix 条件下的注意力对照](results/E11_stem_vs_E11_stem_cbam/comparison.json) |

`compare_runs.py` 会检查种子、划分和训练协议。E4 与正式 E11_stem 的初始学习率分别为 `0.1` 和 `0.01`，报告把这组写为两种训练方案的比较。三种子结果从各自的摘要和官方测试指标汇总。

推理计时使用相同输入尺寸的两个模型。下面测量单张和 128 张整批前向时间，显式指定输出文件，便于保留原计时结果。

```powershell
& $python benchmark_models.py --checkpoints runs/E11_stem_lr0.01/seed42/best.pt runs/E11_stem_cbam_lr0.01/seed42/best.pt --batch-sizes 1 128 --output "$recheck/E11_stem_vs_CBAM_benchmark.json"
```

计时使用本机同设备的 FP32 合成输入，记录模型前向，含预热与多次重复。结果包含批延迟、吞吐量和每次计时。已有记录见 [E0/E1](results/E0_vs_E1/inference_benchmark.json)、[E9/E10](results/E9_vs_E10/inference_benchmark.json) 和 [E11_stem/CBAM](results/E11_stem_vs_E11_stem_cbam/inference_benchmark.json)。

## 6 增强示例和可解释性图

生成新的增强示例时，可以指定独立目录。预览种子只控制展示抽样，训练种子在配置文件中设置。

```powershell
& $python preview_mixup.py --config configs/e3.json --output-dir "$recheck/mixup_preview"
& $python preview_cutmix.py --config configs/e4.json --output-dir "$recheck/cutmix_preview"
```

已生成 [MixUp 示例](results/E3_mixup_preview/mixup_examples.png)、[CutMix 示例](results/E4_cutmix_preview/cutmix_examples.png)、[猫类别 Grad-CAM](results/explainability_seed42/grad_cam_layer3_cat.png)、[狗类别 Grad-CAM](results/explainability_seed42/grad_cam_layer3_dog.png)、[十类案例](results/explainability_seed42/grad_cam_by_class.png) 和 [三层特征图](results/explainability_seed42/feature_maps.png)。

Grad-CAM 和特征图由已保存权重离线生成，复核所需的权重路径、样本索引、目标类别、层名和选通道规则见 [可解释性记录](results/explainability_seed42/provenance.json)。Grad-CAM 用目标类别分数对 `layer3` 或 `layer4` 的特征求梯度，在空间维度取平均作为通道权重，再将加权特征的正值放大到原图大小。特征图在 `layer1`、`layer3`、`layer4` 取同一输入和通道编号，每层使用统一色标。

## 7 常见问题与文件入口

| 情况 | 处理办法 |
|---|---|
| 提示找不到 `split_manifest.json` | 先运行第 1 节的数据准备命令 |
| 输出目录已存在 | 续训时加 `--resume`。新实验另取 `--run-name` |
| 中断后有 `initial.pt`、没有 `last.pt` | `initial.pt` 可供同种子下游模型初始化。如要重跑这组，另取运行名并按需要更新下游配置中的来源路径 |
| 移动项目后权重里的数据路径失效 | 评估时加 `--data-dir data`。原路径发生变化后，重训可在新副本使用原始配置 |
| 启动时停在加载训练依赖 | 等待 PyTorch 导入，随后应看到数据清单校验和 `Epoch`。若终端显示 `KeyboardInterrupt`，先检查是否已生成完整断点 |
| 对照脚本提示缺少验证结果 | 在两组最佳权重旁按默认路径生成验证预测，再执行对照 |
| 复评显存不足或没有 CUDA | 添加 `--device cpu`。续训时沿用原运行的设备设置 |

| 入口 | 内容 |
|---|---|
| [train.py](train.py) · [evaluate.py](evaluate.py) | 训练、续训、验证和测试 |
| [prepare_data.py](prepare_data.py) · [src/data.py](src/data.py) | 固定数据划分、归一化和 DataLoader |
| [src/models](src/models) · [src/engine.py](src/engine.py) | CNN、ResNet、CBAM 与训练循环 |
| [compare_runs.py](compare_runs.py) · [benchmark_models.py](benchmark_models.py) | 配对分析与推理计时 |
| [文档目录](docs) · [实验实施计划](docs/实验三完整实施计划.md) · [E4 操作说明](docs/E4操作说明.md) | 实验方案与分项操作记录 |

运行代码检查可用：

```powershell
& $python -X utf8 -m unittest discover -s tests -v
```

`.gitignore` 排除了 `.venv/`、`data/` 和 `runs/`。如果通过 Git 获取源码，可按第 1 节重建环境和数据。若要直接复评报告，另需对应运行目录中的 `best.pt`，以及同一份数据划分。

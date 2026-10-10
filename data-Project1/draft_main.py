# my_experiment.py

# 1. 导入依赖并设置实验参数
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# 使用非交互式绘图后端，确保在 Anaconda Prompt 中也能直接保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
MLP_EPOCHS = 50
MINI_BATCH_SIZE = 128
OUTPUT_DIR = Path("outputs")

CLASS_NAMES = {
    0: "comp.graphics",
    1: "comp.os.ms-windows.misc",
    2: "comp.sys.ibm.pc.hardware",
    3: "rec.autos",
    4: "rec.motorcycles",
    5: "rec.sport.baseball",
    6: "sci.crypt",
    7: "sci.electronics",
    8: "sci.med",
    9: "talk.politics.mideast",
}

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_all = train_df['text'].astype(str).tolist()
    y_all = train_df['target'].values
    X_test_unlabeled = test_df['text'].astype(str).tolist()
    return X_all, y_all, X_test_unlabeled

# 2. 加载全部有标签数据和官方无标签测试集
#    这个函数会自动读取 .csv 文件并返回你需要的所有内容
X_all, y_all, X_test_unlabeled = load_data()

# 3. (验证步骤) 检查一下数据是否加载成功
print("--- 数据加载成功 ---")
print(f"全部有标签样本数量: {len(X_all)}")
print(f"全部标签数量: {len(y_all)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_all[0])
print(f"\n第一个训练样本的标签: {y_all[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# 4. 将有标签数据按 8:2 分层划分为训练集和验证集
# stratify=y_all 保证训练集和验证集中的类别比例与原数据基本一致。
# random_state 固定随机划分，使后续模型比较可以复现。
X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all,
)

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_train) + len(X_val) == len(X_all)

print("--- 数据划分完成 ---")
print(f"训练集样本数量: {len(X_train)}")
print(f"验证集样本数量: {len(X_val)}")
print("\n训练集各类别数量:")
print(pd.Series(y_train).value_counts().sort_index().to_string())
print("\n验证集各类别数量:")
print(pd.Series(y_val).value_counts().sort_index().to_string())
print("=" * 50)

# 5. TF-IDF 特征提取
# 只在训练集上拟合词表和 IDF，避免验证集、测试集信息泄露。
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    lowercase=True,
    ngram_range=(1, 1),
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test_unlabeled)

# 检查三个数据集是否使用了完全相同的特征空间。
vocabulary_size = len(vectorizer.vocabulary_)
assert X_train_tfidf.shape == (len(X_train), vocabulary_size)
assert X_val_tfidf.shape == (len(X_val), vocabulary_size)
assert X_test_tfidf.shape == (len(X_test_unlabeled), vocabulary_size)
assert vocabulary_size <= vectorizer.max_features

train_density = X_train_tfidf.nnz / (
    X_train_tfidf.shape[0] * X_train_tfidf.shape[1]
)

print("--- TF-IDF 特征提取完成 ---")
print(f"词表大小: {vocabulary_size}")
print(f"训练集特征矩阵形状: {X_train_tfidf.shape}")
print(f"验证集特征矩阵形状: {X_val_tfidf.shape}")
print(f"测试集特征矩阵形状: {X_test_tfidf.shape}")
print(f"训练集特征矩阵非零元素数量: {X_train_tfidf.nnz}")
print(f"训练集特征矩阵密度: {train_density:.4%}")
print("=" * 50)

# 6. MLP 参数对照实验
# 其他训练条件完全一致，只改变隐藏层结构与 L2 正则化系数 alpha。
MLP_CONFIGS = [
    {"name": "h50_a0p0001", "hidden_layer_sizes": (50,), "alpha": 0.0001},
    {"name": "h100_a0p0001", "hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"name": "h100x50_a0p0001", "hidden_layer_sizes": (100, 50), "alpha": 0.0001},
    {"name": "h50_a0p001", "hidden_layer_sizes": (50,), "alpha": 0.001},
    {"name": "h100_a0p001", "hidden_layer_sizes": (100,), "alpha": 0.001},
    {"name": "h100x50_a0p001", "hidden_layer_sizes": (100, 50), "alpha": 0.001},
]

comparison_dir = OUTPUT_DIR / "mlp_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)
classes = pd.Series(y_train).sort_values().unique()
comparison_results = []

overall_best_model = None
overall_best_config = None
overall_best_epoch = 0
overall_best_val_macro_f1 = -1.0

for config_number, config in enumerate(MLP_CONFIGS, start=1):
    print("=" * 70)
    print(
        f"--- MLP 配置 {config_number}/{len(MLP_CONFIGS)}: {config['name']} | "
        f"hidden_layers={config['hidden_layer_sizes']} | alpha={config['alpha']} ---"
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=config["alpha"],
        # 外层循环已经显式划分 mini-batch；auto 会把传入的每批作为一个批次。
        batch_size="auto",
        learning_rate_init=0.001,
        random_state=RANDOM_STATE,
        shuffle=False,
    )

    history = []
    mini_batch_history = []
    config_best_model = None
    config_best_epoch = 0
    config_best_val_macro_f1 = -1.0
    start_time = perf_counter()
    cumulative_fit_seconds = 0.0
    random_generator = np.random.RandomState(RANDOM_STATE)
    update_step = 0
    first_update = True

    for epoch in range(1, MLP_EPOCHS + 1):
        # 每组配置使用相同种子生成批次顺序，确保参数对照公平。
        shuffled_indices = random_generator.permutation(len(y_train))
        for batch_number, start_index in enumerate(
            range(0, len(y_train), MINI_BATCH_SIZE),
            start=1,
        ):
            batch_indices = shuffled_indices[
                start_index : start_index + MINI_BATCH_SIZE
            ]
            X_batch = X_train_tfidf[batch_indices]
            y_batch = y_train[batch_indices]

            fit_update_start = perf_counter()
            if first_update:
                mlp_model.partial_fit(X_batch, y_batch, classes=classes)
                first_update = False
            else:
                mlp_model.partial_fit(X_batch, y_batch)
            cumulative_fit_seconds += perf_counter() - fit_update_start

            # 记录本批参数更新后的交叉熵；它保留随机批次带来的局部波动。
            batch_probabilities = mlp_model.predict_proba(X_batch)
            batch_loss = log_loss(y_batch, batch_probabilities, labels=classes)
            update_step += 1
            mini_batch_history.append(
                {
                    "update_step": update_step,
                    "epoch": epoch,
                    "batch": batch_number,
                    "batch_size": len(batch_indices),
                    "batch_loss": batch_loss,
                }
            )

        # 两个损失均为在完整数据子集上计算的平均多分类交叉熵。
        train_probabilities = mlp_model.predict_proba(X_train_tfidf)
        val_probabilities = mlp_model.predict_proba(X_val_tfidf)
        train_loss = log_loss(y_train, train_probabilities, labels=classes)
        val_loss = log_loss(y_val, val_probabilities, labels=classes)

        train_predictions = mlp_model.classes_[train_probabilities.argmax(axis=1)]
        val_predictions = mlp_model.classes_[val_probabilities.argmax(axis=1)]
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_macro_f1 = f1_score(y_train, train_predictions, average="macro")
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_macro_f1 = f1_score(y_val, val_predictions, average="macro")

        history.append(
            {
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "train_macro_f1": train_macro_f1,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss_type": "cross_entropy",
                "cumulative_fit_seconds": cumulative_fit_seconds,
            }
        )

        # Macro-F1 相同时保留更早的 epoch，使选择规则确定且可复现。
        if val_macro_f1 > config_best_val_macro_f1:
            config_best_val_macro_f1 = val_macro_f1
            config_best_epoch = epoch
            config_best_model = deepcopy(mlp_model)

        print(
            f"Epoch {epoch:03d}/{MLP_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_accuracy={train_accuracy:.4f} | "
            f"val_accuracy={val_accuracy:.4f} | "
            f"train_macro_f1={train_macro_f1:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

    runtime_seconds = perf_counter() - start_time
    history_df = pd.DataFrame(history)
    history_path = comparison_dir / f"{config['name']}_history.csv"
    history_df.to_csv(history_path, index=False)
    mini_batch_history_df = pd.DataFrame(mini_batch_history)
    mini_batch_history_path = comparison_dir / f"{config['name']}_minibatch_history.csv"
    mini_batch_history_df.to_csv(mini_batch_history_path, index=False)

    best_row = history_df.loc[history_df["epoch"] == config_best_epoch].iloc[0]
    minimum_loss_row = history_df.loc[history_df["val_loss"].idxmin()]
    trainable_parameters = sum(array.size for array in config_best_model.coefs_)
    trainable_parameters += sum(array.size for array in config_best_model.intercepts_)
    comparison_results.append(
        {
            "algorithm": "MLP",
            "config_order": config_number,
            "config": config["name"],
            "hidden_layer_sizes": str(config["hidden_layer_sizes"]),
            "alpha": config["alpha"],
            "best_epoch": config_best_epoch,
            "train_accuracy": best_row["train_accuracy"],
            "train_macro_f1": best_row["train_macro_f1"],
            "val_accuracy": best_row["val_accuracy"],
            "val_macro_f1": best_row["val_macro_f1"],
            "selection_metric": "val_macro_f1",
            "loss_type": "cross_entropy",
            "train_loss": best_row["train_loss"],
            "val_loss": best_row["val_loss"],
            "minimum_val_loss_epoch": int(minimum_loss_row["epoch"]),
            "minimum_val_loss": minimum_loss_row["val_loss"],
            "fit_seconds": best_row["cumulative_fit_seconds"],
            "runtime_seconds": runtime_seconds,
            "complexity_name": "trainable_parameters",
            "complexity_value": trainable_parameters,
        }
    )

    # 每个配置单独保存一张训练/验证损失曲线，避免多条曲线相互遮挡。
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history_df["epoch"], history_df["train_loss"], label="Training loss")
    axis.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    axis.axvline(
        config_best_epoch,
        color="tab:red",
        linestyle="--",
        label=f"Best Macro-F1 epoch ({config_best_epoch})",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title(
        f"MLP Loss: hidden={config['hidden_layer_sizes']}, alpha={config['alpha']}"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        comparison_dir / f"{config['name']}_loss.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    # 原始 mini-batch 损失显示局部波动；一个 epoch 的移动平均显示整体趋势。
    batches_per_epoch = int(np.ceil(len(y_train) / MINI_BATCH_SIZE))
    rolling_batch_loss = mini_batch_history_df["batch_loss"].rolling(
        window=batches_per_epoch,
        min_periods=1,
    ).mean()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        mini_batch_history_df["update_step"],
        mini_batch_history_df["batch_loss"],
        color="tab:gray",
        alpha=0.45,
        linewidth=0.8,
        label="Raw mini-batch loss",
    )
    axis.plot(
        mini_batch_history_df["update_step"],
        rolling_batch_loss,
        color="tab:blue",
        linewidth=1.8,
        label=f"Moving average ({batches_per_epoch} batches)",
    )
    axis.set_xlabel("Parameter update step")
    axis.set_ylabel("Post-update mini-batch cross-entropy")
    axis.set_title(
        f"MLP Mini-batch Loss: hidden={config['hidden_layer_sizes']}, "
        f"alpha={config['alpha']}"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        comparison_dir / f"{config['name']}_minibatch_loss.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    if config_best_val_macro_f1 > overall_best_val_macro_f1:
        overall_best_val_macro_f1 = config_best_val_macro_f1
        overall_best_epoch = config_best_epoch
        overall_best_config = config.copy()
        overall_best_model = deepcopy(config_best_model)

# 7. 保存统一的参数对照结果
comparison_df = pd.DataFrame(comparison_results).sort_values(
    by=["val_macro_f1", "config_order"],
    ascending=[False, True],
    ignore_index=True,
)
comparison_path = comparison_dir / "mlp_comparison_results.csv"
comparison_df.to_csv(comparison_path, index=False)

# 绘制各配置最佳 Macro-F1 对比图。
figure, axis = plt.subplots(figsize=(10, 5))
axis.bar(comparison_df["config"], comparison_df["val_macro_f1"])
axis.set_xlabel("MLP configuration")
axis.set_ylabel("Best validation Macro-F1")
axis.set_title("MLP Hyperparameter Comparison")
axis.set_ylim(max(0.0, comparison_df["val_macro_f1"].min() - 0.02), 1.0)
axis.tick_params(axis="x", rotation=25)
axis.grid(axis="y", alpha=0.3)
figure.tight_layout()
comparison_figure_path = comparison_dir / "mlp_macro_f1_comparison.png"
figure.savefig(comparison_figure_path, dpi=160, bbox_inches="tight")
plt.close(figure)

# 8. 保存六组实验中验证 Macro-F1 最佳的模型及分类报告
best_val_predictions = overall_best_model.predict(X_val_tfidf)
classification_report_df = pd.DataFrame(
    classification_report(
        y_val,
        best_val_predictions,
        labels=classes,
        target_names=[CLASS_NAMES[label] for label in classes],
        output_dict=True,
        zero_division=0,
    )
).transpose()
classification_report_path = comparison_dir / "mlp_best_classification_report.csv"
classification_report_df.to_csv(classification_report_path)

matrix = confusion_matrix(y_val, best_val_predictions, labels=classes)
matrix_df = pd.DataFrame(matrix, index=classes, columns=classes)
matrix_path = comparison_dir / "mlp_best_confusion_matrix.csv"
matrix_df.to_csv(matrix_path)

figure, axis = plt.subplots(figsize=(8, 7))
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
axis.set_title("Best MLP Validation Confusion Matrix")
figure.tight_layout()
confusion_figure_path = comparison_dir / "mlp_best_confusion_matrix.png"
figure.savefig(confusion_figure_path, dpi=160, bbox_inches="tight")
plt.close(figure)

model_path = comparison_dir / "mlp_best.joblib"
joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": overall_best_model,
        "algorithm": "MLP",
        "configuration": overall_best_config["name"],
        "config": overall_best_config,
        "best_epoch": overall_best_epoch,
        "train_accuracy": float(comparison_df.iloc[0]["train_accuracy"]),
        "train_macro_f1": float(comparison_df.iloc[0]["train_macro_f1"]),
        "validation_accuracy": float(comparison_df.iloc[0]["val_accuracy"]),
        "validation_macro_f1": overall_best_val_macro_f1,
        "selection_metric": "val_macro_f1",
        "loss_type": "cross_entropy",
        "train_loss": float(comparison_df.iloc[0]["train_loss"]),
        "validation_loss": float(comparison_df.iloc[0]["val_loss"]),
        "fit_seconds": float(comparison_df.iloc[0]["fit_seconds"]),
        "trainable_parameters": int(comparison_df.iloc[0]["complexity_value"]),
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "tfidf_config": {
            "max_features": 5000,
            "lowercase": True,
            "ngram_range": (1, 1),
        },
        "class_names": CLASS_NAMES,
    },
    model_path,
)

print("=" * 70)
print("--- MLP 参数对照实验完成 ---")
print(comparison_df.to_string(index=False))
print("\n最佳 MLP 配置:")
print(f"配置名称: {overall_best_config['name']}")
print(f"隐藏层: {overall_best_config['hidden_layer_sizes']}")
print(f"alpha: {overall_best_config['alpha']}")
print(f"最佳 epoch: {overall_best_epoch}")
print(f"验证集 Macro-F1: {overall_best_val_macro_f1:.4f}")
print(f"参数对照表: {comparison_path}")
print(f"参数对比图: {comparison_figure_path}")
print(f"最佳模型分类报告: {classification_report_path}")
print(f"最佳模型混淆矩阵: {confusion_figure_path}")
print(f"最佳模型: {model_path}")

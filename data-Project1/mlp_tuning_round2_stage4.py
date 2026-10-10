"""MLP round-2 stage 4: tune the TF-IDF feature dimension.

This stage fixes the M3-selected network configuration and changes only
``max_features``. Raw email text, the stratified split, training procedure,
and random seeds remain controlled. The official test set is not loaded.
"""

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
BATCH_RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
HIDDEN_LAYER_SIZES = (100,)
ALPHA = 0.01
LEARNING_RATE_INIT = 0.0005
MINI_BATCH_SIZE = 128
MAX_EPOCHS = 80
MAX_FEATURE_VALUES = [5000, 10000, 20000]
PRACTICAL_TIE_TOLERANCE = 0.001
OUTPUT_DIR = Path("outputs") / "mlp_tuning_round2" / "stage4_tfidf_dimension"

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


def load_labeled_data():
    """Load labeled samples only; exclude the official test set."""
    train_df = pd.read_csv("train_data.csv")
    texts = train_df["text"].astype(str).tolist()
    targets = train_df["target"].to_numpy()
    return texts, targets


def plot_epoch_loss(history_df, max_features, best_epoch, output_path):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history_df["epoch"], history_df["train_loss"], label="Training loss")
    axis.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    axis.axvline(
        best_epoch,
        color="tab:red",
        linestyle="--",
        label=f"Best Macro-F1 epoch ({best_epoch})",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title(
        "MLP M4 Loss: "
        f"max_features={max_features}, hidden=(100,), alpha={ALPHA:g}"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_minibatch_loss(minibatch_df, max_features, batches_per_epoch, output_path):
    rolling_loss = minibatch_df["batch_loss"].rolling(
        window=batches_per_epoch,
        min_periods=1,
    ).mean()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        minibatch_df["update_step"],
        minibatch_df["batch_loss"],
        color="tab:gray",
        alpha=0.45,
        linewidth=0.8,
        label="Raw mini-batch loss",
    )
    axis.plot(
        minibatch_df["update_step"],
        rolling_loss,
        color="tab:blue",
        linewidth=1.8,
        label=f"Moving average ({batches_per_epoch} batches)",
    )
    axis.set_xlabel("Parameter update step")
    axis.set_ylabel("Post-update mini-batch cross-entropy")
    axis.set_title(f"MLP M4 Mini-batch Loss: max_features={max_features}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_dimension_comparison(results_df, selected_dimension):
    plot_df = results_df.sort_values("max_features")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(
        plot_df["max_features"],
        plot_df["val_macro_f1"],
        marker="o",
        label="Validation Macro-F1",
    )
    axes[0].plot(
        plot_df["max_features"],
        plot_df["val_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )
    axes[0].axvline(
        selected_dimension,
        color="tab:red",
        linestyle="--",
        label=f"Selected {selected_dimension}",
    )
    axes[0].set_xlabel("TF-IDF max_features")
    axes[0].set_ylabel("Validation score")
    axes[0].set_title("MLP M4: TF-IDF Dimension")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        plot_df["max_features"],
        plot_df["trainable_parameters"],
        marker="o",
        color="tab:purple",
        label="Trainable parameters",
    )
    axes[1].set_xlabel("TF-IDF max_features")
    axes[1].set_ylabel("Trainable parameters")
    axes[1].set_title("MLP M4: Model Size")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mlp_stage4_dimension_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    texts, targets = load_labeled_data()
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        targets,
        test_size=VALIDATION_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=targets,
    )
    classes = np.sort(np.unique(y_train))
    batches_per_epoch = int(np.ceil(len(y_train) / MINI_BATCH_SIZE))

    print("--- MLP 第二轮调参：M4阶段（TF-IDF维度）---")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"max_features 候选值: {MAX_FEATURE_VALUES}")
    print(f"固定隐藏层: {HIDDEN_LAYER_SIZES}")
    print(f"固定 alpha: {ALPHA}")
    print(f"固定 learning_rate_init: {LEARNING_RATE_INIT}")
    print(f"固定 mini-batch size: {MINI_BATCH_SIZE}")
    print(f"最大 epoch: {MAX_EPOCHS}")
    print("选择指标: validation Macro-F1")
    print(f"近似持平阈值: {PRACTICAL_TIE_TOLERANCE}")
    print("官方测试集: 未加载")
    print("=" * 76)

    results = []
    best_models = {}
    fitted_vectorizers = {}

    for config_order, max_features in enumerate(MAX_FEATURE_VALUES, start=1):
        config_name = f"d{max_features}_h100_a0p01_lr0p0005"
        print(
            f"--- 配置 {config_order}/{len(MAX_FEATURE_VALUES)}: "
            f"max_features={max_features} ---"
        )

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            ngram_range=(1, 1),
        )
        vectorize_start = perf_counter()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        vectorize_seconds = perf_counter() - vectorize_start
        actual_feature_count = len(vectorizer.vocabulary_)

        assert X_train_tfidf.shape == (len(X_train), actual_feature_count)
        assert X_val_tfidf.shape == (len(X_val), actual_feature_count)

        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            activation="relu",
            solver="adam",
            alpha=ALPHA,
            batch_size="auto",
            learning_rate_init=LEARNING_RATE_INIT,
            random_state=MODEL_RANDOM_STATE,
            shuffle=False,
        )

        history = []
        minibatch_history = []
        config_best_model = None
        config_best_epoch = None
        config_best_val_macro_f1 = -1.0
        random_generator = np.random.RandomState(BATCH_RANDOM_STATE)
        first_update = True
        update_step = 0
        cumulative_fit_seconds = 0.0
        runtime_start = perf_counter()

        for epoch in range(1, MAX_EPOCHS + 1):
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

                fit_start = perf_counter()
                if first_update:
                    model.partial_fit(X_batch, y_batch, classes=classes)
                    first_update = False
                else:
                    model.partial_fit(X_batch, y_batch)
                cumulative_fit_seconds += perf_counter() - fit_start

                batch_probabilities = model.predict_proba(X_batch)
                batch_loss = log_loss(
                    y_batch,
                    batch_probabilities,
                    labels=classes,
                )
                update_step += 1
                minibatch_history.append(
                    {
                        "update_step": update_step,
                        "epoch": epoch,
                        "batch": batch_number,
                        "batch_size": len(batch_indices),
                        "batch_loss": batch_loss,
                    }
                )

            train_probabilities = model.predict_proba(X_train_tfidf)
            val_probabilities = model.predict_proba(X_val_tfidf)
            train_predictions = model.classes_[train_probabilities.argmax(axis=1)]
            val_predictions = model.classes_[val_probabilities.argmax(axis=1)]

            row = {
                "epoch": epoch,
                "train_accuracy": accuracy_score(y_train, train_predictions),
                "train_macro_f1": f1_score(
                    y_train,
                    train_predictions,
                    average="macro",
                ),
                "val_accuracy": accuracy_score(y_val, val_predictions),
                "val_macro_f1": f1_score(
                    y_val,
                    val_predictions,
                    average="macro",
                ),
                "train_loss": log_loss(
                    y_train,
                    train_probabilities,
                    labels=classes,
                ),
                "val_loss": log_loss(
                    y_val,
                    val_probabilities,
                    labels=classes,
                ),
                "loss_type": "cross_entropy",
                "cumulative_fit_seconds": cumulative_fit_seconds,
            }
            history.append(row)

            if row["val_macro_f1"] > config_best_val_macro_f1:
                config_best_val_macro_f1 = row["val_macro_f1"]
                config_best_epoch = epoch
                config_best_model = deepcopy(model)

            print(
                f"features={max_features} | epoch={epoch:03d}/{MAX_EPOCHS} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"train_f1={row['train_macro_f1']:.4f} | "
                f"val_f1={row['val_macro_f1']:.4f}"
            )

        runtime_seconds = perf_counter() - runtime_start
        history_df = pd.DataFrame(history)
        minibatch_df = pd.DataFrame(minibatch_history)
        history_df.to_csv(
            OUTPUT_DIR / f"{config_name}_history.csv",
            index=False,
        )
        minibatch_df.to_csv(
            OUTPUT_DIR / f"{config_name}_minibatch_history.csv",
            index=False,
        )

        best_row = history_df.loc[
            history_df["epoch"] == config_best_epoch
        ].iloc[0]
        minimum_loss_row = history_df.loc[history_df["val_loss"].idxmin()]
        trainable_parameters = sum(
            array.size for array in config_best_model.coefs_
        ) + sum(array.size for array in config_best_model.intercepts_)

        results.append(
            {
                "algorithm": "MLP",
                "stage": "round2_stage4_tfidf_dimension",
                "config_order": config_order,
                "config": config_name,
                "max_features": max_features,
                "actual_feature_count": actual_feature_count,
                "hidden_layer_sizes": str(HIDDEN_LAYER_SIZES),
                "alpha": ALPHA,
                "learning_rate_init": LEARNING_RATE_INIT,
                "mini_batch_size": MINI_BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "best_epoch": config_best_epoch,
                "train_accuracy": best_row["train_accuracy"],
                "train_macro_f1": best_row["train_macro_f1"],
                "val_accuracy": best_row["val_accuracy"],
                "val_macro_f1": best_row["val_macro_f1"],
                "selection_metric": "val_macro_f1",
                "train_loss": best_row["train_loss"],
                "val_loss": best_row["val_loss"],
                "minimum_val_loss_epoch": int(minimum_loss_row["epoch"]),
                "minimum_val_loss": minimum_loss_row["val_loss"],
                "trainable_parameters": trainable_parameters,
                "train_nonzero_values": int(X_train_tfidf.nnz),
                "val_nonzero_values": int(X_val_tfidf.nnz),
                "vectorize_seconds": vectorize_seconds,
                "fit_seconds": best_row["cumulative_fit_seconds"],
                "runtime_seconds": runtime_seconds,
            }
        )
        best_models[max_features] = config_best_model
        fitted_vectorizers[max_features] = vectorizer

        plot_epoch_loss(
            history_df,
            max_features,
            config_best_epoch,
            OUTPUT_DIR / f"{config_name}_loss.png",
        )
        plot_minibatch_loss(
            minibatch_df,
            max_features,
            batches_per_epoch,
            OUTPUT_DIR / f"{config_name}_minibatch_loss.png",
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "val_accuracy", "val_loss", "max_features"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    results_df.to_csv(
        OUTPUT_DIR / "mlp_stage4_dimension_results.csv",
        index=False,
    )

    strict_best_row = results_df.iloc[0]
    strict_best_dimension = int(strict_best_row["max_features"])
    best_score = float(strict_best_row["val_macro_f1"])
    practical_ties = results_df[
        results_df["val_macro_f1"] >= best_score - PRACTICAL_TIE_TOLERANCE
    ].copy()
    recommended_row = practical_ties.sort_values(
        by=["max_features", "val_macro_f1", "val_accuracy", "val_loss"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).iloc[0]
    selected_dimension = int(recommended_row["max_features"])
    selected_model = best_models[selected_dimension]
    selected_vectorizer = fitted_vectorizers[selected_dimension]
    X_val_selected = selected_vectorizer.transform(X_val)
    selected_predictions = selected_model.predict(X_val_selected)

    save_dimension_comparison(results_df, selected_dimension)

    report_df = pd.DataFrame(
        classification_report(
            y_val,
            selected_predictions,
            labels=classes,
            target_names=[CLASS_NAMES[label] for label in classes],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(
        OUTPUT_DIR / "mlp_stage4_selected_classification_report.csv"
    )

    matrix = confusion_matrix(y_val, selected_predictions, labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / "mlp_stage4_selected_confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(
        "Selected MLP M4 Confusion Matrix "
        f"(max_features={selected_dimension})"
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mlp_stage4_selected_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    joblib.dump(
        {
            "vectorizer": selected_vectorizer,
            "model": selected_model,
            "algorithm": "MLP",
            "stage": "round2_stage4_tfidf_dimension",
            "max_features": selected_dimension,
            "actual_feature_count": int(
                recommended_row["actual_feature_count"]
            ),
            "strict_best_max_features": strict_best_dimension,
            "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
            "alpha": ALPHA,
            "learning_rate_init": LEARNING_RATE_INIT,
            "mini_batch_size": MINI_BATCH_SIZE,
            "best_epoch": int(recommended_row["best_epoch"]),
            "validation_accuracy": float(recommended_row["val_accuracy"]),
            "validation_macro_f1": float(recommended_row["val_macro_f1"]),
            "validation_loss": float(recommended_row["val_loss"]),
            "trainable_parameters": int(
                recommended_row["trainable_parameters"]
            ),
            "split_random_state": SPLIT_RANDOM_STATE,
            "model_random_state": MODEL_RANDOM_STATE,
            "batch_random_state": BATCH_RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
            "tfidf_config": {
                "max_features": selected_dimension,
                "lowercase": True,
                "ngram_range": (1, 1),
                "fit_on_training_subset_only": True,
            },
            "class_names": CLASS_NAMES,
        },
        OUTPUT_DIR / "mlp_stage4_selected.joblib",
    )

    decision = {
        "strict_best_max_features": strict_best_dimension,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "strict_best_val_accuracy": float(strict_best_row["val_accuracy"]),
        "strict_best_val_loss": float(strict_best_row["val_loss"]),
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "recommended_max_features": selected_dimension,
        "recommended_val_macro_f1": float(
            recommended_row["val_macro_f1"]
        ),
        "recommended_val_accuracy": float(recommended_row["val_accuracy"]),
        "recommended_val_loss": float(recommended_row["val_loss"]),
        "recommendation_rule": (
            "Choose the smallest max_features within 0.001 Macro-F1 of "
            "the strict best."
        ),
    }
    with (OUTPUT_DIR / "stage4_decision.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    settings = {
        "stage": "round2_stage4_tfidf_dimension",
        "controlled_variable": "tfidf_max_features",
        "max_feature_values": MAX_FEATURE_VALUES,
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "alpha": ALPHA,
        "learning_rate_init": LEARNING_RATE_INIT,
        "mini_batch_size": MINI_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
        "batch_random_state": BATCH_RANDOM_STATE,
        "primary_selection_metric": "val_macro_f1",
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "practical_tie_breaker": "smaller_max_features",
        "raw_text_headers_preserved": True,
        "official_test_set_used": False,
        "tfidf_fixed_settings": {
            "lowercase": True,
            "ngram_range": [1, 1],
            "fit_on_training_subset_only": True,
        },
    }
    with (OUTPUT_DIR / "experiment_settings.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("--- MLP M4阶段完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n严格最高 Macro-F1 的维度: {strict_best_dimension}")
    print(f"严格最高验证 Macro-F1: {strict_best_row['val_macro_f1']:.6f}")
    print(f"0.001 等价阈值下推荐维度: {selected_dimension}")
    print(f"推荐模型验证 Macro-F1: {recommended_row['val_macro_f1']:.6f}")
    print(f"推荐模型验证 Accuracy: {recommended_row['val_accuracy']:.6f}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

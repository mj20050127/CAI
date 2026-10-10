"""MLP round-2 stage 1: tune the width of one hidden layer.

This experiment keeps the raw text, split, TF-IDF representation, optimizer,
regularization, learning rate, and mini-batch size fixed. Only the number of
hidden units changes.
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
MAX_EPOCHS = 60
MINI_BATCH_SIZE = 128
ALPHA = 0.0001
LEARNING_RATE_INIT = 0.001
HIDDEN_WIDTHS = [50, 75, 100, 150, 200]
OUTPUT_DIR = Path("outputs") / "mlp_tuning_round2" / "stage1_width"

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
    """Load raw labeled texts; the official test set remains sealed."""
    train_df = pd.read_csv("train_data.csv")
    texts = train_df["text"].astype(str).tolist()
    targets = train_df["target"].to_numpy()
    return texts, targets


def plot_epoch_loss(history_df, width, best_epoch, output_path):
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
    axis.set_title(f"MLP Width Tuning: hidden=({width},), alpha={ALPHA}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_minibatch_loss(minibatch_df, width, batches_per_epoch, output_path):
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
    axis.set_title(f"MLP Mini-batch Loss: hidden=({width},), alpha={ALPHA}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
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

    vectorizer = TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        ngram_range=(1, 1),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    classes = np.sort(np.unique(y_train))
    batches_per_epoch = int(np.ceil(len(y_train) / MINI_BATCH_SIZE))

    print("--- MLP 第二轮调参：第一阶段（单隐藏层宽度）---")
    print(f"训练集形状: {X_train_tfidf.shape}")
    print(f"验证集形状: {X_val_tfidf.shape}")
    print(f"隐藏层宽度: {HIDDEN_WIDTHS}")
    print(f"固定 alpha: {ALPHA}")
    print(f"固定 learning_rate_init: {LEARNING_RATE_INIT}")
    print(f"固定 mini-batch size: {MINI_BATCH_SIZE}")
    print(f"最大 epoch: {MAX_EPOCHS}")
    print("选择指标: validation Macro-F1")
    print("=" * 72)

    results = []
    overall_best_model = None
    overall_best_width = None
    overall_best_epoch = None
    overall_best_score = -1.0

    for config_order, width in enumerate(HIDDEN_WIDTHS, start=1):
        config_name = f"h{width}_a0p0001"
        print(
            f"--- 配置 {config_order}/{len(HIDDEN_WIDTHS)}: "
            f"hidden=({width},), alpha={ALPHA} ---"
        )

        model = MLPClassifier(
            hidden_layer_sizes=(width,),
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
        best_model = None
        best_epoch = None
        best_val_macro_f1 = -1.0
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

            if row["val_macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = row["val_macro_f1"]
                best_epoch = epoch
                best_model = deepcopy(model)

            print(
                f"width={width:3d} | epoch={epoch:03d}/{MAX_EPOCHS} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"train_f1={row['train_macro_f1']:.4f} | "
                f"val_f1={row['val_macro_f1']:.4f}"
            )

        runtime_seconds = perf_counter() - runtime_start
        history_df = pd.DataFrame(history)
        minibatch_df = pd.DataFrame(minibatch_history)
        history_df.to_csv(OUTPUT_DIR / f"{config_name}_history.csv", index=False)
        minibatch_df.to_csv(
            OUTPUT_DIR / f"{config_name}_minibatch_history.csv",
            index=False,
        )

        best_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]
        minimum_loss_row = history_df.loc[history_df["val_loss"].idxmin()]
        trainable_parameters = sum(array.size for array in best_model.coefs_)
        trainable_parameters += sum(
            array.size for array in best_model.intercepts_
        )

        results.append(
            {
                "algorithm": "MLP",
                "stage": "round2_stage1_width",
                "config_order": config_order,
                "config": config_name,
                "hidden_layer_sizes": str((width,)),
                "hidden_width": width,
                "alpha": ALPHA,
                "learning_rate_init": LEARNING_RATE_INIT,
                "mini_batch_size": MINI_BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "best_epoch": best_epoch,
                "train_accuracy": best_row["train_accuracy"],
                "train_macro_f1": best_row["train_macro_f1"],
                "val_accuracy": best_row["val_accuracy"],
                "val_macro_f1": best_row["val_macro_f1"],
                "selection_metric": "val_macro_f1",
                "train_loss": best_row["train_loss"],
                "val_loss": best_row["val_loss"],
                "minimum_val_loss_epoch": int(minimum_loss_row["epoch"]),
                "minimum_val_loss": minimum_loss_row["val_loss"],
                "fit_seconds": best_row["cumulative_fit_seconds"],
                "runtime_seconds": runtime_seconds,
                "trainable_parameters": trainable_parameters,
            }
        )

        plot_epoch_loss(
            history_df,
            width,
            best_epoch,
            OUTPUT_DIR / f"{config_name}_loss.png",
        )
        plot_minibatch_loss(
            minibatch_df,
            width,
            batches_per_epoch,
            OUTPUT_DIR / f"{config_name}_minibatch_loss.png",
        )

        if best_val_macro_f1 > overall_best_score:
            overall_best_score = best_val_macro_f1
            overall_best_width = width
            overall_best_epoch = best_epoch
            overall_best_model = deepcopy(best_model)

    results_df = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "trainable_parameters", "config_order"],
        ascending=[False, True, True],
        ignore_index=True,
    )
    results_path = OUTPUT_DIR / "mlp_stage1_width_results.csv"
    results_df.to_csv(results_path, index=False)

    plot_df = results_df.sort_values("hidden_width")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        plot_df["hidden_width"],
        plot_df["val_macro_f1"],
        marker="o",
        label="Validation Macro-F1",
    )
    axis.plot(
        plot_df["hidden_width"],
        plot_df["val_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )
    axis.set_xlabel("Hidden units")
    axis.set_ylabel("Validation score")
    axis.set_title("MLP Stage 1: Single Hidden-Layer Width")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    comparison_figure_path = OUTPUT_DIR / "mlp_stage1_width_comparison.png"
    figure.savefig(comparison_figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    best_val_predictions = overall_best_model.predict(X_val_tfidf)
    report_df = pd.DataFrame(
        classification_report(
            y_val,
            best_val_predictions,
            labels=classes,
            target_names=[CLASS_NAMES[label] for label in classes],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(OUTPUT_DIR / "mlp_stage1_best_classification_report.csv")

    matrix = confusion_matrix(y_val, best_val_predictions, labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / "mlp_stage1_best_confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(
        f"Best MLP Stage-1 Confusion Matrix (hidden=({overall_best_width},))"
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mlp_stage1_best_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    best_row = results_df.iloc[0]
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": overall_best_model,
            "algorithm": "MLP",
            "stage": "round2_stage1_width",
            "hidden_layer_sizes": (overall_best_width,),
            "alpha": ALPHA,
            "learning_rate_init": LEARNING_RATE_INIT,
            "mini_batch_size": MINI_BATCH_SIZE,
            "best_epoch": overall_best_epoch,
            "validation_accuracy": float(best_row["val_accuracy"]),
            "validation_macro_f1": float(best_row["val_macro_f1"]),
            "split_random_state": SPLIT_RANDOM_STATE,
            "model_random_state": MODEL_RANDOM_STATE,
            "batch_random_state": BATCH_RANDOM_STATE,
            "tfidf_config": {
                "max_features": 5000,
                "lowercase": True,
                "ngram_range": (1, 1),
            },
            "class_names": CLASS_NAMES,
        },
        OUTPUT_DIR / "mlp_stage1_best.joblib",
    )

    settings = {
        "stage": "round2_stage1_width",
        "controlled_variable": "hidden_width",
        "hidden_widths": HIDDEN_WIDTHS,
        "alpha": ALPHA,
        "learning_rate_init": LEARNING_RATE_INIT,
        "mini_batch_size": MINI_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
        "batch_random_state": BATCH_RANDOM_STATE,
        "selection_metric": "val_macro_f1",
        "raw_text_headers_preserved": True,
        "official_test_set_used": False,
        "tfidf": {
            "max_features": 5000,
            "lowercase": True,
            "ngram_range": [1, 1],
            "fit_on_training_subset_only": True,
        },
    }
    with (OUTPUT_DIR / "experiment_settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("--- MLP 第二轮第一阶段完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n最佳隐藏层: ({overall_best_width},)")
    print(f"最佳 epoch: {overall_best_epoch}")
    print(f"最佳验证 Macro-F1: {overall_best_score:.6f}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

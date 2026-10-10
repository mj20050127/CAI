"""MLP round-2 stage 2: tune L2 regularization alpha at width 100.

Only ``alpha`` changes in this stage. Raw text, the stratified split, TF-IDF,
network width, optimizer, learning rate, mini-batch size, and epoch budget are
kept identical across configurations. The official unlabeled test set is not
loaded during tuning.
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
LEARNING_RATE_INIT = 0.001
MINI_BATCH_SIZE = 128
MAX_EPOCHS = 60
ALPHA_CONFIGS = [
    ("a0p00001", 0.00001),
    ("a0p0001", 0.0001),
    ("a0p0003", 0.0003),
    ("a0p001", 0.001),
    ("a0p003", 0.003),
    ("a0p01", 0.01),
]
OUTPUT_DIR = Path("outputs") / "mlp_tuning_round2" / "stage2_alpha"

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
    train_df = pd.read_csv("train_data.csv")
    texts = train_df["text"].astype(str).tolist()
    targets = train_df["target"].to_numpy()
    return texts, targets


def plot_epoch_loss(history_df, alpha, best_epoch, output_path):
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
    axis.set_title(f"MLP Alpha Tuning: hidden=(100,), alpha={alpha:g}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_minibatch_loss(minibatch_df, alpha, batches_per_epoch, output_path):
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
    axis.set_title(f"MLP Mini-batch Loss: hidden=(100,), alpha={alpha:g}")
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

    print("--- MLP 第二轮调参：M2阶段（L2正则化alpha）---")
    print(f"训练集形状: {X_train_tfidf.shape}")
    print(f"验证集形状: {X_val_tfidf.shape}")
    print(f"固定隐藏层: {HIDDEN_LAYER_SIZES}")
    print(f"alpha候选值: {[value for _, value in ALPHA_CONFIGS]}")
    print(f"固定 learning_rate_init: {LEARNING_RATE_INIT}")
    print(f"固定 mini-batch size: {MINI_BATCH_SIZE}")
    print(f"最大 epoch: {MAX_EPOCHS}")
    print("选择指标: validation Macro-F1")
    print("=" * 72)

    results = []
    best_models = {}

    for config_order, (alpha_name, alpha) in enumerate(ALPHA_CONFIGS, start=1):
        config_name = f"h100_{alpha_name}"
        print(
            f"--- 配置 {config_order}/{len(ALPHA_CONFIGS)}: "
            f"hidden={HIDDEN_LAYER_SIZES}, alpha={alpha:g} ---"
        )

        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            activation="relu",
            solver="adam",
            alpha=alpha,
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
                f"alpha={alpha:0.5g} | epoch={epoch:03d}/{MAX_EPOCHS} | "
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
                "stage": "round2_stage2_alpha",
                "config_order": config_order,
                "config": config_name,
                "hidden_layer_sizes": str(HIDDEN_LAYER_SIZES),
                "alpha": alpha,
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
                "fit_seconds": best_row["cumulative_fit_seconds"],
                "runtime_seconds": runtime_seconds,
                "trainable_parameters": trainable_parameters,
            }
        )
        best_models[alpha] = config_best_model

        plot_epoch_loss(
            history_df,
            alpha,
            config_best_epoch,
            OUTPUT_DIR / f"{config_name}_loss.png",
        )
        plot_minibatch_loss(
            minibatch_df,
            alpha,
            batches_per_epoch,
            OUTPUT_DIR / f"{config_name}_minibatch_loss.png",
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "val_accuracy", "val_loss", "alpha"],
        ascending=[False, False, True, False],
        ignore_index=True,
    )
    results_path = OUTPUT_DIR / "mlp_stage2_alpha_results.csv"
    results_df.to_csv(results_path, index=False)

    plot_df = results_df.sort_values("alpha")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        plot_df["alpha"],
        plot_df["val_macro_f1"],
        marker="o",
        label="Validation Macro-F1",
    )
    axis.plot(
        plot_df["alpha"],
        plot_df["val_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Alpha (log scale)")
    axis.set_ylabel("Validation score")
    axis.set_title("MLP M2: L2 Regularization at Hidden Width 100")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    comparison_figure_path = OUTPUT_DIR / "mlp_stage2_alpha_comparison.png"
    figure.savefig(comparison_figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    strict_best_row = results_df.iloc[0]
    strict_best_alpha = float(strict_best_row["alpha"])
    strict_best_model = best_models[strict_best_alpha]
    best_val_predictions = strict_best_model.predict(X_val_tfidf)

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
    report_df.to_csv(OUTPUT_DIR / "mlp_stage2_best_classification_report.csv")

    matrix = confusion_matrix(y_val, best_val_predictions, labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / "mlp_stage2_best_confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(
        f"Best MLP M2 Confusion Matrix (hidden=(100,), alpha={strict_best_alpha:g})"
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mlp_stage2_best_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": strict_best_model,
            "algorithm": "MLP",
            "stage": "round2_stage2_alpha",
            "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
            "alpha": strict_best_alpha,
            "learning_rate_init": LEARNING_RATE_INIT,
            "mini_batch_size": MINI_BATCH_SIZE,
            "best_epoch": int(strict_best_row["best_epoch"]),
            "validation_accuracy": float(strict_best_row["val_accuracy"]),
            "validation_macro_f1": float(strict_best_row["val_macro_f1"]),
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
        OUTPUT_DIR / "mlp_stage2_best.joblib",
    )

    best_score = float(results_df["val_macro_f1"].max())
    practical_ties = results_df[
        results_df["val_macro_f1"] >= best_score - 0.001
    ].copy()
    practical_ties = practical_ties.sort_values(
        by=["val_accuracy", "val_loss", "alpha"],
        ascending=[False, True, False],
        ignore_index=True,
    )
    recommended_row = practical_ties.iloc[0]

    settings = {
        "stage": "round2_stage2_alpha",
        "controlled_variable": "alpha",
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "alpha_values": [value for _, value in ALPHA_CONFIGS],
        "learning_rate_init": LEARNING_RATE_INIT,
        "mini_batch_size": MINI_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
        "batch_random_state": BATCH_RANDOM_STATE,
        "primary_selection_metric": "val_macro_f1",
        "practical_tie_tolerance": 0.001,
        "practical_tie_breakers": ["val_accuracy", "val_loss", "larger_alpha"],
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

    decision = {
        "strict_best_alpha": strict_best_alpha,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "strict_best_val_accuracy": float(strict_best_row["val_accuracy"]),
        "recommended_alpha_under_0p001_tolerance": float(
            recommended_row["alpha"]
        ),
        "recommended_val_macro_f1": float(recommended_row["val_macro_f1"]),
        "recommended_val_accuracy": float(recommended_row["val_accuracy"]),
        "recommended_val_loss": float(recommended_row["val_loss"]),
    }
    with (OUTPUT_DIR / "stage2_decision.json").open("w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("--- MLP M2阶段完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n严格最高Macro-F1的alpha: {strict_best_alpha:g}")
    print(f"严格最高验证Macro-F1: {strict_best_row['val_macro_f1']:.6f}")
    print(
        "0.001等价阈值下推荐alpha: "
        f"{float(recommended_row['alpha']):g}"
    )
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

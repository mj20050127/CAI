"""Extend the MLP TF-IDF dimension experiment with 15k and 25k features.

This script keeps the original M4 experiment unchanged, trains only the two
new dimensions, and combines the new rows with the existing 5k/10k/20k CSV.
The official unlabeled test set is never loaded.
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from mlp_tuning_round2_stage4 import (
    ALPHA,
    BATCH_RANDOM_STATE,
    HIDDEN_LAYER_SIZES,
    LEARNING_RATE_INIT,
    MAX_EPOCHS,
    MINI_BATCH_SIZE,
    MODEL_RANDOM_STATE,
    PRACTICAL_TIE_TOLERANCE,
    SPLIT_RANDOM_STATE,
    VALIDATION_SIZE,
    load_labeled_data,
    plot_epoch_loss,
    plot_minibatch_loss,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


NEW_MAX_FEATURE_VALUES = [15000, 25000]
ORIGINAL_OUTPUT_DIR = (
    Path("outputs") / "mlp_tuning_round2" / "stage4_tfidf_dimension"
)
ORIGINAL_RESULTS_PATH = ORIGINAL_OUTPUT_DIR / "mlp_stage4_dimension_results.csv"
OUTPUT_DIR = (
    Path("outputs")
    / "mlp_tuning_round2"
    / "stage4_tfidf_dimension_extension"
)


def save_combined_plot(results_df, selected_dimension):
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
        label=f"Recommended {selected_dimension}",
    )
    axes[0].set_xlabel("TF-IDF max_features")
    axes[0].set_ylabel("Validation score")
    axes[0].set_title("MLP TF-IDF Dimension Extension")
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
    axes[1].set_title("MLP Model Size")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mlp_stage4_all_dimension_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    if not ORIGINAL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Original M4 results not found: {ORIGINAL_RESULTS_PATH}. "
            "Run mlp_tuning_round2_stage4.py first."
        )

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

    print("--- MLP M4 TF-IDF维度扩展实验 ---")
    print(f"新增候选维度: {NEW_MAX_FEATURE_VALUES}")
    print(f"固定隐藏层: {HIDDEN_LAYER_SIZES}")
    print(f"固定 alpha: {ALPHA}")
    print(f"固定 learning_rate_init: {LEARNING_RATE_INIT}")
    print(f"固定 mini-batch size: {MINI_BATCH_SIZE}")
    print(f"最大 epoch: {MAX_EPOCHS}")
    print("官方测试集: 未加载")
    print("=" * 76)

    new_results = []
    for config_order, max_features in enumerate(NEW_MAX_FEATURE_VALUES, start=1):
        config_name = f"d{max_features}_h100_a0p01_lr0p0005"
        print(
            f"--- 新配置 {config_order}/{len(NEW_MAX_FEATURE_VALUES)}: "
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
                    y_train, train_predictions, average="macro"
                ),
                "val_accuracy": accuracy_score(y_val, val_predictions),
                "val_macro_f1": f1_score(
                    y_val, val_predictions, average="macro"
                ),
                "train_loss": log_loss(
                    y_train, train_probabilities, labels=classes
                ),
                "val_loss": log_loss(y_val, val_probabilities, labels=classes),
                "loss_type": "cross_entropy",
                "cumulative_fit_seconds": cumulative_fit_seconds,
            }
            history.append(row)

            if row["val_macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = row["val_macro_f1"]
                best_epoch = epoch
                best_model = deepcopy(model)

            print(
                f"features={max_features} | epoch={epoch:03d}/{MAX_EPOCHS} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"train_f1={row['train_macro_f1']:.4f} | "
                f"val_f1={row['val_macro_f1']:.4f}",
                flush=True,
            )

        runtime_seconds = perf_counter() - runtime_start
        history_df = pd.DataFrame(history)
        minibatch_df = pd.DataFrame(minibatch_history)
        history_df.to_csv(OUTPUT_DIR / f"{config_name}_history.csv", index=False)
        minibatch_df.to_csv(
            OUTPUT_DIR / f"{config_name}_minibatch_history.csv",
            index=False,
        )
        plot_epoch_loss(
            history_df,
            max_features,
            best_epoch,
            OUTPUT_DIR / f"{config_name}_loss.png",
        )
        plot_minibatch_loss(
            minibatch_df,
            max_features,
            batches_per_epoch,
            OUTPUT_DIR / f"{config_name}_minibatch_loss.png",
        )

        best_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]
        minimum_loss_row = history_df.loc[history_df["val_loss"].idxmin()]
        trainable_parameters = sum(array.size for array in best_model.coefs_)
        trainable_parameters += sum(
            array.size for array in best_model.intercepts_
        )

        result = {
            "algorithm": "MLP",
            "stage": "round2_stage4_tfidf_dimension_extension",
            "config_order": config_order,
            "config": config_name,
            "max_features": max_features,
            "actual_feature_count": actual_feature_count,
            "hidden_layer_sizes": str(HIDDEN_LAYER_SIZES),
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
            "trainable_parameters": trainable_parameters,
            "train_nonzero_values": int(X_train_tfidf.nnz),
            "val_nonzero_values": int(X_val_tfidf.nnz),
            "vectorize_seconds": vectorize_seconds,
            "fit_seconds": best_row["cumulative_fit_seconds"],
            "runtime_seconds": runtime_seconds,
        }
        new_results.append(result)

        joblib.dump(
            {
                "vectorizer": vectorizer,
                "model": best_model,
                "algorithm": "MLP",
                "stage": "round2_stage4_tfidf_dimension_extension",
                "max_features": max_features,
                "best_epoch": int(best_epoch),
                "validation_accuracy": float(best_row["val_accuracy"]),
                "validation_macro_f1": float(best_row["val_macro_f1"]),
                "split_random_state": SPLIT_RANDOM_STATE,
                "model_random_state": MODEL_RANDOM_STATE,
                "batch_random_state": BATCH_RANDOM_STATE,
            },
            OUTPUT_DIR / f"mlp_d{max_features}_best.joblib",
        )

    new_results_df = pd.DataFrame(new_results)
    new_results_df.to_csv(
        OUTPUT_DIR / "mlp_stage4_extension_new_results.csv",
        index=False,
    )

    original_results_df = pd.read_csv(ORIGINAL_RESULTS_PATH)
    combined_df = pd.concat(
        [original_results_df, new_results_df],
        ignore_index=True,
    )
    combined_df = combined_df.drop_duplicates(
        subset=["max_features"], keep="last"
    ).sort_values("max_features", ignore_index=True)
    combined_df["config_order"] = np.arange(1, len(combined_df) + 1)
    combined_df.to_csv(
        OUTPUT_DIR / "mlp_stage4_all_dimension_results.csv",
        index=False,
    )

    ranked_df = combined_df.sort_values(
        by=["val_macro_f1", "val_accuracy", "val_loss", "max_features"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    strict_best_row = ranked_df.iloc[0]
    strict_best_dimension = int(strict_best_row["max_features"])
    best_score = float(strict_best_row["val_macro_f1"])
    practical_ties = ranked_df[
        ranked_df["val_macro_f1"] >= best_score - PRACTICAL_TIE_TOLERANCE
    ].copy()
    recommended_row = practical_ties.sort_values(
        by=["max_features", "val_macro_f1", "val_accuracy", "val_loss"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).iloc[0]
    recommended_dimension = int(recommended_row["max_features"])
    save_combined_plot(combined_df, recommended_dimension)

    decision = {
        "dimensions_compared": combined_df["max_features"].astype(int).tolist(),
        "strict_best_max_features": strict_best_dimension,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "recommended_max_features": recommended_dimension,
        "recommended_val_macro_f1": float(recommended_row["val_macro_f1"]),
        "recommendation_rule": (
            "Choose the smallest max_features within 0.001 Macro-F1 of "
            "the strict best."
        ),
    }
    with (OUTPUT_DIR / "stage4_extension_decision.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    settings = {
        "new_max_feature_values": NEW_MAX_FEATURE_VALUES,
        "combined_with_original_dimensions": [5000, 10000, 20000],
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "alpha": ALPHA,
        "learning_rate_init": LEARNING_RATE_INIT,
        "mini_batch_size": MINI_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
        "batch_random_state": BATCH_RANDOM_STATE,
        "official_test_set_used": False,
    }
    with (OUTPUT_DIR / "experiment_settings.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("--- MLP维度扩展实验完成 ---")
    print(
        combined_df[
            [
                "max_features",
                "best_epoch",
                "val_accuracy",
                "val_macro_f1",
                "val_loss",
                "trainable_parameters",
            ]
        ].to_string(index=False)
    )
    print(f"\n严格最高维度: {strict_best_dimension}")
    print(f"推荐维度: {recommended_dimension}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

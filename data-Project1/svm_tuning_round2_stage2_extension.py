"""Extend the SVM TF-IDF dimension experiment with 15k and 25k features.

This script keeps the original S2 experiment unchanged, trains only the two
new dimensions, and combines the new rows with the existing 5k/10k/20k CSV.
The official unlabeled test set is never loaded.
"""

from pathlib import Path
from time import perf_counter
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, hinge_loss
from sklearn.model_selection import train_test_split

from svm_tuning_round2_stage2 import (
    FIXED_C,
    MODEL_RANDOM_STATE,
    PRACTICAL_TIE_TOLERANCE,
    SPLIT_RANDOM_STATE,
    VALIDATION_SIZE,
    build_model,
    load_labeled_data,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


NEW_MAX_FEATURE_VALUES = [15000, 25000]
ORIGINAL_OUTPUT_DIR = (
    Path("outputs") / "svm_tuning_round2" / "stage2_tfidf_dimension"
)
ORIGINAL_RESULTS_PATH = ORIGINAL_OUTPUT_DIR / "svm_stage2_dimension_results.csv"
OUTPUT_DIR = (
    Path("outputs")
    / "svm_tuning_round2"
    / "stage2_tfidf_dimension_extension"
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
    axes[0].set_title("SVM TF-IDF Dimension Extension")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        plot_df["max_features"],
        plot_df["val_hinge_loss"],
        marker="o",
        color="tab:orange",
        label="Validation hinge loss",
    )
    axes[1].set_xlabel("TF-IDF max_features")
    axes[1].set_title("SVM Loss and Complexity")
    axes[1].grid(alpha=0.3)
    left_axis = axes[1]
    left_axis.set_ylabel("Validation hinge loss")
    right_axis = left_axis.twinx()
    right_axis.plot(
        plot_df["max_features"],
        plot_df["support_vectors"],
        marker="s",
        color="tab:purple",
        label="Support vectors",
    )
    right_axis.set_ylabel("Support vectors")
    lines_left, labels_left = left_axis.get_legend_handles_labels()
    lines_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(lines_left + lines_right, labels_left + labels_right)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "svm_stage2_all_dimension_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    if not ORIGINAL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Original SVM S2 results not found: {ORIGINAL_RESULTS_PATH}. "
            "Run svm_tuning_round2_stage2.py first."
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

    print("--- SVM S2 TF-IDF维度扩展实验 ---")
    print(f"新增候选维度: {NEW_MAX_FEATURE_VALUES}")
    print(f"固定 C: {FIXED_C:g}")
    print("固定 kernel: linear")
    print("官方测试集: 未加载")
    print("=" * 76)

    new_results = []
    for config_order, max_features in enumerate(NEW_MAX_FEATURE_VALUES, start=1):
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

        model = build_model()
        fit_start = perf_counter()
        model.fit(X_train_tfidf, y_train)
        fit_seconds = perf_counter() - fit_start

        prediction_start = perf_counter()
        train_predictions = model.predict(X_train_tfidf)
        val_predictions = model.predict(X_val_tfidf)
        prediction_seconds = perf_counter() - prediction_start

        train_decision = model.decision_function(X_train_tfidf)
        val_decision = model.decision_function(X_val_tfidf)
        iteration_counts = np.asarray(model.n_iter_)
        row = {
            "algorithm": "Linear SVM (SVC, linear kernel)",
            "stage": "round2_stage2_tfidf_dimension_extension",
            "config_order": config_order,
            "config": f"max_features={max_features}",
            "C": FIXED_C,
            "max_features": max_features,
            "actual_feature_count": actual_feature_count,
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "train_macro_f1": f1_score(
                y_train, train_predictions, average="macro"
            ),
            "val_accuracy": accuracy_score(y_val, val_predictions),
            "val_macro_f1": f1_score(
                y_val, val_predictions, average="macro"
            ),
            "selection_metric": "val_macro_f1",
            "loss_type": "multiclass_hinge_without_regularization_term",
            "train_hinge_loss": hinge_loss(
                y_train, train_decision, labels=classes
            ),
            "val_hinge_loss": hinge_loss(y_val, val_decision, labels=classes),
            "support_vectors": int(model.n_support_.sum()),
            "sum_binary_iterations": int(iteration_counts.sum()),
            "max_binary_iterations": int(iteration_counts.max()),
            "train_nonzero_values": int(X_train_tfidf.nnz),
            "val_nonzero_values": int(X_val_tfidf.nnz),
            "vectorize_seconds": vectorize_seconds,
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
        }
        new_results.append(row)

        joblib.dump(
            {
                "vectorizer": vectorizer,
                "model": model,
                "algorithm": "Linear SVM (SVC, linear kernel)",
                "stage": "round2_stage2_tfidf_dimension_extension",
                "C": FIXED_C,
                "max_features": max_features,
                "validation_accuracy": float(row["val_accuracy"]),
                "validation_macro_f1": float(row["val_macro_f1"]),
                "validation_hinge_loss": float(row["val_hinge_loss"]),
                "support_vectors": int(row["support_vectors"]),
                "split_random_state": SPLIT_RANDOM_STATE,
                "model_random_state": MODEL_RANDOM_STATE,
            },
            OUTPUT_DIR / f"svm_d{max_features}.joblib",
        )

        print(
            f"features={actual_feature_count} | "
            f"train_hinge={row['train_hinge_loss']:.6f} | "
            f"val_hinge={row['val_hinge_loss']:.6f} | "
            f"train_macro_f1={row['train_macro_f1']:.6f} | "
            f"val_macro_f1={row['val_macro_f1']:.6f} | "
            f"support_vectors={row['support_vectors']} | "
            f"fit_seconds={fit_seconds:.2f}",
            flush=True,
        )

    new_results_df = pd.DataFrame(new_results)
    new_results_df.to_csv(
        OUTPUT_DIR / "svm_stage2_extension_new_results.csv",
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
        OUTPUT_DIR / "svm_stage2_all_dimension_results.csv",
        index=False,
    )

    ranked_df = combined_df.sort_values(
        by=[
            "val_macro_f1",
            "val_accuracy",
            "val_hinge_loss",
            "max_features",
        ],
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
        by=["max_features", "val_macro_f1", "val_accuracy", "val_hinge_loss"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).iloc[0]
    recommended_dimension = int(recommended_row["max_features"])
    save_combined_plot(combined_df, recommended_dimension)

    decision = {
        "dimensions_compared": combined_df["max_features"].astype(int).tolist(),
        "strict_best_max_features": strict_best_dimension,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "strict_best_val_hinge_loss": float(strict_best_row["val_hinge_loss"]),
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "recommended_max_features": recommended_dimension,
        "recommended_val_macro_f1": float(recommended_row["val_macro_f1"]),
        "recommendation_rule": (
            "Choose the smallest max_features within 0.001 Macro-F1 of "
            "the strict best."
        ),
    }
    with (OUTPUT_DIR / "stage2_extension_decision.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    settings = {
        "new_max_feature_values": NEW_MAX_FEATURE_VALUES,
        "combined_with_original_dimensions": [5000, 10000, 20000],
        "fixed_C": FIXED_C,
        "kernel": "linear",
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
        "official_test_set_used": False,
    }
    with (OUTPUT_DIR / "experiment_settings.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("--- SVM维度扩展实验完成 ---")
    print(
        combined_df[
            [
                "max_features",
                "val_accuracy",
                "val_macro_f1",
                "val_hinge_loss",
                "support_vectors",
                "fit_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\n严格最高维度: {strict_best_dimension}")
    print(f"推荐维度: {recommended_dimension}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

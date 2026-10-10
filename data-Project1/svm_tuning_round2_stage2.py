"""SVM round-2 stage 2: tune the TF-IDF feature dimension.

This stage fixes C=500 selected in S1 and changes only ``max_features``.
Raw email text, the stratified split, and all other TF-IDF/SVM settings stay
controlled. The official unlabeled test set is not loaded during tuning.
"""

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
    hinge_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
FIXED_C = 500.0
MAX_FEATURE_VALUES = [5000, 10000, 20000]
PRACTICAL_TIE_TOLERANCE = 0.001
OUTPUT_DIR = Path("outputs") / "svm_tuning_round2" / "stage2_tfidf_dimension"

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
    """Load labeled data without touching the official test set."""
    train_df = pd.read_csv("train_data.csv")
    texts = train_df["text"].astype(str).tolist()
    targets = train_df["target"].to_numpy()
    return texts, targets


def build_model():
    """Build the fixed SVM configuration selected in S1."""
    return SVC(
        C=FIXED_C,
        kernel="linear",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=0.001,
        cache_size=512,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=MODEL_RANDOM_STATE,
    )


def save_confusion_matrix(matrix, classes, selected_dimension):
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / "svm_stage2_selected_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(
        "Selected SVM S2 Confusion Matrix "
        f"(max_features={selected_dimension}, C={FIXED_C:g})"
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "svm_stage2_selected_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_comparison_plot(results_df, selected_dimension):
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
    axes[0].set_title("SVM S2: TF-IDF Dimension")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        plot_df["max_features"],
        plot_df["train_hinge_loss"],
        marker="o",
        label="Training hinge loss",
    )
    axes[1].plot(
        plot_df["max_features"],
        plot_df["val_hinge_loss"],
        marker="s",
        label="Validation hinge loss",
    )
    axes[1].axvline(
        selected_dimension,
        color="tab:red",
        linestyle="--",
        label=f"Selected {selected_dimension}",
    )
    axes[1].set_xlabel("TF-IDF max_features")
    axes[1].set_ylabel("Multiclass hinge loss")
    axes[1].set_title("SVM S2: Hinge Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "svm_stage2_dimension_comparison.png",
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

    print("--- SVM 第二轮调参：S2阶段（TF-IDF维度）---")
    print(f"训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"固定 C: {FIXED_C:g}")
    print(f"max_features 候选值: {MAX_FEATURE_VALUES}")
    print("固定 ngram_range: (1, 1)")
    print("固定 lowercase: True")
    print("kernel: linear")
    print("选择指标: validation Macro-F1")
    print(f"近似持平阈值: {PRACTICAL_TIE_TOLERANCE}")
    print("官方测试集: 未加载")
    print("=" * 76)

    results = []
    trained_models = {}
    fitted_vectorizers = {}

    for config_order, max_features in enumerate(MAX_FEATURE_VALUES, start=1):
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
            "stage": "round2_stage2_tfidf_dimension",
            "config_order": config_order,
            "config": f"max_features={max_features}",
            "C": FIXED_C,
            "max_features": max_features,
            "actual_feature_count": actual_feature_count,
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
            "selection_metric": "val_macro_f1",
            "loss_type": "multiclass_hinge_without_regularization_term",
            "train_hinge_loss": hinge_loss(
                y_train,
                train_decision,
                labels=classes,
            ),
            "val_hinge_loss": hinge_loss(
                y_val,
                val_decision,
                labels=classes,
            ),
            "support_vectors": int(model.n_support_.sum()),
            "sum_binary_iterations": int(iteration_counts.sum()),
            "max_binary_iterations": int(iteration_counts.max()),
            "train_nonzero_values": int(X_train_tfidf.nnz),
            "val_nonzero_values": int(X_val_tfidf.nnz),
            "vectorize_seconds": vectorize_seconds,
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
        }
        results.append(row)
        trained_models[max_features] = model
        fitted_vectorizers[max_features] = vectorizer

        print(
            f"features={actual_feature_count} | "
            f"train_hinge={row['train_hinge_loss']:.6f} | "
            f"val_hinge={row['val_hinge_loss']:.6f} | "
            f"train_acc={row['train_accuracy']:.6f} | "
            f"val_acc={row['val_accuracy']:.6f} | "
            f"train_macro_f1={row['train_macro_f1']:.6f} | "
            f"val_macro_f1={row['val_macro_f1']:.6f} | "
            f"support_vectors={row['support_vectors']} | "
            f"fit_seconds={fit_seconds:.2f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "val_accuracy", "val_hinge_loss", "max_features"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    results_df.to_csv(
        OUTPUT_DIR / "svm_stage2_dimension_results.csv",
        index=False,
    )

    strict_best_row = results_df.iloc[0]
    strict_best_dimension = int(strict_best_row["max_features"])
    best_score = float(strict_best_row["val_macro_f1"])
    practical_ties = results_df[
        results_df["val_macro_f1"] >= best_score - PRACTICAL_TIE_TOLERANCE
    ].copy()
    recommended_row = practical_ties.sort_values(
        by=["max_features", "val_macro_f1", "val_accuracy", "val_hinge_loss"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).iloc[0]
    selected_dimension = int(recommended_row["max_features"])
    selected_model = trained_models[selected_dimension]
    selected_vectorizer = fitted_vectorizers[selected_dimension]
    X_val_selected = selected_vectorizer.transform(X_val)
    selected_predictions = selected_model.predict(X_val_selected)

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
        OUTPUT_DIR / "svm_stage2_selected_classification_report.csv"
    )

    matrix = confusion_matrix(y_val, selected_predictions, labels=classes)
    save_confusion_matrix(matrix, classes, selected_dimension)
    save_comparison_plot(results_df, selected_dimension)

    joblib.dump(
        {
            "vectorizer": selected_vectorizer,
            "model": selected_model,
            "algorithm": "Linear SVM (SVC, linear kernel)",
            "stage": "round2_stage2_tfidf_dimension",
            "C": FIXED_C,
            "max_features": selected_dimension,
            "actual_feature_count": int(
                recommended_row["actual_feature_count"]
            ),
            "strict_best_max_features": strict_best_dimension,
            "validation_accuracy": float(recommended_row["val_accuracy"]),
            "validation_macro_f1": float(recommended_row["val_macro_f1"]),
            "validation_hinge_loss": float(
                recommended_row["val_hinge_loss"]
            ),
            "support_vectors": int(recommended_row["support_vectors"]),
            "split_random_state": SPLIT_RANDOM_STATE,
            "model_random_state": MODEL_RANDOM_STATE,
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
        OUTPUT_DIR / "svm_stage2_selected.joblib",
    )

    decision = {
        "strict_best_max_features": strict_best_dimension,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "strict_best_val_accuracy": float(strict_best_row["val_accuracy"]),
        "strict_best_val_hinge_loss": float(
            strict_best_row["val_hinge_loss"]
        ),
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "recommended_max_features": selected_dimension,
        "recommended_val_macro_f1": float(
            recommended_row["val_macro_f1"]
        ),
        "recommended_val_accuracy": float(recommended_row["val_accuracy"]),
        "recommended_val_hinge_loss": float(
            recommended_row["val_hinge_loss"]
        ),
        "recommendation_rule": (
            "Choose the smallest max_features within 0.001 Macro-F1 of "
            "the strict best."
        ),
    }
    with (OUTPUT_DIR / "stage2_decision.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    settings = {
        "stage": "round2_stage2_tfidf_dimension",
        "controlled_variable": "tfidf_max_features",
        "fixed_C": FIXED_C,
        "max_feature_values": MAX_FEATURE_VALUES,
        "kernel": "linear",
        "tol": 0.001,
        "shrinking": True,
        "class_weight": None,
        "probability": False,
        "cache_size_mb": 512,
        "max_iter": -1,
        "decision_function_shape": "ovr",
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
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
    print("--- SVM S2阶段完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n严格最高 Macro-F1 的维度: {strict_best_dimension}")
    print(f"严格最高验证 Macro-F1: {strict_best_row['val_macro_f1']:.6f}")
    print(f"0.001 等价阈值下推荐维度: {selected_dimension}")
    print(f"推荐模型验证 Macro-F1: {recommended_row['val_macro_f1']:.6f}")
    print(f"推荐模型验证 Accuracy: {recommended_row['val_accuracy']:.6f}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

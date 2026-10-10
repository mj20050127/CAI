"""SVM round-2 stage 1: refine the regularization parameter C.

This stage keeps the raw email text, stratified split, and TF-IDF settings
identical to the MLP experiments. Only ``C`` changes. The official unlabeled
test set is deliberately not loaded during model selection.
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


# 使用非交互式后端，确保在 Anaconda Prompt 中直接保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
C_VALUES = [10.0, 30.0, 50.0, 100.0, 200.0, 500.0]
PRACTICAL_TIE_TOLERANCE = 0.001
OUTPUT_DIR = Path("outputs") / "svm_tuning_round2" / "stage1_c"

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
    """Load labeled data only; the official test set is excluded from tuning."""
    train_df = pd.read_csv("train_data.csv")
    texts = train_df["text"].astype(str).tolist()
    targets = train_df["target"].to_numpy()
    return texts, targets


def build_model(c_value):
    """Create one controlled linear-kernel SVM configuration."""
    return SVC(
        C=c_value,
        kernel="linear",
        degree=3,              # 线性核下不参与计算，保留默认值便于记录。
        gamma="scale",        # 线性核下不参与计算。
        coef0=0.0,             # 线性核下不参与计算。
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


def save_confusion_matrix(matrix, classes, selected_c):
    matrix_path = OUTPUT_DIR / "svm_stage1_selected_confusion_matrix.csv"
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(matrix_path)

    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"Selected SVM S1 Confusion Matrix (C={selected_c:g})")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "svm_stage1_selected_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_comparison_plot(results_df, selected_c):
    plot_df = results_df.sort_values("C")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(
        plot_df["C"],
        plot_df["val_macro_f1"],
        marker="o",
        label="Validation Macro-F1",
    )
    axes[0].plot(
        plot_df["C"],
        plot_df["val_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )
    axes[0].axvline(
        selected_c,
        color="tab:red",
        linestyle="--",
        label=f"Selected C={selected_c:g}",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C (log scale)")
    axes[0].set_ylabel("Validation score")
    axes[0].set_title("SVM S1: Validation Scores")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        plot_df["C"],
        plot_df["train_hinge_loss"],
        marker="o",
        label="Training hinge loss",
    )
    axes[1].plot(
        plot_df["C"],
        plot_df["val_hinge_loss"],
        marker="s",
        label="Validation hinge loss",
    )
    axes[1].axvline(
        selected_c,
        color="tab:red",
        linestyle="--",
        label=f"Selected C={selected_c:g}",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C (log scale)")
    axes[1].set_ylabel("Multiclass hinge loss")
    axes[1].set_title("SVM S1: Hinge Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "svm_stage1_c_comparison.png",
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

    # TF-IDF 只在训练子集上拟合，防止验证集信息泄漏。
    vectorizer = TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        ngram_range=(1, 1),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    classes = np.sort(np.unique(y_train))

    assert X_train_tfidf.shape == (len(X_train), len(vectorizer.vocabulary_))
    assert X_val_tfidf.shape[1] == X_train_tfidf.shape[1]

    print("--- SVM 第二轮调参：S1阶段（C）---")
    print(f"训练集形状: {X_train_tfidf.shape}")
    print(f"验证集形状: {X_val_tfidf.shape}")
    print(f"C 候选值: {C_VALUES}")
    print("kernel: linear")
    print("tol: 0.001")
    print("shrinking: True")
    print("class_weight: None")
    print("decision_function_shape: ovr")
    print("probability: False")
    print("cache_size: 512 MB")
    print("max_iter: -1")
    print("选择指标: validation Macro-F1")
    print(f"近似持平阈值: {PRACTICAL_TIE_TOLERANCE}")
    print("官方测试集: 未加载")
    print("=" * 76)

    results = []
    trained_models = {}

    for config_order, c_value in enumerate(C_VALUES, start=1):
        print(
            f"--- 配置 {config_order}/{len(C_VALUES)}: C={c_value:g} ---"
        )
        model = build_model(c_value)

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
            "stage": "round2_stage1_c",
            "config_order": config_order,
            "config": f"C={c_value:g}",
            "C": c_value,
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
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
        }
        results.append(row)
        trained_models[c_value] = model

        print(
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
        by=["val_macro_f1", "val_accuracy", "val_hinge_loss", "C"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    results_path = OUTPUT_DIR / "svm_stage1_c_results.csv"
    results_df.to_csv(results_path, index=False)

    # 严格最优用于呈现最高分；近似持平时推荐更小的 C。
    strict_best_row = results_df.iloc[0]
    strict_best_c = float(strict_best_row["C"])
    best_score = float(strict_best_row["val_macro_f1"])
    practical_ties = results_df[
        results_df["val_macro_f1"] >= best_score - PRACTICAL_TIE_TOLERANCE
    ].copy()
    recommended_row = practical_ties.sort_values(
        by=["C", "val_macro_f1", "val_accuracy", "val_hinge_loss"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).iloc[0]
    selected_c = float(recommended_row["C"])
    selected_model = trained_models[selected_c]
    selected_val_predictions = selected_model.predict(X_val_tfidf)

    report_df = pd.DataFrame(
        classification_report(
            y_val,
            selected_val_predictions,
            labels=classes,
            target_names=[CLASS_NAMES[label] for label in classes],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(
        OUTPUT_DIR / "svm_stage1_selected_classification_report.csv"
    )

    matrix = confusion_matrix(y_val, selected_val_predictions, labels=classes)
    save_confusion_matrix(matrix, classes, selected_c)
    save_comparison_plot(results_df, selected_c)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": selected_model,
            "algorithm": "Linear SVM (SVC, linear kernel)",
            "stage": "round2_stage1_c",
            "C": selected_c,
            "strict_best_C": strict_best_c,
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
                "max_features": 5000,
                "lowercase": True,
                "ngram_range": (1, 1),
                "fit_on_training_subset_only": True,
            },
            "class_names": CLASS_NAMES,
        },
        OUTPUT_DIR / "svm_stage1_selected.joblib",
    )

    decision = {
        "strict_best_C": strict_best_c,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "strict_best_val_accuracy": float(strict_best_row["val_accuracy"]),
        "strict_best_val_hinge_loss": float(
            strict_best_row["val_hinge_loss"]
        ),
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "recommended_C": selected_c,
        "recommended_val_macro_f1": float(
            recommended_row["val_macro_f1"]
        ),
        "recommended_val_accuracy": float(recommended_row["val_accuracy"]),
        "recommended_val_hinge_loss": float(
            recommended_row["val_hinge_loss"]
        ),
        "recommendation_rule": (
            "Choose the smallest C within 0.001 Macro-F1 of the strict best."
        ),
    }
    with (OUTPUT_DIR / "stage1_decision.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    settings = {
        "stage": "round2_stage1_c",
        "controlled_variable": "C",
        "C_values": C_VALUES,
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
        "practical_tie_breaker": "smaller_C",
        "raw_text_headers_preserved": True,
        "official_test_set_used": False,
        "tfidf": {
            "max_features": 5000,
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
    print("--- SVM S1阶段完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n严格最高 Macro-F1 的 C: {strict_best_c:g}")
    print(f"严格最高验证 Macro-F1: {strict_best_row['val_macro_f1']:.6f}")
    print(f"0.001 等价阈值下推荐 C: {selected_c:g}")
    print(f"推荐模型验证 Macro-F1: {recommended_row['val_macro_f1']:.6f}")
    print(f"推荐模型验证 Accuracy: {recommended_row['val_accuracy']:.6f}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

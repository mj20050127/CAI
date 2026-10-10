"""Linear SVM hyperparameter experiment for the text classification task."""

from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
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


# 使用非交互式后端，确保在 Anaconda Prompt 中可以直接保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
SVM_C_VALUES = [0.1, 1.0, 10.0, 100.0]
OUTPUT_DIR = Path("outputs") / "svm_comparison"

# 标签名称来自对训练文本主题的核对；训练和评分仍使用原始整数标签。
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
    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data_unlabeled.csv")
    X_all = train_df["text"].astype(str).tolist()
    y_all = train_df["target"].to_numpy()
    X_test_unlabeled = test_df["text"].astype(str).tolist()
    return X_all, y_all, X_test_unlabeled


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 使用与 MLP 完全相同的分层划分。
    X_all, y_all, X_test_unlabeled = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    # 2. 只在训练子集上拟合 TF-IDF，避免验证集和测试集信息泄露。
    vectorizer = TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        ngram_range=(1, 1),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test_unlabeled)
    classes = sorted(pd.unique(y_train))

    assert X_train_tfidf.shape == (len(X_train), len(vectorizer.vocabulary_))
    assert X_val_tfidf.shape[1] == X_train_tfidf.shape[1]
    assert X_test_tfidf.shape[1] == X_train_tfidf.shape[1]

    print("--- 线性 SVM 参数设置 ---")
    print(f"C 候选值: {SVM_C_VALUES}")
    print("kernel: linear")
    print("tol: 0.001")
    print("shrinking: True")
    print("class_weight: None")
    print("decision_function_shape: ovr")
    print("probability: False")
    print("cache_size: 512 MB")
    print("max_iter: -1（不限制迭代次数，直到收敛）")
    print(f"random_state: {RANDOM_STATE}")
    print(f"训练集形状: {X_train_tfidf.shape}")
    print(f"验证集形状: {X_val_tfidf.shape}")
    print("=" * 70)

    results = []
    trained_models = {}

    # 3. 只改变 C，其余条件保持一致。
    for C in SVM_C_VALUES:
        print(f"--- 训练线性 SVM: C={C:g} ---")
        model = SVC(
            C=C,
            kernel="linear",
            degree=3,             # 线性核下不参与计算，显式保留默认值。
            gamma="scale",        # 线性核下不参与计算，显式保留默认值。
            coef0=0.0,            # 线性核下不参与计算，显式保留默认值。
            shrinking=True,
            probability=False,
            tol=0.001,
            cache_size=512,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=RANDOM_STATE,
        )

        fit_start = perf_counter()
        model.fit(X_train_tfidf, y_train)
        fit_seconds = perf_counter() - fit_start

        prediction_start = perf_counter()
        train_predictions = model.predict(X_train_tfidf)
        val_predictions = model.predict(X_val_tfidf)
        prediction_seconds = perf_counter() - prediction_start

        train_decision = model.decision_function(X_train_tfidf)
        val_decision = model.decision_function(X_val_tfidf)

        result = {
            "algorithm": "Linear SVM",
            "config": f"C={C:g}",
            "C": C,
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "train_macro_f1": f1_score(y_train, train_predictions, average="macro"),
            "val_accuracy": accuracy_score(y_val, val_predictions),
            "val_macro_f1": f1_score(y_val, val_predictions, average="macro"),
            "selection_metric": "val_macro_f1",
            # 此处 hinge loss 不包含正则化项，用于统一比较分类间隔表现。
            "loss_type": "hinge",
            "train_loss": hinge_loss(y_train, train_decision, labels=classes),
            "val_loss": hinge_loss(y_val, val_decision, labels=classes),
            "support_vectors": int(model.n_support_.sum()),
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "complexity_name": "support_vectors",
            "complexity_value": int(model.n_support_.sum()),
        }
        results.append(result)
        trained_models[C] = model

        print(
            f"train_loss={result['train_loss']:.4f} | "
            f"val_loss={result['val_loss']:.4f} | "
            f"train_accuracy={result['train_accuracy']:.4f} | "
            f"val_accuracy={result['val_accuracy']:.4f} | "
            f"train_macro_f1={result['train_macro_f1']:.4f} | "
            f"val_macro_f1={result['val_macro_f1']:.4f} | "
            f"support_vectors={result['support_vectors']} | "
            f"fit_seconds={fit_seconds:.2f}"
        )

    # 4. 按验证 Macro-F1 排序；完全相同时优先选择更小的 C。
    results_df = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "C"],
        ascending=[False, True],
        ignore_index=True,
    )
    results_path = OUTPUT_DIR / "svm_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    best_C = float(results_df.iloc[0]["C"])
    best_model = trained_models[best_C]
    best_val_predictions = best_model.predict(X_val_tfidf)

    # 5. 保存最佳 SVM 的分类报告、混淆矩阵和验证阶段模型。
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
    report_path = OUTPUT_DIR / "svm_best_classification_report.csv"
    report_df.to_csv(report_path)

    matrix = confusion_matrix(y_val, best_val_predictions, labels=classes)
    matrix_df = pd.DataFrame(matrix, index=classes, columns=classes)
    matrix_path = OUTPUT_DIR / "svm_best_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"Linear SVM Validation Confusion Matrix (C={best_C:g})")
    figure.tight_layout()
    confusion_figure_path = OUTPUT_DIR / "svm_best_confusion_matrix.png"
    figure.savefig(confusion_figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    model_path = OUTPUT_DIR / "svm_best.joblib"
    best_row = results_df.iloc[0]
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": best_model,
            "algorithm": "Linear SVM",
            "configuration": f"C={best_C:g}",
            "C": best_C,
            "train_accuracy": float(best_row["train_accuracy"]),
            "train_macro_f1": float(best_row["train_macro_f1"]),
            "validation_accuracy": float(best_row["val_accuracy"]),
            "validation_macro_f1": float(best_row["val_macro_f1"]),
            "selection_metric": "val_macro_f1",
            "loss_type": best_row["loss_type"],
            "train_loss": float(best_row["train_loss"]),
            "validation_loss": float(best_row["val_loss"]),
            "fit_seconds": float(best_row["fit_seconds"]),
            "support_vectors": int(best_row["support_vectors"]),
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

    # 6. 绘制 C 与验证指标的关系。C 使用对数坐标。
    plot_df = results_df.sort_values("C")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(plot_df["C"], plot_df["val_macro_f1"], marker="o", label="Macro-F1")
    axis.plot(plot_df["C"], plot_df["val_accuracy"], marker="s", label="Accuracy")
    axis.set_xscale("log")
    axis.set_xlabel("C (log scale)")
    axis.set_ylabel("Validation score")
    axis.set_title("Linear SVM Hyperparameter Comparison")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    comparison_figure_path = OUTPUT_DIR / "svm_score_comparison.png"
    figure.savefig(comparison_figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    # 7. 输出便于实验报告直接引用的 SVM 摘要。
    print("=" * 70)
    print("--- 线性 SVM 参数对照实验完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n最佳 C: {best_C:g}")
    print(f"最佳验证 Accuracy: {results_df.iloc[0]['val_accuracy']:.4f}")
    print(f"最佳验证 Macro-F1: {results_df.iloc[0]['val_macro_f1']:.4f}")
    print(f"SVM 对照表: {results_path}")
    print(f"最佳模型分类报告: {report_path}")
    print(f"最佳模型混淆矩阵: {confusion_figure_path}")
    print(f"最佳验证阶段模型: {model_path}")


if __name__ == "__main__":
    main()

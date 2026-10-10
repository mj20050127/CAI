"""Compare raw text and P1 cleaning with the selected linear SVM.

P0 keeps each complete raw email. P1 keeps only Subject and body, using the
same deterministic parser as the MLP preprocessing experiment. All other
conditions are fixed: split seed 42, 20,000 unigram TF-IDF features, and
linear SVC with C=500. The official test set is not loaded during tuning.
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

from mlp_preprocessing_experiment import extract_subject_and_body


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLIT_RANDOM_STATE = 42
MODEL_RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
FIXED_C = 500.0
MAX_FEATURES = 20000
PRACTICAL_TIE_TOLERANCE = 0.001
OUTPUT_DIR = Path("outputs") / "svm_preprocessing_p1"

CONFIGS = [
    {
        "name": "P0_raw",
        "description": "Raw email text",
        "preprocessing": "raw",
    },
    {
        "name": "P1_subject_body",
        "description": "Keep Subject and body; remove other headers",
        "preprocessing": "subject_body",
    },
]

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


def transform_texts(texts, preprocessing):
    if preprocessing == "raw":
        return list(texts)
    if preprocessing == "subject_body":
        return [extract_subject_and_body(text) for text in texts]
    raise ValueError(f"Unknown preprocessing mode: {preprocessing}")


def build_model():
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


def text_statistics(texts, config_name, split_name):
    character_counts = np.asarray([len(text) for text in texts], dtype=float)
    word_counts = np.asarray([len(text.split()) for text in texts], dtype=float)
    return {
        "config": config_name,
        "split": split_name,
        "samples": len(texts),
        "empty_texts": sum(not text.strip() for text in texts),
        "mean_characters": float(character_counts.mean()),
        "median_characters": float(np.median(character_counts)),
        "mean_words": float(word_counts.mean()),
        "median_words": float(np.median(word_counts)),
    }


def save_comparison_plot(results_df, selected_name):
    plot_df = results_df.sort_values("config_order")
    positions = np.arange(len(plot_df))
    selected_position = int(
        np.flatnonzero(plot_df["config"].to_numpy() == selected_name)[0]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(
        positions,
        plot_df["val_macro_f1"],
        marker="o",
        label="Validation Macro-F1",
    )
    axes[0].plot(
        positions,
        plot_df["val_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )
    axes[0].axvline(
        selected_position,
        color="tab:red",
        linestyle="--",
        label=f"Selected {selected_name}",
    )
    axes[0].set_xticks(positions, plot_df["config"], rotation=12)
    axes[0].set_ylabel("Validation score")
    axes[0].set_title("SVM P1 Preprocessing Scores")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        positions,
        plot_df["train_hinge_loss"],
        marker="o",
        label="Training hinge loss",
    )
    axes[1].plot(
        positions,
        plot_df["val_hinge_loss"],
        marker="s",
        label="Validation hinge loss",
    )
    axes[1].axvline(
        selected_position,
        color="tab:red",
        linestyle="--",
        label=f"Selected {selected_name}",
    )
    axes[1].set_xticks(positions, plot_df["config"], rotation=12)
    axes[1].set_ylabel("Multiclass hinge loss")
    axes[1].set_title("SVM P1 Preprocessing Hinge Loss")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "svm_p1_preprocessing_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_selected_outputs(matrix, classes, selected_name):
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / "svm_p1_selected_confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"Selected SVM Preprocessing ({selected_name})")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "svm_p1_selected_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_texts, all_targets = load_labeled_data()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        all_texts,
        all_targets,
        test_size=VALIDATION_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=all_targets,
    )
    classes = np.sort(np.unique(y_train))

    print("--- SVM P1数据清洗对照实验 ---")
    print(f"训练样本: {len(X_train_raw)}")
    print(f"验证样本: {len(X_val_raw)}")
    print(f"配置: {[config['name'] for config in CONFIGS]}")
    print(f"固定 C: {FIXED_C:g}")
    print(f"固定 TF-IDF max_features: {MAX_FEATURES}")
    print("固定 ngram_range: (1, 1)")
    print("选择指标: validation Macro-F1")
    print("官方测试集: 未加载")
    print("=" * 76)

    results = []
    statistics = []
    trained_models = {}
    fitted_vectorizers = {}
    processed_validation_sets = {}

    for config_order, config in enumerate(CONFIGS, start=1):
        config_name = config["name"]
        print(
            f"--- 配置 {config_order}/{len(CONFIGS)}: "
            f"{config_name} ---"
        )
        X_train = transform_texts(X_train_raw, config["preprocessing"])
        X_val = transform_texts(X_val_raw, config["preprocessing"])
        statistics.append(text_statistics(X_train, config_name, "train"))
        statistics.append(text_statistics(X_val, config_name, "validation"))

        train_empty_count = sum(not text.strip() for text in X_train)
        val_empty_count = sum(not text.strip() for text in X_val)
        if train_empty_count or val_empty_count:
            raise ValueError(
                f"{config_name} produced empty texts: "
                f"train={train_empty_count}, validation={val_empty_count}"
            )

        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
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
            "config_order": config_order,
            "config": config_name,
            "description": config["description"],
            "preprocessing": config["preprocessing"],
            "C": FIXED_C,
            "max_features": MAX_FEATURES,
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
        trained_models[config_name] = model
        fitted_vectorizers[config_name] = vectorizer
        processed_validation_sets[config_name] = X_val

        print(
            f"{config_name} | train_hinge={row['train_hinge_loss']:.6f} | "
            f"val_hinge={row['val_hinge_loss']:.6f} | "
            f"train_acc={row['train_accuracy']:.6f} | "
            f"val_acc={row['val_accuracy']:.6f} | "
            f"train_macro_f1={row['train_macro_f1']:.6f} | "
            f"val_macro_f1={row['val_macro_f1']:.6f} | "
            f"support_vectors={row['support_vectors']} | "
            f"fit_seconds={fit_seconds:.2f}"
        )

    pd.DataFrame(statistics).to_csv(
        OUTPUT_DIR / "svm_p1_text_statistics.csv",
        index=False,
    )
    results_df = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "val_accuracy", "val_hinge_loss", "config_order"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    results_df.to_csv(
        OUTPUT_DIR / "svm_p1_preprocessing_results.csv",
        index=False,
    )

    strict_best_row = results_df.iloc[0]
    strict_best_name = str(strict_best_row["config"])
    best_score = float(strict_best_row["val_macro_f1"])
    practical_ties = results_df[
        results_df["val_macro_f1"] >= best_score - PRACTICAL_TIE_TOLERANCE
    ].copy()
    recommended_row = practical_ties.sort_values(
        by=["config_order", "val_macro_f1", "val_accuracy", "val_hinge_loss"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).iloc[0]
    selected_name = str(recommended_row["config"])
    selected_model = trained_models[selected_name]
    selected_vectorizer = fitted_vectorizers[selected_name]
    X_val_selected = selected_vectorizer.transform(
        processed_validation_sets[selected_name]
    )
    selected_predictions = selected_model.predict(X_val_selected)

    save_comparison_plot(results_df, selected_name)
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
        OUTPUT_DIR / "svm_p1_selected_classification_report.csv"
    )
    matrix = confusion_matrix(y_val, selected_predictions, labels=classes)
    save_selected_outputs(matrix, classes, selected_name)

    selected_config = next(
        config for config in CONFIGS if config["name"] == selected_name
    )
    joblib.dump(
        {
            "vectorizer": selected_vectorizer,
            "model": selected_model,
            "algorithm": "Linear SVM (SVC, linear kernel)",
            "experiment": "P0_vs_P1_preprocessing",
            "preprocessing_config": selected_config,
            "strict_best_config": strict_best_name,
            "C": FIXED_C,
            "max_features": MAX_FEATURES,
            "validation_accuracy": float(recommended_row["val_accuracy"]),
            "validation_macro_f1": float(recommended_row["val_macro_f1"]),
            "validation_hinge_loss": float(
                recommended_row["val_hinge_loss"]
            ),
            "support_vectors": int(recommended_row["support_vectors"]),
            "split_random_state": SPLIT_RANDOM_STATE,
            "model_random_state": MODEL_RANDOM_STATE,
            "validation_size": VALIDATION_SIZE,
            "tfidf_config": {
                "max_features": MAX_FEATURES,
                "lowercase": True,
                "ngram_range": (1, 1),
                "fit_on_training_subset_only": True,
            },
            "class_names": CLASS_NAMES,
        },
        OUTPUT_DIR / "svm_p1_selected.joblib",
    )

    decision = {
        "strict_best_config": strict_best_name,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "strict_best_val_accuracy": float(strict_best_row["val_accuracy"]),
        "strict_best_val_hinge_loss": float(
            strict_best_row["val_hinge_loss"]
        ),
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "recommended_config": selected_name,
        "recommended_val_macro_f1": float(
            recommended_row["val_macro_f1"]
        ),
        "recommended_val_accuracy": float(recommended_row["val_accuracy"]),
        "recommended_val_hinge_loss": float(
            recommended_row["val_hinge_loss"]
        ),
        "recommendation_rule": (
            "Choose raw text when it is within 0.001 Macro-F1 of P1; "
            "otherwise choose the strict best."
        ),
    }
    with (OUTPUT_DIR / "svm_p1_decision.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    settings = {
        "experiment": "P0_vs_P1_preprocessing",
        "configurations": CONFIGS,
        "fixed_C": FIXED_C,
        "max_features": MAX_FEATURES,
        "ngram_range": [1, 1],
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
        "primary_selection_metric": "val_macro_f1",
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "official_test_set_used": False,
        "tfidf_fit_on_training_subset_only": True,
    }
    with (OUTPUT_DIR / "experiment_settings.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("--- SVM P1数据清洗对照实验完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n严格最高 Macro-F1 配置: {strict_best_name}")
    print(f"严格最高验证 Macro-F1: {strict_best_row['val_macro_f1']:.6f}")
    print(f"0.001 等价阈值下推荐配置: {selected_name}")
    print(f"推荐配置验证 Macro-F1: {recommended_row['val_macro_f1']:.6f}")
    print(f"推荐配置验证 Accuracy: {recommended_row['val_accuracy']:.6f}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

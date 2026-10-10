"""Controlled MLP experiment for three practical text-cleaning steps.

P0: raw email text.
P1: keep Subject and body; remove other email headers.
P2: P1 plus removal of quoted reply lines and standard signatures.
P3: P2 plus English stop-word removal in TF-IDF.

Only labeled data are loaded. The official test set remains sealed while the
cleaning strategy is selected on the validation set.
"""

from copy import deepcopy
from email import policy
from email.parser import Parser
from pathlib import Path
from time import perf_counter
import json
import re

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
MAX_FEATURES = 20000
HIDDEN_LAYER_SIZES = (100,)
ALPHA = 0.01
LEARNING_RATE_INIT = 0.0005
MINI_BATCH_SIZE = 128
MAX_EPOCHS = 80
PRACTICAL_TIE_TOLERANCE = 0.001
OUTPUT_DIR = Path("outputs") / "mlp_preprocessing_comparison"

CONFIGS = [
    {
        "name": "P0_raw",
        "description": "Raw email text",
        "text_mode": "raw",
        "remove_quotes_and_signature": False,
        "stop_words": None,
    },
    {
        "name": "P1_subject_body",
        "description": "Keep Subject and body; remove other headers",
        "text_mode": "subject_body",
        "remove_quotes_and_signature": False,
        "stop_words": None,
    },
    {
        "name": "P2_subject_body_clean_reply",
        "description": "P1 plus quoted-line and signature removal",
        "text_mode": "subject_body",
        "remove_quotes_and_signature": True,
        "stop_words": None,
    },
    {
        "name": "P3_clean_reply_stopwords",
        "description": "P2 plus English stop-word removal",
        "text_mode": "subject_body",
        "remove_quotes_and_signature": True,
        "stop_words": "english",
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

SIGNATURE_SEPARATOR = re.compile(r"^\s*--\s*$")
QUOTED_LINE = re.compile(r"^\s*>+")
WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


def load_labeled_data():
    train_df = pd.read_csv("train_data.csv")
    texts = train_df["text"].astype(str).tolist()
    targets = train_df["target"].to_numpy()
    return texts, targets


def fallback_body(raw_text):
    """Return text after the first blank line when email parsing is incomplete."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", normalized, maxsplit=1)
    return parts[1] if len(parts) == 2 else normalized


def extract_subject_and_body(raw_text):
    """Extract the topical Subject field and readable plain-text body."""
    try:
        message = Parser(policy=policy.default).parsestr(raw_text)
        subject = str(message.get("Subject", "")).strip()
        body_parts = []

        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() != "text/plain":
                    continue
                if part.get_content_disposition() == "attachment":
                    continue
                try:
                    content = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True)
                    content = (
                        payload.decode("utf-8", errors="replace")
                        if isinstance(payload, bytes)
                        else str(payload or "")
                    )
                body_parts.append(str(content))
            body = "\n".join(body_parts)
        else:
            try:
                body = str(message.get_content())
            except Exception:
                payload = message.get_payload(decode=True)
                body = (
                    payload.decode("utf-8", errors="replace")
                    if isinstance(payload, bytes)
                    else str(payload or "")
                )

        if not body.strip():
            body = fallback_body(raw_text)
    except Exception:
        subject = ""
        body = fallback_body(raw_text)

    subject = WHITESPACE.sub(" ", subject).strip()
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if subject:
        return f"Subject: {subject}\n\n{body}".strip()
    return body


def remove_quotes_and_signature(text):
    """Remove explicit reply quotations and content after a standard signature."""
    kept_lines = []
    for line in text.splitlines():
        if SIGNATURE_SEPARATOR.match(line):
            break
        if QUOTED_LINE.match(line):
            continue
        kept_lines.append(WHITESPACE.sub(" ", line).rstrip())
    cleaned = "\n".join(kept_lines).strip()
    return BLANK_LINES.sub("\n\n", cleaned)


def preprocess_texts(texts, config):
    if config["text_mode"] == "raw":
        processed = list(texts)
    else:
        processed = [extract_subject_and_body(text) for text in texts]

    if config["remove_quotes_and_signature"]:
        processed = [remove_quotes_and_signature(text) for text in processed]
    return processed


def text_statistics(texts, split_name, config_name):
    char_counts = np.asarray([len(text) for text in texts], dtype=float)
    word_counts = np.asarray([len(text.split()) for text in texts], dtype=float)
    empty_count = sum(not text.strip() for text in texts)
    return {
        "config": config_name,
        "split": split_name,
        "samples": len(texts),
        "empty_texts": empty_count,
        "mean_characters": float(char_counts.mean()),
        "median_characters": float(np.median(char_counts)),
        "mean_words": float(word_counts.mean()),
        "median_words": float(np.median(word_counts)),
    }


def save_sample_comparison(raw_sample, transformed_samples):
    path = OUTPUT_DIR / "preprocessing_sample_comparison.txt"
    with path.open("w", encoding="utf-8") as file:
        file.write("=== ORIGINAL RAW SAMPLE ===\n")
        file.write(raw_sample)
        file.write("\n\n")
        for config, sample in zip(CONFIGS, transformed_samples):
            file.write(f"=== {config['name']}: {config['description']} ===\n")
            file.write(sample)
            file.write("\n\n")


def plot_epoch_loss(history_df, config_name, best_epoch):
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
    axis.set_title(f"MLP Preprocessing: {config_name}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / f"{config_name}_loss.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_minibatch_loss(minibatch_df, config_name, batches_per_epoch):
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
    axis.set_title(f"MLP Mini-batch Loss: {config_name}")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / f"{config_name}_minibatch_loss.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_score_plot(results_df, selected_name):
    plot_df = results_df.sort_values("config_order")
    positions = np.arange(len(plot_df))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        positions,
        plot_df["val_macro_f1"],
        marker="o",
        label="Validation Macro-F1",
    )
    axis.plot(
        positions,
        plot_df["val_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )
    selected_position = int(
        np.flatnonzero(plot_df["config"].to_numpy() == selected_name)[0]
    )
    axis.axvline(
        selected_position,
        color="tab:red",
        linestyle="--",
        label=f"Selected {selected_name}",
    )
    axis.set_xticks(positions, plot_df["config"], rotation=15, ha="right")
    axis.set_ylabel("Validation score")
    axis.set_title("MLP Text Preprocessing Comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mlp_preprocessing_score_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_outputs(matrix, classes, selected_name):
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        OUTPUT_DIR / "mlp_preprocessing_selected_confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"Selected Preprocessing Confusion Matrix ({selected_name})")
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "mlp_preprocessing_selected_confusion_matrix.png",
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
    batches_per_epoch = int(np.ceil(len(y_train) / MINI_BATCH_SIZE))

    print("--- MLP 数据清洗对照实验 ---")
    print(f"训练样本: {len(X_train_raw)}")
    print(f"验证样本: {len(X_val_raw)}")
    print(f"配置: {[config['name'] for config in CONFIGS]}")
    print(f"固定 TF-IDF max_features: {MAX_FEATURES}")
    print(f"固定隐藏层: {HIDDEN_LAYER_SIZES}")
    print(f"固定 alpha: {ALPHA}")
    print(f"固定 learning_rate_init: {LEARNING_RATE_INIT}")
    print(f"固定 mini-batch size: {MINI_BATCH_SIZE}")
    print(f"最大 epoch: {MAX_EPOCHS}")
    print("选择指标: validation Macro-F1")
    print("官方测试集: 未加载")
    print("=" * 80)

    results = []
    statistics = []
    best_models = {}
    fitted_vectorizers = {}
    processed_validation_sets = {}
    first_samples = []

    for config_order, config in enumerate(CONFIGS, start=1):
        config_name = config["name"]
        print(
            f"--- 配置 {config_order}/{len(CONFIGS)}: "
            f"{config_name} ---"
        )

        X_train = preprocess_texts(X_train_raw, config)
        X_val = preprocess_texts(X_val_raw, config)
        first_samples.append(preprocess_texts([all_texts[0]], config)[0])
        statistics.append(text_statistics(X_train, "train", config_name))
        statistics.append(text_statistics(X_val, "validation", config_name))

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
            stop_words=config["stop_words"],
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
                "cumulative_fit_seconds": cumulative_fit_seconds,
            }
            history.append(row)

            if row["val_macro_f1"] > config_best_val_macro_f1:
                config_best_val_macro_f1 = row["val_macro_f1"]
                config_best_epoch = epoch
                config_best_model = deepcopy(model)

            print(
                f"{config_name} | epoch={epoch:03d}/{MAX_EPOCHS} | "
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
        plot_epoch_loss(history_df, config_name, config_best_epoch)
        plot_minibatch_loss(minibatch_df, config_name, batches_per_epoch)

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
                "config_order": config_order,
                "config": config_name,
                "description": config["description"],
                "text_mode": config["text_mode"],
                "remove_quotes_and_signature": config[
                    "remove_quotes_and_signature"
                ],
                "stop_words": str(config["stop_words"]),
                "max_features": MAX_FEATURES,
                "actual_feature_count": actual_feature_count,
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
        best_models[config_name] = config_best_model
        fitted_vectorizers[config_name] = vectorizer
        processed_validation_sets[config_name] = X_val

    save_sample_comparison(all_texts[0], first_samples)
    pd.DataFrame(statistics).to_csv(
        OUTPUT_DIR / "preprocessing_text_statistics.csv",
        index=False,
    )

    results_df = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "val_accuracy", "val_loss", "config_order"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    results_df.to_csv(
        OUTPUT_DIR / "mlp_preprocessing_results.csv",
        index=False,
    )

    strict_best_row = results_df.iloc[0]
    strict_best_name = str(strict_best_row["config"])
    best_score = float(strict_best_row["val_macro_f1"])
    practical_ties = results_df[
        results_df["val_macro_f1"] >= best_score - PRACTICAL_TIE_TOLERANCE
    ].copy()
    recommended_row = practical_ties.sort_values(
        by=["config_order", "val_macro_f1", "val_accuracy", "val_loss"],
        ascending=[True, False, False, True],
        ignore_index=True,
    ).iloc[0]
    selected_name = str(recommended_row["config"])
    selected_model = best_models[selected_name]
    selected_vectorizer = fitted_vectorizers[selected_name]
    selected_val_texts = processed_validation_sets[selected_name]
    X_val_selected = selected_vectorizer.transform(selected_val_texts)
    selected_predictions = selected_model.predict(X_val_selected)

    save_score_plot(results_df, selected_name)
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
        OUTPUT_DIR / "mlp_preprocessing_selected_classification_report.csv"
    )
    matrix = confusion_matrix(y_val, selected_predictions, labels=classes)
    save_confusion_outputs(matrix, classes, selected_name)

    selected_config = next(
        config for config in CONFIGS if config["name"] == selected_name
    )
    joblib.dump(
        {
            "vectorizer": selected_vectorizer,
            "model": selected_model,
            "algorithm": "MLP",
            "experiment": "text_preprocessing_comparison",
            "preprocessing_config": selected_config,
            "strict_best_config": strict_best_name,
            "max_features": MAX_FEATURES,
            "hidden_layer_sizes": HIDDEN_LAYER_SIZES,
            "alpha": ALPHA,
            "learning_rate_init": LEARNING_RATE_INIT,
            "mini_batch_size": MINI_BATCH_SIZE,
            "best_epoch": int(recommended_row["best_epoch"]),
            "validation_accuracy": float(recommended_row["val_accuracy"]),
            "validation_macro_f1": float(recommended_row["val_macro_f1"]),
            "validation_loss": float(recommended_row["val_loss"]),
            "split_random_state": SPLIT_RANDOM_STATE,
            "model_random_state": MODEL_RANDOM_STATE,
            "batch_random_state": BATCH_RANDOM_STATE,
            "class_names": CLASS_NAMES,
        },
        OUTPUT_DIR / "mlp_preprocessing_selected.joblib",
    )

    decision = {
        "strict_best_config": strict_best_name,
        "strict_best_val_macro_f1": float(strict_best_row["val_macro_f1"]),
        "strict_best_val_accuracy": float(strict_best_row["val_accuracy"]),
        "strict_best_val_loss": float(strict_best_row["val_loss"]),
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "recommended_config": selected_name,
        "recommended_val_macro_f1": float(
            recommended_row["val_macro_f1"]
        ),
        "recommended_val_accuracy": float(recommended_row["val_accuracy"]),
        "recommended_val_loss": float(recommended_row["val_loss"]),
        "recommendation_rule": (
            "Choose the simplest earlier configuration within 0.001 "
            "Macro-F1 of the strict best."
        ),
    }
    with (OUTPUT_DIR / "preprocessing_decision.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)

    settings = {
        "experiment": "text_preprocessing_comparison",
        "configurations": CONFIGS,
        "validation_size": VALIDATION_SIZE,
        "split_random_state": SPLIT_RANDOM_STATE,
        "model_random_state": MODEL_RANDOM_STATE,
        "batch_random_state": BATCH_RANDOM_STATE,
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "alpha": ALPHA,
        "learning_rate_init": LEARNING_RATE_INIT,
        "mini_batch_size": MINI_BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "primary_selection_metric": "val_macro_f1",
        "practical_tie_tolerance": PRACTICAL_TIE_TOLERANCE,
        "official_test_set_used": False,
        "tfidf": {
            "max_features": MAX_FEATURES,
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

    print("=" * 80)
    print("--- 数据清洗对照实验完成 ---")
    print(results_df.to_string(index=False))
    print(f"\n严格最高 Macro-F1 配置: {strict_best_name}")
    print(f"严格最高验证 Macro-F1: {strict_best_row['val_macro_f1']:.6f}")
    print(f"0.001 等价阈值下推荐配置: {selected_name}")
    print(f"推荐配置验证 Macro-F1: {recommended_row['val_macro_f1']:.6f}")
    print(f"推荐配置验证 Accuracy: {recommended_row['val_accuracy']:.6f}")
    print(f"结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

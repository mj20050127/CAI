"""Repeat the selected MLP and linear SVM experiment across data-split seeds.

The official unlabeled test set is never loaded.  For each seed, the labeled
data are split independently with stratification, TF-IDF is fitted only on the
training subset, and both selected models are evaluated on the same validation
subset.

Run in Anaconda Prompt:

    conda activate project1
    cd /d C:\\Users\\20921\\Desktop\\当代人工智能\\实验一\\data-Project1
    python repeated_seed_experiment.py
"""

from email import policy
from email.parser import Parser
from pathlib import Path
from time import perf_counter
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


SEEDS = (42, 52, 62)
VALIDATION_SIZE = 0.2
MAX_FEATURES = 20_000

MLP_HIDDEN_LAYER_SIZES = (100,)
MLP_ALPHA = 0.01
MLP_LEARNING_RATE_INIT = 0.0005
MLP_BATCH_SIZE = 128
MLP_EPOCHS = 39

SVM_C = 500.0

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "train_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "repeated_seeds"

WHITESPACE = re.compile(r"[ \t]+")


def fallback_body(raw_text):
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", normalized, maxsplit=1)
    return parts[1] if len(parts) == 2 else normalized


def extract_subject_and_body(raw_text):
    """Apply P1 preprocessing: keep Subject and readable plain-text body."""
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
    return f"Subject: {subject}\n\n{body}".strip() if subject else body


def train_mlp(X_train, y_train, X_val, y_val, seed, classes):
    model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=MLP_ALPHA,
        batch_size="auto",
        learning_rate_init=MLP_LEARNING_RATE_INIT,
        random_state=seed,
        shuffle=False,
    )
    random_generator = np.random.RandomState(seed)
    first_update = True
    fit_start = perf_counter()

    for epoch in range(1, MLP_EPOCHS + 1):
        shuffled_indices = random_generator.permutation(len(y_train))
        for start_index in range(0, len(y_train), MLP_BATCH_SIZE):
            batch_indices = shuffled_indices[
                start_index : start_index + MLP_BATCH_SIZE
            ]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            if first_update:
                model.partial_fit(X_batch, y_batch, classes=classes)
                first_update = False
            else:
                model.partial_fit(X_batch, y_batch)
        print(
            f"seed={seed} | MLP epoch={epoch:02d}/{MLP_EPOCHS}",
            flush=True,
        )

    fit_seconds = perf_counter() - fit_start
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    return {
        "algorithm": "MLP",
        "seed": seed,
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "train_macro_f1": f1_score(
            y_train, train_predictions, average="macro"
        ),
        "val_accuracy": accuracy_score(y_val, val_predictions),
        "val_macro_f1": f1_score(y_val, val_predictions, average="macro"),
        "fit_seconds": fit_seconds,
    }


def train_svm(X_train, y_train, X_val, y_val, seed):
    model = SVC(
        kernel="linear",
        C=SVM_C,
        probability=False,
        tol=0.001,
        max_iter=-1,
        random_state=seed,
    )
    fit_start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - fit_start
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    return {
        "algorithm": "Linear SVM",
        "seed": seed,
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "train_macro_f1": f1_score(
            y_train, train_predictions, average="macro"
        ),
        "val_accuracy": accuracy_score(y_val, val_predictions),
        "val_macro_f1": f1_score(y_val, val_predictions, average="macro"),
        "fit_seconds": fit_seconds,
    }


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    raw_texts = train_df["text"].astype(str).tolist()
    targets = train_df["target"].to_numpy()
    classes = np.sort(np.unique(targets))

    print("正在执行P1文本预处理……", flush=True)
    processed_texts = [extract_subject_and_body(text) for text in raw_texts]
    empty_count = sum(not text.strip() for text in processed_texts)
    if empty_count:
        raise ValueError(f"P1 preprocessing produced {empty_count} empty texts")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in SEEDS:
        X_train_text, X_val_text, y_train, y_val = train_test_split(
            processed_texts,
            targets,
            test_size=VALIDATION_SIZE,
            random_state=seed,
            stratify=targets,
        )

        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            lowercase=True,
            ngram_range=(1, 1),
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)
        print(
            f"seed={seed} | train={X_train.shape} | val={X_val.shape}",
            flush=True,
        )

        mlp_result = train_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            seed,
            classes,
        )
        mlp_result.update(
            {
                "validation_size": VALIDATION_SIZE,
                "max_features": MAX_FEATURES,
                "configuration": (
                    "P1; hidden=(100,); alpha=0.01; "
                    "lr=0.0005; batch=128; epochs=39"
                ),
            }
        )
        results.append(mlp_result)
        print(
            f"seed={seed} | MLP val Macro-F1="
            f"{mlp_result['val_macro_f1']:.6f}",
            flush=True,
        )

        svm_result = train_svm(X_train, y_train, X_val, y_val, seed)
        svm_result.update(
            {
                "validation_size": VALIDATION_SIZE,
                "max_features": MAX_FEATURES,
                "configuration": "P1; linear kernel; C=500",
            }
        )
        results.append(svm_result)
        print(
            f"seed={seed} | SVM val Macro-F1="
            f"{svm_result['val_macro_f1']:.6f}",
            flush=True,
        )

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "repeated_seed_results.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby("algorithm")
        .agg(
            runs=("seed", "count"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
        )
        .reset_index()
    )
    summary_path = OUTPUT_DIR / "repeated_seed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 80)
    print("三随机种子重复实验完成")
    print(results_df.to_string(index=False))
    print("\n均值与标准差：")
    print(summary_df.to_string(index=False))
    print(f"\n逐次结果: {results_path}")
    print(f"汇总结果: {summary_path}")
    print("官方无标签测试集未加载。")


if __name__ == "__main__":
    main()

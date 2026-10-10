r"""Train the selected final MLP on all labeled data and predict the test set.

The model configuration was selected using the fixed validation split in
``mlp_preprocessing_experiment.py``.  The official unlabeled test set is only
transformed and predicted after all choices have been fixed.

Run in Anaconda Prompt:

    conda activate project1
    cd /d C:\Users\20921\Desktop\当代人工智能\实验一\data-Project1
    python final_predict_mlp.py

The output ``predictions.csv`` contains 2,457 integer labels in one column,
without a header, matching the supplied example format.
"""

from email import policy
from email.parser import Parser
from pathlib import Path
from time import perf_counter
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


RANDOM_STATE = 42
MAX_FEATURES = 20_000
HIDDEN_LAYER_SIZES = (100,)
ALPHA = 0.01
LEARNING_RATE_INIT = 0.0005
MINI_BATCH_SIZE = 128
DEFAULT_EPOCHS = 39

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "train_data.csv"
TEST_PATH = PROJECT_DIR / "test_data_unlabeled.csv"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "predictions.csv"

WHITESPACE = re.compile(r"[ \t]+")


def fallback_body(raw_text):
    """Return the content after the first blank line if parsing is incomplete."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", normalized, maxsplit=1)
    return parts[1] if len(parts) == 2 else normalized


def extract_subject_and_body(raw_text):
    """Apply the selected P1 preprocessing: keep Subject and plain-text body."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain the selected MLP on all labels and predict test data."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of full-data training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Prediction CSV path (default: predictions.csv in the project folder).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer")

    start_time = perf_counter()
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required_train_columns = {"text", "target"}
    if not required_train_columns.issubset(train_df.columns):
        raise ValueError(
            f"{TRAIN_PATH.name} must contain columns: "
            f"{sorted(required_train_columns)}"
        )
    if "text" not in test_df.columns:
        raise ValueError(f"{TEST_PATH.name} must contain a text column")

    raw_train_texts = train_df["text"].astype(str).tolist()
    raw_test_texts = test_df["text"].astype(str).tolist()
    y_train = train_df["target"].to_numpy()
    classes = np.sort(np.unique(y_train))

    print("--- 最终 MLP 测试集预测 ---")
    print(f"全部有标签样本: {len(raw_train_texts)}")
    print(f"无标签测试样本: {len(raw_test_texts)}")
    print("预处理: P1（保留 Subject 和正文，删除其他邮件头）")
    print(f"训练轮数: {args.epochs}")
    print("正在进行 P1 文本预处理……", flush=True)

    train_texts = [extract_subject_and_body(text) for text in raw_train_texts]
    test_texts = [extract_subject_and_body(text) for text in raw_test_texts]
    train_empty_count = sum(not text.strip() for text in train_texts)
    test_empty_count = sum(not text.strip() for text in test_texts)
    if train_empty_count or test_empty_count:
        raise ValueError(
            "P1 preprocessing produced empty text: "
            f"train={train_empty_count}, test={test_empty_count}"
        )

    # Fit vocabulary and IDF only on labeled data.  The test set is transformed
    # with the fitted vectorizer and never affects feature construction.
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        lowercase=True,
        ngram_range=(1, 1),
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    vocabulary_size = len(vectorizer.vocabulary_)

    assert X_train.shape == (len(train_texts), vocabulary_size)
    assert X_test.shape == (len(test_texts), vocabulary_size)
    print(f"训练集 TF-IDF: {X_train.shape}")
    print(f"测试集 TF-IDF: {X_test.shape}")

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        alpha=ALPHA,
        batch_size="auto",
        learning_rate_init=LEARNING_RATE_INIT,
        random_state=RANDOM_STATE,
        shuffle=False,
    )

    random_generator = np.random.RandomState(RANDOM_STATE)
    first_update = True
    for epoch in range(1, args.epochs + 1):
        shuffled_indices = random_generator.permutation(len(y_train))
        for start_index in range(0, len(y_train), MINI_BATCH_SIZE):
            batch_indices = shuffled_indices[
                start_index : start_index + MINI_BATCH_SIZE
            ]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            if first_update:
                model.partial_fit(X_batch, y_batch, classes=classes)
                first_update = False
            else:
                model.partial_fit(X_batch, y_batch)

        print(f"Epoch {epoch:02d}/{args.epochs} 完成", flush=True)

    predictions = model.predict(X_test).astype(int)
    if len(predictions) != len(test_df):
        raise ValueError(
            f"Prediction count mismatch: {len(predictions)} != {len(test_df)}"
        )
    if not set(np.unique(predictions)).issubset(set(classes)):
        raise ValueError("Predictions contain labels absent from the training set")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(predictions, dtype="int64").to_csv(
        output_path,
        index=False,
        header=False,
        lineterminator="\n",
    )

    # Reopen the output to ensure its physical submission format is correct.
    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    invalid_lines = [line for line in output_lines if not re.fullmatch(r"[0-9]+", line)]
    if len(output_lines) != len(test_df) or invalid_lines:
        raise ValueError(
            "Invalid prediction CSV: "
            f"lines={len(output_lines)}, invalid_lines={len(invalid_lines)}"
        )

    elapsed_seconds = perf_counter() - start_time
    class_counts = pd.Series(predictions).value_counts().sort_index()
    print("=" * 60)
    print("预测完成")
    print(f"输出文件: {output_path}")
    print(f"预测行数: {len(predictions)}")
    print("预测类别分布:")
    print(class_counts.to_string())
    print(f"总用时: {elapsed_seconds:.1f} 秒")


if __name__ == "__main__":
    main()

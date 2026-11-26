"""分类指标：行表示真实类别，列表示预测类别，准确率取值为 0–1。"""

from collections.abc import Sequence

import numpy as np


def classification_metrics(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """从完整预测计算样本加权的总体准确率及每类指标。

    每类 accuracy 定义为该真实类别中预测正确的比例，即 recall。
    没有样本、没有预测或分母为零的类别指标记作 0，并保留 support。
    返回值只包含 Python 内置类型，可以直接保存为 JSON。
    """
    names = list(class_names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names 必须包含非空的类别名称。")
    if len(set(names)) != len(names):
        raise ValueError("class_names 中的类别名称必须唯一。")

    true = np.asarray(targets)
    pred = np.asarray(predictions)
    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("targets 和 predictions 必须是一维数组。")
    if true.size == 0 or true.size != pred.size:
        raise ValueError("真实标签与预测标签必须非空且数量相等。")
    if true.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise ValueError("标签必须是整数，不能使用浮点数、布尔值或概率。")

    num_classes = len(names)
    if np.any(true < 0) or np.any(true >= num_classes):
        raise ValueError("真实标签超出类别编号范围。")
    if np.any(pred < 0) or np.any(pred >= num_classes):
        raise ValueError("预测标签超出类别编号范围。")

    # bincount 一次累加所有样本，避免对每个 batch 的准确率简单平均。
    true = true.astype(np.int64, copy=False)
    pred = pred.astype(np.int64, copy=False)
    matrix = np.bincount(
        true * num_classes + pred, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    correct = matrix.diagonal()
    precision = np.divide(
        correct, predicted_count, out=np.zeros(num_classes, dtype=float), where=predicted_count != 0
    )
    recall = np.divide(correct, support, out=np.zeros(num_classes, dtype=float), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=float),
        where=precision + recall != 0,
    )

    return {
        "sample_count": int(true.size),
        "overall_accuracy": float(correct.sum() / true.size),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "name": name,
                "support": int(support[index]),
                "correct": int(correct[index]),
                "accuracy": float(recall[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(names)
        ],
        "confusion_matrix": matrix.tolist(),
    }

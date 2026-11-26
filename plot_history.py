"""步骤 4：根据 history.csv 单独生成训练、验证与学习率曲线。"""
import argparse
import csv
from pathlib import Path

from src.plots import plot_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    with (args.run_dir / "history.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("还没有完整 epoch 的记录")
    path = args.run_dir / "curves.png"
    plot_curves(rows, path)
    print(path.resolve())


if __name__ == "__main__":
    main()

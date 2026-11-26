"""步骤 2：显式下载数据、固定划分并计算训练集统计量。"""
import argparse
import json

from src.common import ROOT, load_config
from src.data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/e0.json"))
    parser.add_argument("--download", action="store_true", help="允许 torchvision 下载官方 CIFAR-10")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare(config, args.download)
    print(json.dumps({"classes": manifest["classes"], "mean": manifest["mean"], "std": manifest["std"],
                      "sizes": {k: len(v) for k, v in manifest["indices"].items()},
                      "manifest": config["data_dir"] + "/split_manifest.json"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

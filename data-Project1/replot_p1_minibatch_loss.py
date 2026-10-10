"""Redraw the existing P1 MLP loss records without training a model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import ScalarFormatter
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "mlp_preprocessing_comparison"
CONFIG_NAME = "P1_subject_body"
SMOOTHING_WINDOW = 10
ZOOM_EPOCHS = (35, 40)


def main():
    batches = pd.read_csv(OUTPUT_DIR / f"{CONFIG_NAME}_minibatch_history.csv")
    epochs = pd.read_csv(OUTPUT_DIR / f"{CONFIG_NAME}_history.csv")
    if batches["batch_loss"].isna().any() or (batches["batch_loss"] <= 0).any():
        raise ValueError("Log-scale plotting requires finite, positive batch losses.")
    best_row = epochs.loc[epochs["val_macro_f1"].idxmax()]
    best_epoch = int(best_row["epoch"])
    best_step = int(batches.loc[batches["epoch"] == best_epoch, "update_step"].max())
    zoom = batches[batches["epoch"].between(*ZOOM_EPOCHS)]
    if zoom.empty:
        raise ValueError("No batch records found for the zoom interval.")

    chinese_font = Path("C:/Windows/Fonts/msyh.ttc")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(chinese_font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11

    average = batches["batch_loss"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
    figure, axis = plt.subplots(figsize=(11, 6), layout="constrained")
    axis.plot(batches["update_step"], batches["batch_loss"], color="#577B99",
              linewidth=0.7, alpha=0.72, label="原始 mini-batch 损失", zorder=2)
    axis.plot(batches["update_step"], average, color="#D97716",
              linewidth=1.5, label=f"{SMOOTHING_WINDOW} 个 batch 移动平均", zorder=3)
    axis.set_yscale("log")
    axis.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.set_ylim(float(batches["batch_loss"].min()) * 0.85, 3)
    axis.set_xlim(0, int(batches["update_step"].max()))
    axis.set_xlabel("参数更新步数")
    axis.set_ylabel("更新后当前 batch 的交叉熵（对数刻度）")
    axis.set_title("P1 MLP 的 mini-batch 损失与局部波动", pad=12, fontsize=15)
    axis.grid(which="major", color="#D9E1E7", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.axvspan(int(zoom["update_step"].min()), int(zoom["update_step"].max()),
                 color="#E6EDF3", alpha=0.5, zorder=0)
    axis.axvline(best_step, color="#B84242", linestyle="--", linewidth=1.1)
    axis.annotate(f"第 {best_epoch} 轮结束\n最佳验证 Macro-F1 = {best_row['val_macro_f1']:.4f}",
                  xy=(best_step, 0.08), xytext=(0.53, 0.40), textcoords="axes fraction",
                  color="#A33535", fontsize=10,
                  arrowprops={"arrowstyle": "-", "color": "#B84242", "lw": 0.8})
    axis.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)

    inset = axis.inset_axes([0.55, 0.57, 0.42, 0.36])
    inset.plot(zoom["update_step"], zoom["batch_loss"], color="#577B99", linewidth=0.8)
    inset.plot(zoom["update_step"], average.loc[zoom.index], color="#D97716", linewidth=1.5)
    inset.axvline(best_step, color="#B84242", linestyle="--", linewidth=1)
    inset.set_xlim(int(zoom["update_step"].min()), int(zoom["update_step"].max()))
    padding = (zoom["batch_loss"].max() - zoom["batch_loss"].min()) * 0.1
    inset.set_ylim(max(0, zoom["batch_loss"].min() - padding), zoom["batch_loss"].max() + padding)
    inset.set_title("第 35–40 轮局部放大（普通刻度）", fontsize=10, pad=6)
    inset.set_xlabel("参数更新步数", fontsize=9, labelpad=2)
    inset.set_ylabel("交叉熵", fontsize=9, labelpad=2)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.3, linewidth=0.5)

    output = OUTPUT_DIR / f"{CONFIG_NAME}_minibatch_loss_improved.png"
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)
    print(f"Saved: {output}")
    print(f"Records: {len(batches)}, best epoch: {best_epoch}, best update step: {best_step}")


if __name__ == "__main__":
    main()

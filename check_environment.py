"""步骤 1：验证当前解释器、PyTorch/torchvision 和 GPU；不会下载或训练。"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="只检查 CPU 联调环境")
    args = parser.parse_args()
    print("Python:", sys.version)
    print("Interpreter:", sys.executable)
    try:
        import numpy
        import matplotlib
        import PIL
        import torch
        import torchvision
        from src.models import SimpleCNN
    except (ImportError, RuntimeError, OSError) as exc:
        print("依赖检查失败：", exc)
        print("请按 README 在 .venv 中安装相互匹配的 torch/torchvision 和其它依赖。")
        return 1
    print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
    print("numpy:", numpy.__version__, "matplotlib:", matplotlib.__version__, "Pillow:", PIL.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not args.cpu and not torch.cuda.is_available():
        print("GPU 检查未通过；CPU 版 PyTorch 不能使用 RTX 4060。请核对安装源和解释器。")
        return 1
    device = torch.device("cpu" if args.cpu else "cuda")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    with torch.inference_mode():
        model = SimpleCNN().to(device).eval()
        output = model(torch.randn(2, 3, 32, 32, device=device))
        assert output.shape == (2, 10)
    print("环境通过，模型输出形状为 (2, 10)。下一步可准备数据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

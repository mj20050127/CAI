"""训练和评估共用模型入口；旧 E0 checkpoint 默认对应 SimpleCNN。"""
import hashlib
import json
import torch

from src.common import resolve_model_name
from .simple_cnn import SimpleCNN

RESNET_RESIDUAL_PREFIXES = ("layer1.", "layer2.", "layer3.", "layer4.")


def build_model(config):
    name = resolve_model_name(config)
    if name == "simple_cnn":
        return SimpleCNN(dropout=config.get("dropout", 0.3))
    if name == "cifar_resnet18_cbam":
        from .cbam_resnet import CIFARCBAMResNet18
        return CIFARCBAMResNet18(dropout=config.get("dropout", 0.3),
                                 reduction=config["cbam_reduction"],
                                 spatial_kernel_size=config["cbam_spatial_kernel_size"])
    if name == "standard_resnet18":
        from .standard_resnet import build_standard_resnet18
        # Fresh E10 training loads ImageNet separately; saved models only need the architecture.
        return build_standard_resnet18()
    # E0 导入模型入口时不额外加载 torchvision 的模型实现。
    from .cifar_resnet import CIFARResNet18
    return CIFARResNet18(dropout=config.get("dropout", 0.3))


def model_name_from_checkpoint(checkpoint):
    name = resolve_model_name(checkpoint["config"])
    if checkpoint.get("model_name", name) != name:
        raise ValueError("checkpoint 的模型标识与其配置不一致")
    return name


def state_dict_sha256(state):
    """按参数名/形状/类型/字节计算内容哈希，不依赖 torch.save 的容器字节。"""
    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], ensure_ascii=False)
        hasher.update(metadata.encode("utf-8"))
        hasher.update(tensor.reshape(-1).view(dtype=torch.uint8).numpy().tobytes())
    return hasher.hexdigest()

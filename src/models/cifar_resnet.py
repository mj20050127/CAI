"""E1：随机初始化的 CIFAR ResNet-18，保留与 E0 相同的头部 Dropout。"""

import math

from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class CIFARResNet18(ResNet):
    """32×32 输入；四阶段残差块 [2, 2, 2, 2]；GAP 后执行 Dropout。

    本类直接构建网络结构，不调用预训练模型构建器，也不会下载权重。
    它是用于架构对照的 CIFAR 适配版本，并非原始 ImageNet 入口版本。
    """

    def __init__(self, dropout: float = 0.3, num_classes: int = 10) -> None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError("num_classes 必须是正整数。")
        if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("dropout 必须是 [0, 1) 区间内的有限数值。")

        # 残差分支的末层 BN 也初始化为 1，不额外引入零初始化技巧。
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2],
                         num_classes=num_classes, zero_init_residual=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        # 继承的 forward 已在 fc 前完成 GAP 与 flatten，无需复制残差前向逻辑。
        self.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(512, num_classes))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """对全部层（含替换后的入口与分类层）显式应用 E0 的初始化规则。"""
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

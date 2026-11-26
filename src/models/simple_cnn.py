"""面向 CIFAR-10 的六层卷积 CNN，不使用预训练权重。"""

from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """三个卷积块：32×32 → 16×16 → 8×8 → 4×4 → 十分类。"""

    def __init__(self, dropout: float = 0.3, num_classes: int = 10) -> None:
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError("num_classes 必须是正整数。")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        self.features = nn.Sequential(
            self._block(3, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))
        self._initialize_weights()

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        # 卷积后已有 BatchNorm 的偏移参数，所以卷积本身无需 bias。
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _initialize_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        features = self.pool(features).flatten(1)
        # 输出 logits。训练的 CrossEntropyLoss 已包含 log-softmax。
        return self.classifier(features)

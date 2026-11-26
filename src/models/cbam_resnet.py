"""E5: CIFAR ResNet-18 with CBAM on each residual branch."""

import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock

from .cifar_resnet import CIFARResNet18


class CBAM(nn.Module):
    """Apply channel attention followed by spatial attention."""

    def __init__(self, channels: int, reduction: int, spatial_kernel_size: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2, bias=False,
        )
        for layer in (self.channel_mlp[0], self.channel_mlp[2], self.spatial_conv):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        average = x.mean(dim=(2, 3), keepdim=True)
        # A single maximum per channel, with CUDA deterministic backward support.
        maximum = x.flatten(2).max(dim=2, keepdim=True).values.unsqueeze(-1)
        channel_weights = torch.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        features = x * channel_weights

        average = features.mean(dim=1, keepdim=True)
        maximum = features.max(dim=1, keepdim=True).values
        spatial_weights = torch.sigmoid(self.spatial_conv(torch.cat((average, maximum), dim=1)))
        return features * spatial_weights


class CBAMBasicBlock(nn.Module):
    """Reuse the original block's modules and insert CBAM before the residual sum."""

    def __init__(self, block: BasicBlock, reduction: int, spatial_kernel_size: int) -> None:
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.stride = block.stride
        self.cbam = CBAM(block.conv2.out_channels, reduction, spatial_kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.cbam(self.bn2(self.conv2(residual)))
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(residual + shortcut)


class CIFARCBAMResNet18(CIFARResNet18):
    """E1 backbone and classifier with eight channel/spatial attention modules."""

    def __init__(self, dropout: float = 0.3, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        if type(reduction) is not int or reduction < 1:
            raise ValueError("CBAM reduction 必须是正整数")
        if type(spatial_kernel_size) is not int or spatial_kernel_size < 1 or spatial_kernel_size % 2 == 0:
            raise ValueError("CBAM spatial_kernel_size 必须是正奇数")
        super().__init__(dropout=dropout)
        # Preserve the E1 construction's CPU RNG position for data augmentation.
        with torch.random.fork_rng(devices=[]):
            for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
                for index, block in enumerate(stage):
                    stage[index] = CBAMBasicBlock(block, reduction, spatial_kernel_size)

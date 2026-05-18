import torch
from torch import nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int, 
            padding: int=0, 
            kernels_per_layer: int=1
        ) -> None:
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels*kernels_per_layer, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels
        )

        self.pointwise = nn.Conv2d(
            in_channels=self.depthwise.out_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio=16) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels//reduction_ratio, in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scaled = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scaled


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int=7) -> None:
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scaled = x * torch.sigmoid(out)
        return scaled

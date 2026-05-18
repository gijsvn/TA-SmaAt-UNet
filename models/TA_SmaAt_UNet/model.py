import torch
from torch import nn
from models.SmaAt_UNet.blocks import DoubleConvDS, UpDS, DownDS, OutConv, CBAM
from models.TA_SmaAt_UNet.layers import TimeConditioning

class TA_SmaAt_UNet(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernels_per_layer: int=2, 
            bilinear: bool=True, 
            reduction_ratio: int=16
        ) -> None:
        super().__init__()

        self.name = "TA-SmaAt-UNet"
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        factor = 2 if self.bilinear else 1

        self.inc = DoubleConvDS(self.in_channels, 64, kernels_per_layer=kernels_per_layer)

        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.cond1 = TimeConditioning(num_channels=64)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.cond2 = TimeConditioning(num_channels=128)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.cond3 = TimeConditioning(num_channels=256)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        self.cond4 = TimeConditioning(num_channels=512)
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.cond5 = TimeConditioning(num_channels=1024 // factor)
        
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x1Cond = self.cond1(x1Att, t)

        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x2Cond = self.cond2(x2Att, t)

        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x3Cond = self.cond3(x3Att, t)
        
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x4Cond = self.cond4(x4Att, t)
        
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x5Cond = self.cond5(x5Att, t)

        x = self.up1(x5Cond, x4Cond)
        x = self.up2(x, x3Cond)
        x = self.up3(x, x2Cond)
        x = self.up4(x, x1Cond)

        logits = self.outc(x)

        return logits
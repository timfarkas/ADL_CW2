"""
AI usage statement:

AI was used to assist with researching and debugging, as well as helping
with creating docstrings. All code was writte, reviewed and/or modified by a human.
"""

import torch


class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)

        self.pool = torch.nn.MaxPool2d(2)

        self.middle = conv_block(base_ch * 4, base_ch * 4)

        self.up2 = torch.nn.ConvTranspose2d(
            base_ch * 4, base_ch * 2, kernel_size=2, stride=2
        )
        self.dec2 = conv_block(base_ch * 4 + base_ch * 2, base_ch * 2)

        self.up1 = torch.nn.ConvTranspose2d(
            base_ch * 2, base_ch, kernel_size=2, stride=2
        )
        self.dec1 = conv_block(base_ch * 2 + base_ch, base_ch)

        self.out = torch.nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.middle(self.pool(e3))

        d2 = self.up2(m)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)
        out = torch.nn.functional.interpolate(
            out, size=input_size, mode="bilinear", align_corners=False
        )

        return out

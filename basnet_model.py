import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolutional block
def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class BASNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(BASNet, self).__init__()

        # Encoder
        self.conv1 = conv_block(n_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bridge = conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        # Bridge
        b = self.bridge(p4)

        # Decoder
        u4 = self.upconv4(b)
        merge4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(merge4)

        u3 = self.upconv3(d4)
        merge3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(merge3)

        u2 = self.upconv2(d3)
        merge2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(merge2)

        u1 = self.upconv1(d2)
        merge1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(merge1)

        out = self.final(d1)
        out = torch.sigmoid(out)
        return out, out, out, out, out, out, out  # mimic original BASNet multi-output

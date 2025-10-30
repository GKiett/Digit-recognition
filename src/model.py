import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=1, channels=(32, 64, 128), fc_dim=128, dropout=0.3, num_classes=10):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            ConvBlock(in_ch, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x

def build_model(cfg):
    mcfg = cfg["model"]
    return SimpleCNN(
        in_ch=1,
        channels=tuple(mcfg.get("channels", [32, 64, 128])),
        fc_dim=mcfg.get("fc_dim", 128),
        dropout=mcfg.get("dropout", 0.3),
        num_classes=10
    )

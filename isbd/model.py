"""
ISBD v1.00 — Model Architecture Suite
- TinyUNet: 117K params (legacy, CPU-fast)
- SmallUNet: ~500K params (better quality, still CPU-trainable)
- Both use residual connections + attention for better gradient flow
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class ResDoubleConv(nn.Module):
    """Residual DoubleConv: adds skip connection when input/output channels match."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class ChannelAttention(nn.Module):
    """Lightweight channel attention (squeeze-excite)."""
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class TinyUNet(nn.Module):
    """Input 3ch -> Output 3ch. ~117k params at base_ch=16. Legacy model."""

    def __init__(self, base_ch: int = 16):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.d1 = DoubleConv(3, c1)          # 64x64
        self.p1 = nn.MaxPool2d(2)            # -> 32x32
        self.d2 = DoubleConv(c1, c2)
        self.p2 = nn.MaxPool2d(2)            # -> 16x16
        self.mid = DoubleConv(c2, c3)        # bottleneck
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out = nn.Conv2d(c1, 3, 1)

    def forward(self, x):
        s1 = self.d1(x)                      # skip 1
        s2 = self.d2(self.p1(s1))            # skip 2
        m = self.mid(self.p2(s2))
        x = self.u2(m)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.u1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.out(x)


class SmallUNet(nn.Module):
    """
    Upgraded U-Net: ~500K params at base_ch=24.
    - Residual blocks (better gradient flow)
    - Channel attention in bottleneck
    -4-level encoder/decoder (deeper)
    - Same input/output as TinyUNet (drop-in replacement)
    """

    def __init__(self, base_ch: int = 24):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Encoder
        self.d1 = ResDoubleConv(3, c1)       # 64x64
        self.p1 = nn.MaxPool2d(2)            # -> 32x32
        self.d2 = ResDoubleConv(c1, c2)      # 32x32
        self.p2 = nn.MaxPool2d(2)            # -> 16x16
        self.d3 = ResDoubleConv(c2, c3)      # 16x16
        self.p3 = nn.MaxPool2d(2)            # -> 8x8

        # Bottleneck with attention
        self.mid = ResDoubleConv(c3, c4)     # 8x8
        self.attn = ChannelAttention(c4)

        # Decoder
        self.u3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ResDoubleConv(c3 + c3, c3)
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ResDoubleConv(c2 + c2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ResDoubleConv(c1 + c1, c1)

        self.out = nn.Conv2d(c1, 3, 1)

    def forward(self, x):
        s1 = self.d1(x)                      # 64x64
        s2 = self.d2(self.p1(s1))            # 32x32
        s3 = self.d3(self.p2(s2))            # 16x16
        m = self.mid(self.p3(s3))            # 8x8
        m = self.attn(m)

        x = self.u3(m)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.u2(x)
        x = self.dec2(torch.cat([x, s2], dim=1))
        x = self.u1(x)
        x = self.dec1(torch.cat([x, s1], dim=1))
        return self.out(x)


MODEL = SmallUNet  # Default model for new training runs


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for name, cls in [("TinyUNet", TinyUNet), ("SmallUNet", SmallUNet)]:
        m = cls()
        print(f"{name} params: {param_count(m):,}")
        y = m(torch.randn(1, 3, 64, 64))
        print(f"  output: {y.shape}")

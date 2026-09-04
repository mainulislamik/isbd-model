"""
ISBD v1.00 — Tiny U-Net (CPU-trainable)
Architecture: 3-level encoder/decoder U-Net, 16-32-64 channels.
Future high-hardware path: same code scales up (depth, base_ch, IMG_SIZE in data.py).
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


class TinyUNet(nn.Module):
    """Input 3ch -> Output 3ch. ~145k params at base_ch=16."""

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
        return self.out(x)                   # residual-free logits in [0,1]-ish range


MODEL = TinyUNet


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = MODEL()
    print(f"TinyUNet params: {param_count(m):,}")
    y = m(torch.randn(1, 3, 64, 64))
    print("output:", y.shape)

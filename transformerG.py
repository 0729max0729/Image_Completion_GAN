import torch
from torch import nn


# ======================
# 生成器：TransUNet-lite（Conv encoder + Transformer bottleneck + Conv decoder）
# ======================
class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MHSA(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim*mlp_ratio), drop=drop)

    def forward(self, x):  # x: (B, N, C)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerBottleneck(nn.Module):
    """
    接收 2D feature map，攤平成 tokens，加入可學習的 2D 位置編碼，通過多層 Transformer。
    H=W=256/(2^4)=16，N=H*W=256；C=512（可調）。
    """
    def __init__(self, c, h, w, depth=4, num_heads=8, drop=0.2):
        super().__init__()
        self.c, self.h, self.w = c, h, w
        self.pos_embed = nn.Parameter(torch.zeros(1, h*w, c))  # (1, N, C)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(c, num_heads=num_heads, mlp_ratio=4.0, drop=drop)
            for _ in range(depth)
        ])

    def forward(self, feat):  # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        assert (C, H, W) == (self.c, self.h, self.w), f"Expected ({self.c},{self.h},{self.w}), got ({C},{H},{W})"
        x = feat.flatten(2).transpose(1, 2)  # (B, N, C)
        x = x + self.pos_embed  # (B, N, C)
        for blk in self.blocks:
            x = blk(x)  # (B, N, C)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # 回 2D
        return x

class TransformerGen(nn.Module):
    def __init__(self, base=64, drop2d=0.2, trans_depth=4, trans_heads=8):
        """
        base: 通道基數（64）
        drop2d: conv/上採樣中的 Dropout2d 機率（MC Dropout 用）
        trans_depth: Transformer 層數
        trans_heads: MHSA heads
        """
        super().__init__()
        C1, C2, C3, C4, CB = base, base*2, base*4, base*8, base*16  # 64,128,256,512,1024（這裡實際 bottleneck 用 512）

        # Encoder (stride=2 逐層下採樣到 1/16)
        self.enc1 = self._conv_block(4,   C1, drop2d)   # 256 -> 128
        self.enc2 = self._conv_block(C1,  C2, drop2d)   # 128 -> 64
        self.enc3 = self._conv_block(C2,  C3, drop2d)   # 64  -> 32
        self.enc4 = self._conv_block(C3,  C4, drop2d)   # 32  -> 16

        # Bottleneck：用 Transformer（通道用 C4=512, H=W=16）
        self.bottleneck = TransformerBottleneck(c=C4, h=16, w=16, depth=trans_depth, num_heads=trans_heads, drop=0.2)

        # Decoder（U-Net 路徑）
        self.up4 = self._up_block(C4, C3, drop2d)           # 16 -> 32
        self.up3 = self._up_block(C3+C3, C2, drop2d)        # cat skip
        self.up2 = self._up_block(C2+C2, C1, drop2d)
        self.up1 = self._up_block(C1+C1, C1, drop2d)

        self.out = nn.Sequential(
            nn.Conv2d(C1, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()
        )

    def _conv_block(self, in_c, out_c, drop2d, k=3, p=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=2, padding=p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d)   # 讓 MC Dropout 生效
        )

    def _up_block(self, in_c, out_c, drop2d, k=3, p=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=2, padding=p, output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # (B,64,128,128)
        e2 = self.enc2(e1)  # (B,128,64,64)
        e3 = self.enc3(e2)  # (B,256,32,32)
        e4 = self.enc4(e3)  # (B,512,16,16)

        # Bottleneck (Transformer)
        b  = self.bottleneck(e4)  # (B,512,16,16)

        # Decoder with skip
        d4_in = torch.cat([self.up4(b), e3], dim=1)     # (B,256+256,32,32)
        d3_in = torch.cat([self.up3(d4_in), e2], dim=1) # (B,128+128,64,64)
        d2_in = torch.cat([self.up2(d3_in), e1], dim=1) # (B,64+64,128,128)
        d1    = self.up1(d2_in)                         # (B,64,256,256)

        out = self.out(d1)  # (B,3,256,256), [-1,1]
        return out

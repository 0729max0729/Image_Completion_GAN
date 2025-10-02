# tools/completion_tools.py
import os
import io
import uuid
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# 你的模型
from transformerG import TransformerGen

def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_generator(weights_path: str, device=None, base=64, drop2d=0.2, trans_depth=12, trans_heads=8):
    if device is None:
        device = default_device()
    gen = TransformerGen(base=base, drop2d=drop2d, trans_depth=trans_depth, trans_heads=8).to(device)
    state = torch.load(weights_path, map_location=device)
    gen.load_state_dict(state)
    gen.train()  # 啟用 Dropout 方便 MC Dropout
    return gen

def mask_image_center(img_t: torch.Tensor, mask_size: int = 100):
    """
    img_t: (3,H,W) in [-1,1]
    回傳 (masked_img, mask)，mask: (1,H,W) 1=洞/被遮, 0=保留
    """
    mask_size = mask_size - 8
    _, H, W = img_t.shape
    cx, cy = W // 2, H // 2
    half = mask_size // 2
    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, W)
    y1 = min(cy + half, H)

    mask = torch.zeros((1, H, W), device=img_t.device)
    mask[:, y0:y1, x0:x1] = 1.0
    noise = torch.randn_like(img_t) * 0.5
    masked = img_t * mask + noise * (1 - mask)
    return masked, mask

def to_tensor(img_pil: Image.Image, size=256):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    return tfm(img_pil)

def denorm(x: torch.Tensor):
    return (x * 0.5 + 0.5).clamp(0,1)

@torch.no_grad()
def mc_dropout_inpaint(
    gen: nn.Module,
    img_pil: Image.Image,
    mask_size: int = 100,
    n_samples: int = 30,
    device=None,
    save_dir: str = "outputs"
):
    """
    回傳：dict 內含各輸出路徑與數值
    """
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = next(gen.parameters()).device

    img_t = to_tensor(img_pil).to(device)           # (3,H,W) [-1,1]
    masked, mask = mask_image_center(img_t, mask_size)
    x_in = torch.cat([masked.unsqueeze(0), mask.unsqueeze(0)], dim=1)  # (1,4,H,W)

    outs = []
    for _ in range(n_samples):
        out = gen(x_in).cpu()     # (1,3,H,W)
        outs.append(out)
    outs = torch.stack(outs, dim=0)      # (N,1,3,H,W)

    mean = outs.mean(dim=0).squeeze(0)   # (3,H,W)
    std  = outs.std(dim=0).squeeze(0)    # (3,H,W)

    orig_vis   = denorm(img_t.cpu())
    masked_vis = denorm(masked.cpu())
    mean_vis   = denorm(mean)
    residual   = torch.abs(orig_vis - mean_vis)
    std_map    = std.mean(dim=0, keepdim=True)
    std_map    = (std_map - std_map.min()) / (std_map.max() - std_map.min() + 1e-8)
    std_vis    = std_map.repeat(3,1,1)

    uid = uuid.uuid4().hex[:8]
    p_masked   = os.path.join(save_dir, f"masked_{uid}.png")
    p_mean     = os.path.join(save_dir, f"inpaint_{uid}.png")
    p_std      = os.path.join(save_dir, f"uncert_{uid}.png")
    p_residual = os.path.join(save_dir, f"residual_{uid}.png")

    save_image(masked_vis, p_masked)
    save_image(mean_vis,   p_mean)
    save_image(std_vis,    p_std)
    save_image(residual,   p_residual)

    return {
        "masked": p_masked,
        "mean": p_mean,
        "uncertainty": p_std,
        "residual": p_residual,
        "metrics": {
            "residual_mean": float(residual.mean().item()),
            "std_min": float(std_map.min().item()),
            "std_max": float(std_map.max().item()),
            "std_mean": float(std_map.mean().item()),
        }
    }

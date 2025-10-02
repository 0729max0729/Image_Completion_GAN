# test.py
import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from train import mask_image           # 你訓練檔裡的遮罩函數（回傳 masked, mask）
from transformerG import TransformerGen  # 你剛剛換成的 Transformer 版 Generator

# ========== 參數 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator_path     = "model/generator_UNET256.pth"      # 權重路徑（要和你訓練時一致）
test_image_path    = "test_data/img.png"                # 測試圖
out_dir            = "test_results"
out_fixed_img      = os.path.join(out_dir, "generated_mean.png")
out_uncertainty    = os.path.join(out_dir, "uncertainty_std.png")
out_masked_img     = os.path.join(out_dir, "masked_image.png")
out_residual_img   = os.path.join(out_dir, "residual_image.png")
out_triptych       = os.path.join(out_dir, "triptych.png")
num_mc_samples     = 50                                 # MC Dropout 次數
mask_size          = 100                                 # 遮罩方塊大小
torch.set_grad_enabled(False)

# ========== 前處理 ==========
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]
])

def denorm(x):
    # x: tensor in [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)

# ========== 載入模型 ==========
generator = TransformerGen(base=64, drop2d=0.5, trans_depth=12, trans_heads=8).to(device)
state = torch.load(generator_path, map_location=device)
generator.load_state_dict(state)
# **啟用 Dropout** 以便 MC Dropout（重要！）
generator.train()

# ========== 載入單張測試圖 ==========
os.makedirs(out_dir, exist_ok=True)
img = Image.open(test_image_path).convert("RGB")
img_t = transform(img).to(device)          # (3,H,W)

# 加 batch 維度
img_b = img_t.unsqueeze(0)                 # (1,3,H,W)

# 產生遮罩：注意 mask_image 回傳 (masked, mask)
masked_b, mask_b = mask_image(img_b[0].clone(), mask_size=mask_size)
masked_b = masked_b.unsqueeze(0)          # (1,3,H,W)
mask_b   = mask_b.unsqueeze(0).unsqueeze(0)             # (1,3,H,W)
x_in = torch.cat([masked_b, mask_b], dim=1)
# ========== MC Dropout 推論 ==========
outs = []
for _ in range(num_mc_samples):
    out = generator(x_in)              # (1,3,H,W), in [-1,1]
    outs.append(out.cpu())
outs = torch.stack(outs, dim=0)            # (N,1,3,H,W)

mean = outs.mean(dim=0).squeeze(0)         # (3,H,W)
std  = outs.std(dim=0).squeeze(0)          # (3,H,W)

# ========== 後處理與存檔 ==========
orig_vis   = denorm(img_b.squeeze(0).cpu())
masked_vis = denorm(masked_b.squeeze(0).cpu())
mean_vis   = denorm(mean)
# 殘差以 [0,1] 空間計算更直觀
residual   = torch.abs(orig_vis - mean_vis)

# 不確定度用各通道平均成單通道，正規化到 [0,1]
std_map = std.mean(dim=0, keepdim=True)            # (1,H,W)
std_map_norm = (std_map - std_map.min()) / (std_map.max() - std_map.min() + 1e-8)
std_vis = std_map_norm.repeat(3, 1, 1)             # 轉成 3 通道方便存檔

# 存單張結果
save_image(masked_vis,   out_masked_img)
save_image(mean_vis,     out_fixed_img)
save_image(std_vis,      out_uncertainty)
save_image(residual,     out_residual_img)

# 串成一張對比圖：原圖 / masked / mean / 不確定度 / 殘差
grid = make_grid([orig_vis, masked_vis, mean_vis, std_vis, residual], nrow=5)
save_image(grid, out_triptych)

# 印出數據
print(f"[OK] 補全圖已存：{out_fixed_img}")
print(f"[OK] 不確定度圖已存：{out_uncertainty}")
print(f"[OK] 遮罩圖已存：{out_masked_img}")
print(f"[OK] 殘差圖已存：{out_residual_img}")
print(f"[OK] 對比圖已存：{out_triptych}")
print(f"平均殘差 (在 [0,1] 空間): {residual.mean().item():.6f}")
print(f"不確定度（std_map）統計：min={std_map.min().item():.6f}, max={std_map.max().item():.6f}, mean={std_map.mean().item():.6f}")

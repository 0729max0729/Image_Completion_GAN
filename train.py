import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from transformerG import TransformerGen

# ======================
# 基本參數
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
lr = 2e-4
epochs = 50
image_size = 256
save_every = 2                  # 每幾個 epoch 存一次 model
lambda_l1 = 50                  # 重建 L1 loss 權重
lambda_perc = 5                # 感知損失權重（可調）
lambda_fm = 1                   # Feature Matching 權重（可調）
num_mc_samples = 50             # MC Dropout 次數（推論用）
mask_size = 100

os.makedirs("images", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("test_results", exist_ok=True)

# ======================
# 遮罩函式（修正版）
# ======================
def mask_image(img, mask_size=mask_size):
    """
    在圖像上添加遮擋塊並用高斯噪聲填充，遮擋區域限制在中心圓內
    img: (C,H,W) in [-1,1]
    """
    _, h, w = img.size()
    radius = w // 2
    cx, cy = w // 2, h // 2

    while True:
        x0 = torch.randint(0, w - mask_size, (1,)).item()
        y0 = torch.randint(0, h - mask_size, (1,)).item()
        mx = x0 + mask_size // 2
        my = y0 + mask_size // 2
        if (mx - cx)**2 + (my - cy)**2 <= radius**2:
            break

    mask = torch.ones_like(img)            # 1 = 未遮，0 = 遮
    mask[:, y0:y0 + mask_size, x0:x0 + mask_size] = 0.0

    # 填噪聲到被遮的區域；未遮區保持原圖
    noise = torch.normal(mean=0.0, std=1.0, size=img.size(), device=img.device)
    out = img * (1 - mask) + noise * mask + noise * (1 - mask) * 0.5
    return out, mask[0,:,:]

# ======================
# 資料集
# ======================
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  # [-1,1]
    ])
    dataset = CustomDataset(root_dir="ADS_smith_chart_plots", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ======================
# 生成器：UNet + Dropout (MC Dropout 用)
# ======================
class UNetGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3,   64)
        self.enc2 = self.conv_block(64,  128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.dec5 = self.up_block(1024, 512)
        self.dec4 = self.up_block(512+512, 256)
        self.dec3 = self.up_block(256+256, 128)
        self.dec2 = self.up_block(128+128, 64)
        self.dec1 = self.up_block(64+64,  64)

        self.out = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()
        )

    def conv_block(self, in_c, out_c, k=3, p=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=2, padding=p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)  # 關鍵：加 Dropout 以便 MC Dropout
        )

    def up_block(self, in_c, out_c, k=3, p=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=2, padding=p, output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)  # 關鍵：加 Dropout
        )

    def forward(self, x):
        e1 = self.enc1(x)          # 128x128
        e2 = self.enc2(e1)         # 64x64
        e3 = self.enc3(e2)         # 32x32
        e4 = self.enc4(e3)         # 16x16
        e5 = self.enc5(e4)         # 8x8

        d5 = self.dec5(e5)         # 16x16
        d4 = self.dec4(torch.cat([d5, e4], dim=1))  # 32x32
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # 64x64
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # 128x128
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # 256x256

        return self.out(d1)

# ======================
# 判別器（回傳中間特徵以便 Feature Matching）
# ======================
class Discriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        def C(in_c, out_c, k=4, s=2, p=1, bn=True):
            conv = nn.Conv2d(in_c, out_c, k, s, p)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers = [conv]
            if bn:
                layers += [nn.BatchNorm2d(out_c)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        # PatchGAN: 3→64→128→256→512，最後 1 個 conv 出 logits map
        self.b1 = C(3,   64, bn=False)   # 128x128
        self.b2 = C(64,  128)            # 64x64
        self.b3 = C(128, 256)            # 32x32
        self.b4 = C(256, 512, s=1)       # 31x31 (接近70x70感受野)
        # 最後一層不加 BN/Act，直接 logits
        self.out = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # → 約 30x30 logits

    def forward(self, x, return_features=False):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        logits = self.out(f4)  # (B,1,H',W'), 不做 Sigmoid！
        if return_features:
            return logits, [f1, f2, f3, f4]
        return logits


# ======================
# 感知損失（VGG16 中層特徵）
# ======================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        # 取中間層
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(0, 4)])   # relu1_2 輸出 C=64
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])   # relu2_2 輸出 C=128
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # relu3_3 輸出 C=256

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # [-1,1] -> [0,1] -> ImageNet 標準化
        def norm_im(z):
            z = (z + 1) * 0.5
            return (z - self.mean) / self.std

        x = norm_im(x)
        y = norm_im(y)

        # 串接抽特徵（關鍵修正在這裡）
        x1 = self.slice1(x);  y1 = self.slice1(y)   # C=64
        x2 = self.slice2(x1); y2 = self.slice2(y1)  # C=128
        x3 = self.slice3(x2); y3 = self.slice3(y2)  # C=256

        loss  = nn.functional.l1_loss(x1, y1)
        loss += nn.functional.l1_loss(x2, y2)
        loss += nn.functional.l1_loss(x3, y3)
        return loss


# ======================
#梯度損失 + TV 損失（等高線更利落）
#梯度損失：讓邊界/等高線不糊。
# ======================
def grad_loss(pred, gt):
    def dxdy(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy
    pdx, pdy = dxdy(pred); gdx, gdy = dxdy(gt)
    return (pdx - gdx).abs().mean() + (pdy - gdy).abs().mean()

def tv_loss(x):
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


# ======================
# 初始化
# ======================
generator = TransformerGen(base=64, drop2d=0.2, trans_depth=12, trans_heads=8).to(device)
discriminator = Discriminator().to(device)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1  = nn.L1Loss()
perc_loss_fn  = PerceptualLoss().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 可選：載入舊權重
paths = {
    "G": f"model/generator_UNET{image_size}.pth",
    "D": f"model/discriminator_UNET{image_size}.pth",
    "optG": f"model/optimizer_G_UNET{image_size}.pth",
    "optD": f"model/optimizer_D_UNET{image_size}.pth"
}
if all(os.path.exists(p) for p in paths.values()):
    print("發現既有模型，載入中...")
    generator.load_state_dict(torch.load(paths["G"], map_location=device))
    discriminator.load_state_dict(torch.load(paths["D"], map_location=device))
    optimizer_G.load_state_dict(torch.load(paths["optG"], map_location=device))
    optimizer_D.load_state_dict(torch.load(paths["optD"], map_location=device))
else:
    print("未發現舊模型，從頭訓練。")

# ======================
# 訓練
# ======================
if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots()
    g_hist, d_hist = [], []
    line_g, = ax.plot([], [], label="G loss")
    line_d, = ax.plot([], [], label="D loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()

    def update_plot():
        line_g.set_data(range(len(g_hist)), g_hist)
        line_d.set_data(range(len(d_hist)), d_hist)
        ax.relim(); ax.autoscale_view()
        plt.pause(0.01)

    # ---- 可選：instance noise（讓 D 更穩、不那麼兇）----
    def add_instance_noise(x, sigma=0.0):
        if sigma <= 0: return x
        return x + sigma * torch.randn_like(x)

    step = 0
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            bsz = imgs.size(0)

            # 準備 masked 輸入與 mask（mask: 1=洞/被遮、0=已知）
            masked = torch.empty_like(imgs)
            masks  = torch.empty([imgs.size(0),1,image_size,image_size],device=device)
            for j in range(bsz):
                masked[j], masks[j] = mask_image(imgs[j])

            # ------------------
            # 1) 更新 D
            # ------------------
            discriminator.train(); generator.train()
            optimizer_D.zero_grad()

            # 先生成一批供 D/G 共用，避免兩邊分別前向造成梯度/BN抖動
            x_in = torch.cat([masked, masks], dim=1)
            fake = generator(x_in)

            # 判別器輸出是 patch logits：shape [B,1,H',W']；target 必須同形狀
            out_real = discriminator(add_instance_noise(imgs, 0.0))
            out_fake = discriminator(add_instance_noise(fake.detach(), 0.0))

            # target 尺寸對齊（可選 label smoothing / noise）
            # real_t = 0.9 + 0.05 * torch.rand_like(out_real)  # 開啟這行做 smoothing
            # fake_t = 0.05 * torch.rand_like(out_fake)
            real_t = torch.ones_like(out_real)
            fake_t = torch.zeros_like(out_fake)

            loss_real = criterion_gan(out_real, real_t)
            loss_fake = criterion_gan(out_fake, fake_t)
            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # 2) 更新 G
            # ------------------
            optimizer_G.zero_grad()

            out_fake_for_g = discriminator(fake)  # 重新跑或重用 out_fake 皆可
            adv_loss = criterion_gan(out_fake_for_g, torch.ones_like(out_fake_for_g))  # 目標同形狀

            # ---- 重建損失：可依需求計在「已知區域」或「洞區域」----
            # 已知區域（1 - masks）
            recon_known = criterion_l1(fake * (1 - masks), imgs * (1 - masks))
            # 洞區域（masks）
            recon_hole  = criterion_l1(fake * masks,         imgs * masks)
            # 你可以加權合併，洞區域通常給更大權重
            recon_loss = 0.05 * recon_known + 0.95 * recon_hole

            # 感知損失（全圖）
            p_loss = perc_loss_fn(fake, imgs)

            # Feature Matching：回傳中間特徵，逐層 L1
            _, feats_real = discriminator(imgs, return_features=True)
            _, feats_fake = discriminator(fake, return_features=True)
            fm = 0.0
            for fr, ff in zip(feats_real, feats_fake):
                fm += nn.functional.l1_loss(ff, fr.detach())
            fm_loss = fm / len(feats_real)

            g_loss_grad = grad_loss(fake, imgs)  # 邊界清晰
            g_loss_tv = tv_loss(fake)  # 平滑抑噪

            # 建議權重（可微調）
            lambda_grad = 1.0
            lambda_tv = 1e-4  # 或 5e-4, 1e-3 視覺感受調整

            total_g_loss = (
                    adv_loss
                    + lambda_l1 * recon_loss
                    + lambda_perc * p_loss
                    + lambda_fm * fm_loss
                    + lambda_grad * g_loss_grad
                    + lambda_tv * g_loss_tv
            )
            total_g_loss.backward()
            optimizer_G.step()

            # log
            step += 1
            g_hist.append(total_g_loss.item()); d_hist.append(d_loss.item())
            if step % 10 == 0:
                print(
                    f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}] "
                    f"D: {d_loss.item():.4f} | G: {total_g_loss.item():.4f} "
                    f"(adv {adv_loss.item():.3f} | recon {recon_loss.item():.3f} | perc {p_loss.item():.3f} | fm {fm_loss.item():.3f})"
                )
                update_plot()


            # 每個步驟定期存預覽（原圖 / masked / 生成）
            if step % 50 == 0:
                grid = torch.cat([imgs[:4], masked[:4], fake[:4]], dim=0)
                save_image(grid, f"images/preview_step_{step}.png", nrow=4, normalize=True, value_range=(-1,1))

        '''
        # 每個 epoch 存一次 sample
        with torch.no_grad():
            sample = generator(masked[:8])
            grid = torch.cat([imgs[:8], masked[:8], sample], dim=0)
            save_image(grid, f"images/epoch_{epoch+1}.png", nrow=8, normalize=True, value_range=(-1,1))
        '''

        # 存權重
        if (epoch+1) % save_every == 0 or (epoch+1) == epochs:
            torch.save(generator.state_dict(), paths["G"])
            torch.save(discriminator.state_dict(), paths["D"])
            torch.save(optimizer_G.state_dict(), paths["optG"])
            torch.save(optimizer_D.state_dict(), paths["optD"])

    plt.ioff()
    plt.savefig("test_results/loss.png")
    print("訓練完成！")


    # ======================
    # 推論：MC Dropout 不確定度
    # ======================
    def mc_dropout_predict(gen, masked_input, n_samples=50):
        gen.train()  # 關鍵：啟用 dropout
        outs = []
        with torch.no_grad():
            for _ in range(n_samples):
                o = gen(masked_input).cpu().numpy()
                outs.append(o)
        outs = np.stack(outs, axis=0)  # (N, B, C, H, W)
        mean = outs.mean(axis=0)
        std  = outs.std(axis=0)
        return mean, std

    # 示範：對一批 masked 圖做不確定度推論
    generator.eval()   # 先 eval 再切回 train，避免 BN 統計亂飄
    test_imgs = next(iter(dataloader)).to(device)
    test_masked = torch.empty_like(test_imgs)
    test_masks  = torch.empty_like(test_imgs)
    for j in range(test_imgs.size(0)):
        test_masked[j], test_masks[j] = mask_image(test_imgs[j])

    x_in = torch.cat([test_masked, test_masks], dim=1)
    mean, std = mc_dropout_predict(generator, x_in[:4], n_samples=num_mc_samples)  # (B,C,H,W)
    mean_t = torch.from_numpy(mean)
    std_t  = torch.from_numpy(std)

    # 存補全與不確定度圖
    # mean 為補全結果，std 的通道平均作為不確定度熱圖
    save_image(torch.from_numpy(mean), "images/mc_mean.png", nrow=4, normalize=True, value_range=(-1,1))
    # 標準差可視化（把 std 映射到 3 通道方便存圖）
    std_map = std_t.mean(dim=1, keepdim=True)              # (B,1,H,W)
    std_map_img = std_map.repeat(1,3,1,1)                  # 灰階→3通道
    std_map_img = (std_map_img - std_map_img.min()) / (std_map_img.max() - std_map_img.min() + 1e-8)
    save_image(std_map_img, "images/mc_std.png", nrow=4)

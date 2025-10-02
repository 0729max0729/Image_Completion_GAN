# train_numeric_completion.py
# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# ===== 基本設定 =====
torch.set_default_dtype(torch.float32)
assert torch.cuda.is_available(), "需要 GPU（CUDA）環境"
device = torch.device("cuda")

CSV_PATH     = "../ADS_loadpull_data/load_pull_PAE_sim_data_90nm_sweep_source_freq.csv"
TARGET_COL   = "PAE"
GLOBAL_FEATS = ["RFfreq","Vlow","M"]
SEED         = 42

# ===== 訓練超參 =====
EPOCHS          = 800
LR              = 2e-3
WEIGHT_DECAY    = 1e-4
MAX_NORM        = 1.0
USE_AMP         = True
CKPT_PATH       = "rf_numeric_completion.pt"

GROUPS_PER_STEP = 16
S_PER_GROUP     = 81
MICROBATCH_SIZE = 4096

# 只訓練期啟用「把散點池化到固定網格」：
POOL_IN_TRAIN = True
GRID_N   = 500         # 目標固定網格 500x500
GAMMA_MAX = 0.98       # Γ 半徑界
EPS      = 1e-8

# KNN / anchors（只做 LOAD 外插）
N_ANCH_L = 9
K_NEI    = 4
TAU      = 0.30
DROPOUT_P = 0.10

# 推論網格大小
INF_GRID_N = 500

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.makedirs("out", exist_ok=True)

# ===== 讀 CSV（CPU）→ GPU =====
df = pd.read_csv(
    CSV_PATH, low_memory=False, dtype={
        "RFfreq":"float32","Vlow":"float32","M":"float32",
        "real_indexs22":"float32","imag_indexs22":"float32",
        "imag_indexs11":"float32","real_indexs11":"float32",
        "PAE":"float32",
    }
)
# 與你繪圖程式一致順序
df.columns = ["RFfreq","Vlow","M",
              "real_indexs22","imag_indexs22",
              "imag_indexs11","real_indexs11","PAE"]

for col in ["RFfreq","Vlow","M",
            "real_indexs22","imag_indexs22",
            "imag_indexs11","real_indexs11",
            TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

# group by (RFfreq, Vlow, M)
group_key_str = (df["RFfreq"].astype(str)
                 + "|" + df["Vlow"].astype(str)
                 + "|" + df["M"].astype(str)).values
uniq, inv = np.unique(group_key_str, return_inverse=True)
group_id_np = inv.astype(np.int64)

# → GPU tensors
# 只做 Load 外插：用 s22 當 Γ_L
Gl = torch.from_numpy(df[["real_indexs22","imag_indexs22"]].to_numpy()).to(device, torch.float32)
Y  = torch.from_numpy(df[[TARGET_COL]].to_numpy().astype("float32")).to(device, torch.float32)
Gg = torch.from_numpy(df[GLOBAL_FEATS].to_numpy()).to(device, torch.float32)
group_id = torch.from_numpy(group_id_np).to(device=device, dtype=torch.long)

N = Gl.size(0)
num_groups = int(group_id.max().item()) + 1

# 建 group 索引（GPU）
sort_idx   = torch.argsort(group_id)
gid_sorted = group_id[sort_idx]
cuts = torch.nonzero(torch.diff(F.pad(gid_sorted, (1,0), value=-1)) != 0, as_tuple=False).flatten()
cuts = torch.cat([cuts, torch.tensor([N], device=device, dtype=torch.long)], dim=0)
group_indices = []
for gi in range(num_groups):
    start = cuts[gi].item(); end = cuts[gi+1].item()
    group_indices.append(sort_idx[start:end])

# ===== GPU 工具：縮放與採樣 =====
def fit_scaler(x):
    m = x.mean(0, keepdim=True)
    s = x.std(0, keepdim=True)
    s = torch.where(s>0, s, torch.full_like(s, 1e-6))
    return m, s
def transform(x, mean, std):
    return (x - mean) / std
def inverse_transform(xn, mean, std):
    return xn * std + mean

def fps(points, k):
    """ farthest point sampling on (N,2) points (GPU) """
    Np = points.size(0)
    sel = torch.empty(k, dtype=torch.long, device=device)
    sel[0] = torch.randint(0, Np, (1,), device=device)
    d2 = torch.full((Np,), 1e10, device=device)
    for i in range(1, k):
        last = points[sel[i-1]].unsqueeze(0)     # (1,2)
        dist = (points - last).pow(2).sum(-1)    # (N,)
        d2 = torch.minimum(d2, dist)
        sel[i] = torch.argmax(d2)
    return points[sel]  # (k,2)

# ===== 為每 group 取 9 個 Γ_L anchors + 81 筆子樣本 =====
anchor_l_list, reduced_group_indices = [], []
for gi in range(num_groups):
    idxs = group_indices[gi]
    P_l  = Gl[idxs]                    # (Ni,2)

    A_l = fps(P_l, N_ANCH_L)           # (9,2)

    per_anchor = max(1, S_PER_GROUP // N_ANCH_L) # 9
    d_l  = torch.cdist(P_l, A_l, p=2.0)         # (Ni,9)
    used = torch.zeros(P_l.size(0), dtype=torch.bool, device=device)
    chosen = []
    for j in range(N_ANCH_L):
        dist = d_l[:, j]
        order = torch.argsort(dist)            # 最近優先
        got = []
        for idx in order:
            if not used[idx]:
                got.append(idx.item()); used[idx] = True
            if len(got) >= per_anchor:
                break
        if len(got) > 0:
            chosen.append(idxs[torch.tensor(got, device=device)])
    if len(chosen) == 0:
        pick_idx = idxs[torch.randint(0, idxs.numel(), (S_PER_GROUP,), device=device)]
    else:
        pick_idx = torch.cat(chosen, dim=0)
        if pick_idx.numel() < S_PER_GROUP:
            extra = idxs[torch.randint(0, idxs.numel(), (S_PER_GROUP - pick_idx.numel(),), device=device)]
            pick_idx = torch.cat([pick_idx, extra], dim=0)
        else:
            pick_idx = pick_idx[:S_PER_GROUP]

    anchor_l_list.append(A_l)
    reduced_group_indices.append(pick_idx)

anchor_l = torch.stack(anchor_l_list, dim=0)     # (G,9,2)

# ===== 拆特徵 + 縮放（只用訓練群組的 81 子樣本 fit）=====
perm_groups = torch.randperm(num_groups, device=device)
n_val = max(1, int(0.2 * num_groups))
val_groups   = perm_groups[:n_val]
train_groups = perm_groups[n_val:]

train_mask = torch.zeros(N, device=device, dtype=torch.bool)
for gi in train_groups.tolist():
    train_mask[reduced_group_indices[gi]] = True

X_l   = Gl       # (N,2)
X_g   = Gg       # (N,3)
y     = Y        # (N,1)

l_mean, l_std = fit_scaler(X_l[train_mask])
g_mean, g_std = fit_scaler(X_g[train_mask])
y_mean, y_std = fit_scaler(y[train_mask])

X_l_n = transform(X_l, l_mean, l_std)
X_g_n = transform(X_g, g_mean, g_std)
y_n   = transform(y,   y_mean, y_std)

# ===== 訓練時專用：把散點落格 → 平均池化到固定 500x500 =====
def rasterize_and_pool_points(Gl_cur, yt, grid_n=GRID_N, rmax=GAMMA_MAX):
    """
    Gl_cur: (B,2)  batch 的 Γ_L 散點
    yt:     (B,1)  對應目標
    回傳：
      centers: (K,2)  每個有效格中心的座標（落在 [-rmax,rmax]^2）
      avg_y:  (K,1)  該格的平均 y
      mask2d: (grid_n, grid_n) 有效格遮罩
    """
    device = Gl_cur.device
    dtype  = Gl_cur.dtype

    x = Gl_cur[:, 0].clamp(-rmax, rmax)
    y = Gl_cur[:, 1].clamp(-rmax, rmax)

    scale = (grid_n - 1) / (2 * rmax)
    ix = ((x + rmax) * scale).floor().clamp(0, grid_n-1).long()
    iy = ((y + rmax) * scale).floor().clamp(0, grid_n-1).long()

    flat_idx = ix * grid_n + iy
    num_cells = grid_n * grid_n

    sum_y = torch.zeros(num_cells, device=device, dtype=dtype)
    cnt_y = torch.zeros(num_cells, device=device, dtype=dtype)

    sum_y.scatter_add_(0, flat_idx, yt.squeeze(1))
    cnt_y.scatter_add_(0, flat_idx, torch.ones_like(yt.squeeze(1), dtype=dtype))

    valid = cnt_y > 0
    if valid.sum() == 0:
        return Gl_cur, yt, torch.zeros(grid_n, grid_n, device=device, dtype=torch.bool)

    avg_y = (sum_y[valid] / (cnt_y[valid] + EPS)).unsqueeze(1)

    flat_pos = torch.nonzero(valid, as_tuple=False).squeeze(1).to(torch.long)
    cx = flat_pos // grid_n
    cy = flat_pos %  grid_n

    inv_scale = (2 * rmax) / (grid_n - 1)
    gx = cx.to(dtype) * inv_scale - rmax
    gy = cy.to(dtype) * inv_scale - rmax
    centers = torch.stack([gx, gy], dim=1)

    mask2d = valid.view(grid_n, grid_n)
    return centers, avg_y, mask2d

# ===== 模型（只對 Γ_L 做 KNN/RBF 聚合 + MLP）=====
class AnchorKNN_OnlyL(nn.Module):
    def __init__(self, emb_dim=64, k=K_NEI, tau=TAU, p=DROPOUT_P):
        super().__init__()
        self.k = k
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float32))
        self.embed = nn.Sequential(
            nn.Linear(2, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, emb_dim), nn.GELU(),
            nn.Dropout(p)
        )

    def forward(self, Gl_cur, ancL):
        wanted = self.embed[0].weight.dtype
        Gl_cur = Gl_cur.to(wanted)
        ancL   = ancL.to(wanted)

        B, M, _ = ancL.shape
        x_exp = Gl_cur.unsqueeze(1)             # (B,1,2)
        d2 = (ancL - x_exp).pow(2).sum(-1)      # (B,M)
        vals, idxs = torch.topk(-d2, k=self.k, dim=1)

        topA = torch.gather(ancL, 1, idxs.unsqueeze(-1).expand(-1,-1,2))  # (B,k,2)
        topE = self.embed(topA)                                           # (B,k,D)
        w = torch.softmax(vals / (-self.tau), dim=1)
        h = (topE * w.unsqueeze(-1)).sum(1)                               # (B,D)
        return h

class KNN_L_Reg(nn.Module):
    def __init__(self, glob_in, emb_dim=64, trunk_hid=512, p=DROPOUT_P):
        super().__init__()
        self.encL = AnchorKNN_OnlyL(emb_dim=emb_dim, p=p)
        self.glob = nn.Sequential(
            nn.Linear(glob_in, 128), nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, 128), nn.GELU()
        )
        self.trunk = nn.Sequential(
            nn.Linear(emb_dim + 128, trunk_hid), nn.GELU(),
            nn.LayerNorm(trunk_hid),
            nn.Linear(trunk_hid, trunk_hid), nn.GELU(),
            nn.Dropout(p)
        )
        self.head = nn.Linear(trunk_hid, 1)

    def forward(self, Gl_cur, g, ancL):
        hL = self.encL(Gl_cur, ancL)         # (B,D)
        hg = self.glob(g)                    # (B,128)
        h  = self.trunk(torch.cat([hL, hg], dim=-1))
        return self.head(h)                  # (B,1)

model  = KNN_L_Reg(glob_in=len(GLOBAL_FEATS), emb_dim=64, trunk_hid=512, p=DROPOUT_P).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = GradScaler(device="cuda", enabled=USE_AMP)
crit   = nn.SmoothL1Loss(reduction="mean")

# ===== 取批次（每 group 固定 81 筆）=====
def sample_batch_from_groups(gs):
    batch_idx = []
    for gi in gs.tolist():
        idxs = reduced_group_indices[gi]
        batch_idx.append(idxs)
    return torch.cat(batch_idx, dim=0)

# ===== 一個 epoch =====
def one_epoch(groups_tensor, train=True):
    model.train(train)
    total_loss, total_cnt = 0.0, 0
    steps = max(1, groups_tensor.numel() // GROUPS_PER_STEP)
    perm  = torch.randperm(groups_tensor.numel(), device=device)
    groups_perm = groups_tensor[perm]
    ptr = 0
    for _ in range(steps):
        gs = groups_perm[ptr:ptr+GROUPS_PER_STEP]; ptr += GROUPS_PER_STEP
        idx_all = sample_batch_from_groups(gs)
        if train: opt.zero_grad(set_to_none=True)

        B = idx_all.size(0)
        for st in range(0, B, MICROBATCH_SIZE):
            ed = min(st+MICROBATCH_SIZE, B)
            idx = idx_all[st:ed]

            Gl_cur = X_l_n[idx]           # (Mb,2)
            g      = X_g_n[idx]           # (Mb,3)
            yt     = y_n[idx]             # (Mb,1)
            grp    = group_id[idx]        # (Mb,)
            ancL   = anchor_l[grp]        # (Mb,9,2)

            Gl_cur = Gl_cur.float(); g = g.float(); ancL = ancL.float()

            if train and POOL_IN_TRAIN:
                # --- 把 micro-batch 內可能混雜的多個 group，「逐 group」池化後一起算 loss ---
                uniq_grps = torch.unique(grp)
                loss_sum, cnt_sum = 0.0, 0
                with autocast(device_type="cuda", enabled=USE_AMP):
                    for gi in uniq_grps.tolist():
                        mask = (grp == gi)
                        Gl_g  = Gl_cur[mask]
                        yt_g  = yt[mask]
                        if Gl_g.numel() == 0:
                            continue
                        # 池化到固定 500x500
                        Gl_pool, yt_pool, _ = rasterize_and_pool_points(Gl_g, yt_g,
                                                                        grid_n=GRID_N, rmax=GAMMA_MAX)
                        if Gl_pool.numel() == 0:
                            continue
                        Bp = Gl_pool.size(0)
                        # 該 group 的 global/anchor（從該 group任一筆取）
                        g_one   = g[mask][0:1]
                        anc_one = ancL[mask][0:1]
                        g_pool    = g_one.repeat(Bp, 1)
                        ancL_pool = anc_one.repeat(Bp, 1, 1)

                        pred = model(Gl_pool, g_pool, ancL_pool)
                        loss = crit(pred, yt_pool)
                        # 以 pooled 樣本數加權
                        loss_sum += loss * Bp
                        cnt_sum  += Bp

                    if cnt_sum == 0:
                        # 萬一沒有有效 pooled cell，就退回原始散點
                        pred = model(Gl_cur, g, ancL)
                        loss = crit(pred, yt)
                    else:
                        loss = loss_sum / cnt_sum

                scaler.scale(loss).backward()
                if MAX_NORM is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            else:
                # 驗證期（或關掉 POOL_IN_TRAIN）走原先散點版
                if train:
                    with autocast(device_type="cuda", enabled=USE_AMP):
                        pred = model(Gl_cur, g, ancL)
                        loss = crit(pred, yt)
                    scaler.scale(loss).backward()
                    if MAX_NORM is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                else:
                    with torch.no_grad():
                        pred = model(Gl_cur, g, ancL)
                        loss = crit(pred, yt)

            total_loss += loss.item() * (ed - st)
            total_cnt  += (ed - st)
    return total_loss / max(1,total_cnt)

# ===== 訓練主循環 =====
best_val, patience, stale = float("inf"), 40, 0
for ep in range(1, EPOCHS+1):
    tr = one_epoch(train_groups, train=True)
    va = one_epoch(val_groups,   train=False)
    print(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")

    if va + 1e-6 < best_val:
        best_val, stale = va, 0
        torch.save({
            "state_dict": model.state_dict(),
            "l_mean":  l_mean.detach().float().cpu().numpy(),
            "l_std":   l_std.detach().float().cpu().numpy(),
            "g_mean":  g_mean.detach().float().cpu().numpy(),
            "g_std":   g_std.detach().float().cpu().numpy(),
            "y_mean":  y_mean.detach().float().cpu().numpy(),
            "y_std":   y_std.detach().float().cpu().numpy(),
            "global_feats": GLOBAL_FEATS,
            "target": TARGET_COL,
            "anchor_l": anchor_l.detach().float().cpu().numpy(),
        }, CKPT_PATH)
        print(" -> saved", CKPT_PATH)
    else:
        stale += 1
        if stale >= patience:
            print(f"Early stop at epoch {ep} (best val={best_val:.4f})")
            break

# ===== 推論：重建單一 group 的 500x500 Γ_L 網格 =====
def load_ckpt(path=CKPT_PATH):
    ckpt = torch.load(path, map_location=device)
    model = KNN_L_Reg(glob_in=len(ckpt["global_feats"]), emb_dim=64, trunk_hid=512, p=DROPOUT_P).to(device)
    model.load_state_dict(ckpt["state_dict"])
    l_mean = torch.from_numpy(ckpt["l_mean"]).to(device, torch.float32)
    l_std  = torch.from_numpy(ckpt["l_std"]).to(device, torch.float32)
    g_mean = torch.from_numpy(ckpt["g_mean"]).to(device, torch.float32)
    g_std  = torch.from_numpy(ckpt["g_std"]).to(device, torch.float32)
    y_mean = torch.from_numpy(ckpt["y_mean"]).to(device, torch.float32)
    y_std  = torch.from_numpy(ckpt["y_std"]).to(device, torch.float32)
    anchor_l = torch.from_numpy(ckpt["anchor_l"]).to(device, torch.float32)
    return model, l_mean, l_std, g_mean, g_std, y_mean, y_std, anchor_l, ckpt["global_feats"], ckpt["target"]

@torch.no_grad()
def mc_predict(gen, Gl_n, g_n, ancL, T=50):
    gen.train()  # 啟用 dropout
    outs = []
    for _ in range(T):
        pred = gen(Gl_n, g_n, ancL)           # (B,1)
        outs.append(pred)
    YN = torch.stack(outs, 0)                  # (T,B,1)
    mu = YN.mean(0)                            # (B,1)
    sd = YN.std(0)                             # (B,1)
    return mu, sd

# 用驗證集第一個 group 示範
gi = int(val_groups[0].item()) if val_groups.numel() else 0
model, lM, lS, gM, gS, yM, yS, anchorL_all, GFEATS, TGT = load_ckpt(CKPT_PATH)
model.eval()

# 該 group 的 global 條件（原始空間 → 正規化）
g0   = X_g[group_indices[gi][0]].unsqueeze(0)
g0_n = (g0 - gM) / gS

# 固定 500×500 網格（限半徑）
u = torch.linspace(-GAMMA_MAX, GAMMA_MAX, INF_GRID_N, device=device)
U, V = torch.meshgrid(u, u, indexing="ij")
mask = (U**2 + V**2) <= (GAMMA_MAX**2)
Gl_grid = torch.stack([U[mask], V[mask]], dim=1)      # (B,2)

B = Gl_grid.size(0)
g_batch    = g0_n.repeat(B,1)
ancL_batch = anchorL_all[gi].unsqueeze(0).repeat(B,1,1)

Gl_n = (Gl_grid - lM) / lS
mu_n, sd_n = mc_predict(model, Gl_n, g_batch, ancL_batch, T=50)

mu = (mu_n * yS + yM).squeeze(1)   # (B,)
sd = (sd_n * yS).squeeze(1)        # (B,)

Zmu = torch.full(U.shape, float("nan"), device=device)
Zsd = torch.full(U.shape, float("nan"), device=device)
Zmu[mask] = mu; Zsd[mask] = sd

np.save("out/pae_mu.npy", Zmu.detach().cpu().numpy())
np.save("out/pae_sd.npy", Zsd.detach().cpu().numpy())
print("Saved out/pae_mu.npy, out/pae_sd.npy")

def plot_contour(Z, title, out_png, cmap="viridis"):
    Zc = Z.detach().cpu().numpy()
    Uc = U.detach().cpu().numpy()
    Vc = V.detach().cpu().numpy()
    plt.figure(figsize=(6,5))
    cs = plt.contourf(Uc, Vc, Zc, levels=30, cmap=cmap)
    plt.colorbar(cs, label=title)
    plt.gca().set_aspect("equal","box")
    plt.xlabel("Re Γl"); plt.ylabel("Im Γl")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

plot_contour(Zmu, f"{TGT} mean", "out/pae_mu.png", "viridis")
plot_contour(Zsd, f"{TGT} std",  "out/pae_sd.png", "magma")
print("Saved out/pae_mu.png, out/pae_sd.png")

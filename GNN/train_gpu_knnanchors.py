# train_gpu_knnanchors.py
# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_default_dtype(torch.float32)
assert torch.cuda.is_available(), "需要 GPU（CUDA）環境"
device = torch.device("cuda")

# =====================
# 路徑 & 欄位
# =====================
CSV_PATH     = "../ADS_loadpull_data/load_pull_PAE_sim_data_90nm_sweep_source_freq.csv"
TARGET_COL   = "PAE"
GLOBAL_FEATS = ["RFfreq","Vlow","M"]
SEED         = 42

# =====================
# 訓練參數
# =====================
EPOCHS          = 10000
LR              = 2e-3
WEIGHT_DECAY    = 1e-4
MAX_NORM        = 1.0
USE_AMP         = True
CKPT_PATH       = "rf_knnanchors_gpu.pt"
MIN_EPOCHS = 1000                 # 至少跑這麼多 epoch 才允許早停
EARLY_STOP_PATIENCE = 50         # 允許更長停滯
IMPROVE_DELTA = 1e-5             # 最小改善幅度
PLATEAU_PATIENCE = 10            # 幾輪沒進步就降 LR
PLATEAU_FACTOR = 0.5             # LR 乘上 0.5
PLATEAU_MIN_LR = 1e-5            # 最低 LR

# 每 step 用多少個 group、每個 group 取多少 pair（我們只用 9×9）
GROUPS_PER_STEP = 16
S_PER_GROUP     = 81          # 9x9
MICROBATCH_SIZE = 2048

# KNN/RBF 錨點設定
N_ANCH_S = 9                  # ZS 錨點數
N_ANCH_L = 9                  # ZL 錨點數
K_NEI    = 4                  # 取前 K 個鄰居
TAU      = 0.30               # RBF 溫度（exp(-d^2/tau)）

torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =====================
# 讀 CSV（CPU）→ GPU
# =====================
df = pd.read_csv(
    CSV_PATH, low_memory=False, dtype={
        "RFfreq":"float32","Vlow":"float32","M":"float32",
        "real_indexs22":"float32","imag_indexs22":"float32",
        "imag_indexs11":"float32","real_indexs11":"float32",
        "PAE":"float32",
    }
)
df.columns = ["RFfreq","Vlow","M","real_indexs22","imag_indexs22","imag_indexs11","real_indexs11","PAE"]
df = df.dropna().reset_index(drop=True)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# 以 (RFfreq,Vlow,M) 分 group
group_str = (df["RFfreq"].astype(str) + "|" + df["Vlow"].astype(str) + "|" + df["M"].astype(str)).values
uniq, inv = np.unique(group_str, return_inverse=True)
group_id_np = inv.astype(np.int64)

# 搬到 GPU
Gs = torch.from_numpy(df[["real_indexs11","imag_indexs11"]].to_numpy()).to(device)
Gl = torch.from_numpy(df[["real_indexs22","imag_indexs22"]].to_numpy()).to(device)
Y  = torch.from_numpy(df[[TARGET_COL]].to_numpy().astype("float32")).to(device)
Gg = torch.from_numpy(df[GLOBAL_FEATS].to_numpy()).to(device)
group_id = torch.from_numpy(group_id_np).to(device=device, dtype=torch.long)

N = Gs.size(0)
num_groups = int(group_id.max().item()) + 1

# 為每個 group 建索引表（GPU）
sort_idx = torch.argsort(group_id)
gid_sorted = group_id[sort_idx]
cuts = torch.nonzero(torch.diff(F.pad(gid_sorted, (1,0), value=-1)) != 0, as_tuple=False).flatten()
cuts = torch.cat([cuts, torch.tensor([N], device=device, dtype=torch.long)], dim=0)

group_indices = []
for gi in range(num_groups):
    start = cuts[gi].item(); end = cuts[gi+1].item()
    idxs = sort_idx[start:end]
    group_indices.append(idxs)

# =====================
# GPU 工具
# =====================
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
    # farthest point sampling（GPU）；points:(N,2)
    Np = points.size(0)
    sel = torch.empty(k, dtype=torch.long, device=device)
    # 隨機起點
    sel[0] = torch.randint(0, Np, (1,), device=device)
    d2 = torch.full((Np,), 1e10, device=device)
    for i in range(1, k):
        last = points[sel[i-1]].unsqueeze(0)               # (1,2)
        dist = (points - last).pow(2).sum(-1)              # (N,)
        d2 = torch.minimum(d2, dist)
        sel[i] = torch.argmax(d2)
    return points[sel]  # (k,2)

def build_nodes(Gs2, Gl2):
    return torch.stack([Gs2, Gl2], dim=1)  # (B,2,2)

# =====================
# 建「錨點」與「僅 9x9 對應樣本」索引（每 group）
# =====================
anchor_s_list, anchor_l_list, reduced_group_indices = [], [], []
for gi in range(num_groups):
    idxs = group_indices[gi]
    P_s = Gs[idxs]   # (Ni,2)
    P_l = Gl[idxs]   # (Ni,2)

    # 取 9 個 ZS / ZL 錨點（GPU 上 farthest point sampling）
    A_s = fps(P_s, N_ANCH_S)   # (9,2)
    A_l = fps(P_l, N_ANCH_L)   # (9,2)

    # 用最近對（ZS_anchor_i, ZL_anchor_j）→ 在該 group 中找最接近的原始樣本
    # 產生最多 81 個索引（可能有重複，稍後 unique）
    pick_idx = []
    # 向量化：對每個錨點同時計算距離
    # P_s: (Ni,2), A_s: (9,2) -> (Ni,9)
    d_s = torch.cdist(P_s, A_s, p=2.0)
    d_l = torch.cdist(P_l, A_l, p=2.0)
    # 對每對 (i,j) 做 argmin P_s + P_l 的總距離
    # 展開成 (Ni, 9, 9) 再取最小；為節省顯存改成 loop j（9次）即可
    Ni = idxs.size(0)
    for i_s in range(N_ANCH_S):
        for j_l in range(N_ANCH_L):
            tot = d_s[:, i_s] + d_l[:, j_l]     # (Ni,)
            arg = torch.argmin(tot)
            pick_idx.append(idxs[arg])
    pick_idx = torch.unique(torch.stack(pick_idx))  # 去重
    if pick_idx.numel() < S_PER_GROUP:
        # 不足就有放回補滿
        extra = pick_idx[torch.randint(0, pick_idx.numel(), (S_PER_GROUP - pick_idx.numel(),), device=device)]
        pick_idx = torch.cat([pick_idx, extra], dim=0)
    else:
        pick_idx = pick_idx[:S_PER_GROUP]

    anchor_s_list.append(A_s)
    anchor_l_list.append(A_l)
    reduced_group_indices.append(pick_idx)

anchor_s = torch.stack(anchor_s_list, dim=0)   # (G,9,2)
anchor_l = torch.stack(anchor_l_list, dim=0)   # (G,9,2)

# =====================
# 標準化（用訓練 group）
# =====================
perm_groups = torch.randperm(num_groups, device=device)
n_val = max(1, int(0.2 * num_groups))
val_groups = perm_groups[:n_val]
train_groups = perm_groups[n_val:]

train_mask_all = torch.zeros(N, device=device, dtype=torch.bool)
for gi in train_groups.tolist():
    train_mask_all[reduced_group_indices[gi]] = True

X_node = torch.cat([Gs, Gl], dim=1)  # (N,4)
X_glob = Gg                           # (N,3)
y      = Y                            # (N,1)

node_mean, node_std = fit_scaler(X_node[train_mask_all])
glob_mean, glob_std = fit_scaler(X_glob[train_mask_all])
y_mean,    y_std    = fit_scaler(y[train_mask_all])

X_node_n = transform(X_node, node_mean, node_std)
X_glob_n = transform(X_glob, glob_mean, glob_std)
y_n      = transform(y,      y_mean,    y_std)

# =====================
# 模型：錨點 KNN/RBF 聚合 + 互動 + MLP
# =====================
class AnchorKNNEncoder(nn.Module):
    def __init__(self, emb_dim=64, k=K_NEI, tau=TAU, p=0.1):
        super().__init__()
        self.k   = k
        self.tau = tau
        self.embed = nn.Sequential(
            nn.Linear(2, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, emb_dim), nn.GELU()
        )
        self.dropout = nn.Dropout(p)

    def _agg(self, x, A):
        """
        x: (B,2)    單一 Γ
        A: (B,M,2)  對應錨點座標
        """
        # 關鍵：保證與權重同 dtype（通常是 float32；在 AMP 內部會自轉 fp16/bf16）
        wanted = self.embed[0].weight.dtype
        x = x.to(wanted)
        A = A.to(wanted)

        B, M, _ = A.shape
        x_exp = x.unsqueeze(1)                 # (B,1,2)
        d2 = (A - x_exp).pow(2).sum(-1)       # (B,M), same dtype as inputs

        # 取前 k 小距離（用 -d2 取最大）
        vals, idxs = torch.topk(-d2, k=self.k, dim=1)  # (B,k)

        # 取對應 top-k 的 anchor（gather 會保留 dtype）
        topA = torch.gather(A, 1, idxs.unsqueeze(-1).expand(-1, -1, 2))  # (B,k,2)
        topE = self.embed(topA)  # (B,k,D)

        # RBF 權重（注意 tau 是 buffer，float32）
        w = torch.softmax(vals / (-self.tau), dim=1)  # vals = -d2，所以 /(-tau)
        h = (topE * w.unsqueeze(-1)).sum(1)  # (B,D)
        return self.dropout(h)

    def forward(self, nodes_2x2, ancS, ancL):
        # nodes_2x2:(B,2,2) → node0=Gs, node1=Gl
        Gs_cur = nodes_2x2[:,0,:]               # (B,2)
        Gl_cur = nodes_2x2[:,1,:]               # (B,2)
        hs = self._agg(Gs_cur, ancS)            # (B,D)
        hl = self._agg(Gl_cur, ancL)            # (B,D)
        return hs, hl

class KNNAnchorReg(nn.Module):
    def __init__(self, glob_in, emb_dim=64, trunk_hid=512, p=0.1):
        super().__init__()
        self.enc = AnchorKNNEncoder(emb_dim=emb_dim, p=p)
        self.glob_enc = nn.Sequential(
            nn.Linear(glob_in, 128), nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, 128), nn.GELU()
        )
        feat_in = emb_dim*4 + 128  # hs, hl, hs*hl, |hs-hl|, + global
        self.trunk = nn.Sequential(
            nn.Linear(feat_in, trunk_hid), nn.GELU(),
            nn.LayerNorm(trunk_hid),
            nn.Linear(trunk_hid, trunk_hid), nn.GELU(),
            nn.Dropout(p)
        )
        self.head = nn.Linear(trunk_hid, 1)

    def forward(self, nodes_2x2, g, ancS, ancL):
        hs, hl = self.enc(nodes_2x2, ancS, ancL)
        inter = torch.cat([hs, hl, hs*hl, torch.abs(hs-hl)], dim=-1)
        hg = self.glob_enc(g)
        h  = self.trunk(torch.cat([inter, hg], dim=-1))
        return self.head(h)

# =====================
# 訓練 & 驗證一步
# =====================
model  = KNNAnchorReg(glob_in=len(GLOBAL_FEATS), emb_dim=64, trunk_hid=512, p=0.1).to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(
    opt, mode='min', factor=PLATEAU_FACTOR,
    patience=PLATEAU_PATIENCE, threshold=IMPROVE_DELTA,
    min_lr=PLATEAU_MIN_LR, verbose=True
)
scaler = GradScaler(device="cuda", enabled=USE_AMP)
crit   = nn.SmoothL1Loss(reduction="mean")

def sample_batch_from_groups(gs):
    # 每個 group 僅用「reduced_group_indices[gi]」那 9x9 近鄰對
    batch_idx = []
    for gi in gs.tolist():
        idxs = reduced_group_indices[gi]
        # 固定就用全部（S_PER_GROUP），也可隨機抽樣
        batch_idx.append(idxs)
    return torch.cat(batch_idx, dim=0)  # (GROUPS_PER_STEP*S_PER_GROUP,)

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
            nodes = build_nodes(X_node_n[idx,:2], X_node_n[idx,2:])     # (Mb,2,2)
            g     = X_glob_n[idx]                                       # (Mb,3)
            yt    = y_n[idx]                                            # (Mb,1)
            grp   = group_id[idx]                                       # (Mb,)
            ancS  = anchor_s[grp]                                       # (Mb,9,2)
            ancL  = anchor_l[grp]                                       # (Mb,9,2)

            # ---> 這四行是保險，確保 float32，避免混到 float64
            nodes = nodes.to(torch.float32)
            g = g.to(torch.float32)
            ancS = ancS.to(torch.float32)
            ancL = ancL.to(torch.float32)
            if train:
                with autocast(device_type="cuda", enabled=USE_AMP):
                    pred = model(nodes, g, ancS, ancL)
                    loss = crit(pred, yt)
                scaler.scale(loss).backward()
                if MAX_NORM is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    pred = model(nodes, g, ancS, ancL)
                    loss = crit(pred, yt)

            total_loss += loss.item() * (ed - st)
            total_cnt  += (ed - st)
    return total_loss / max(1,total_cnt)


if __name__ == "__main__":
    # =====================
    # 訓練循環
    # =====================
    best_val, patience, stale = float("inf"), 20, 0
    for ep in range(1, EPOCHS+1):
        tr = one_epoch(train_groups, train=True)
        va = one_epoch(val_groups,   train=False)
        print(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")

        if va + 1e-6 < best_val:
            best_val, stale = va, 0
            torch.save({
                "state_dict": model.state_dict(),
                "node_mean": node_mean.detach().float().cpu().numpy(),
                "node_std":  node_std.detach().float().cpu().numpy(),
                "glob_mean": glob_mean.detach().float().cpu().numpy(),
                "glob_std":  glob_std.detach().float().cpu().numpy(),
                "y_mean":    y_mean.detach().float().cpu().numpy(),
                "y_std":     y_std.detach().float().cpu().numpy(),
                "global_feats": GLOBAL_FEATS,
                "target": TARGET_COL,
                # 錨點也要存（每 group）
                "anchor_s": anchor_s.detach().float().cpu().numpy(),
                "anchor_l": anchor_l.detach().float().cpu().numpy(),
            }, CKPT_PATH)
            print(" -> saved", CKPT_PATH)
        else:
            stale += 1
            if (ep >= MIN_EPOCHS) and (stale >= EARLY_STOP_PATIENCE):
                print(f"Early stop at epoch {ep} (best val={best_val:.4f})")
                break
        scheduler.step(va)
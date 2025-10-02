# infer_knn_contours_500.py
# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from torch.amp import autocast
from train_gpu_knnanchors import KNNAnchorReg, build_nodes, transform, inverse_transform

assert torch.cuda.is_available(), "需要 GPU（CUDA）環境"
device = torch.device("cuda")

CKPT_PATH = "rf_knnanchors_gpu.pt"
CSV_PATH  = "../ADS_loadpull_data/load_pull_PAE_sim_data_90nm_sweep_source_freq.csv"

# ==== 你要重建的條件（可自行修改）====
RFfreq = 2e9
Vlow   = 1.2
M      = 2

# ==== 網格解析度 ====
GRID_N = 500     # 500 x 500
RHO    = 0.98    # 掃描半徑（Smith 圖圓內）
CHUNK  = 65536   # 每次前向最多吃多少點，避免爆顯存

# ---------------------------------------------------
# 1) 載入 checkpoint（模型、scaler、anchors）
# ---------------------------------------------------
ckpt = torch.load(CKPT_PATH, map_location=device)

model = KNNAnchorReg(glob_in=len(ckpt["global_feats"]), emb_dim=64, trunk_hid=512, p=0.1).to(device)
model.load_state_dict(ckpt["state_dict"]); model.eval()

node_mean = torch.tensor(ckpt["node_mean"], device=device, dtype=torch.float32)
node_std  = torch.tensor(ckpt["node_std"],  device=device, dtype=torch.float32)
glob_mean = torch.tensor(ckpt["glob_mean"], device=device, dtype=torch.float32)
glob_std  = torch.tensor(ckpt["glob_std"],  device=device, dtype=torch.float32)
y_mean    = torch.tensor(ckpt["y_mean"],    device=device, dtype=torch.float32)
y_std     = torch.tensor(ckpt["y_std"],     device=device, dtype=torch.float32)

anchor_s  = torch.tensor(ckpt["anchor_s"],  device=device, dtype=torch.float32)  # (G,9,2)
anchor_l  = torch.tensor(ckpt["anchor_l"],  device=device, dtype=torch.float32)  # (G,9,2)

# ---------------------------------------------------
# 2) 找到目標 group 的 index（與訓練時一致的分組順序）
#    訓練時 group 由 np.unique(group_str) 產生 → 字典序
# ---------------------------------------------------
df_meta = pd.read_csv(CSV_PATH, low_memory=False)
df_meta.columns = ["RFfreq","Vlow","M",
                   "real_indexs22","imag_indexs22",
                   "imag_indexs11","real_indexs11","PAE"]

df_meta = df_meta.dropna().reset_index(drop=True)

group_str = (df_meta["RFfreq"].astype(str) + "|" +
             df_meta["Vlow"].astype(str)   + "|" +
             df_meta["M"].astype(str)).values
uniq = np.unique(group_str)

target_key = f"{RFfreq}|{Vlow}|{M}"
try:
    gi = int(np.where(uniq == target_key)[0][0])
except IndexError:
    raise RuntimeError(f"在 CSV 裡找不到 group = ({RFfreq}, {Vlow}, {M})")

print(f"[Info] Using group index gi={gi} for (RFfreq={RFfreq}, Vlow={Vlow}, M={M})")

# 該 group 的 9 個錨點
ZS_anchors = anchor_s[gi]  # (9,2)
ZL_anchors = anchor_l[gi]  # (9,2)

# ---------------------------------------------------
# 3) 建立 500x500 ZL 網格（只保留圓內）
# ---------------------------------------------------
u = np.linspace(-RHO, RHO, GRID_N, dtype=np.float32)
v = np.linspace(-RHO, RHO, GRID_N, dtype=np.float32)
U, V = np.meshgrid(u, v)     # (N,N)
mask = (U**2 + V**2) <= (RHO**2)

Gl_grid = np.stack([U[mask], V[mask]], axis=1)  # (K,2)
Gl_grid = torch.from_numpy(Gl_grid).to(device)

# 對應這張 contour 用的 ZL 錨點（同一組 9 點，broadcast 成 K×9×2）
ancL_all = ZL_anchors.unsqueeze(0).expand(Gl_grid.size(0), -1, -1)  # (K,9,2)

# 固定的 global 條件
g_raw = torch.tensor([[RFfreq, Vlow, M]], device=device, dtype=torch.float32)
g_n   = transform(g_raw, glob_mean, glob_std)

# 準備輸出資料夾
os.makedirs("out/contours_500", exist_ok=True)

# ---------------------------------------------------
# 4) 逐一處理九個 ZS，為每個 ZS 產生 500x500 contour
# ---------------------------------------------------
for k in range(ZS_anchors.size(0)):
    zs = ZS_anchors[k]                            # (2,)
    zsK = zs.unsqueeze(0).expand(Gl_grid.size(0), -1)  # (K,2)

    # nodes_raw: (K,4) → (ReΓs,ImΓs, ReΓl,ImΓl)
    nodes_raw = torch.cat([zsK, Gl_grid], dim=1)       # (K,4)
    nodes_n   = transform(nodes_raw, node_mean, node_std).reshape(-1,2,2)
    g_batch   = g_n.repeat(nodes_n.size(0), 1)

    # ancS：這張圖用固定的 9 個 ZS 錨點（與訓練一致，效果更穩）
    ancS_all = ZS_anchors.unsqueeze(0).expand(nodes_n.size(0), -1, -1)  # (K,9,2)

    # ---- 分塊前向，避免顯存爆 ----
    preds = []
    with torch.no_grad():
        # 這裡資料量很大，但算子簡單；AMP 可開可關
        with autocast(device_type="cuda", enabled=False):
            Ntot = nodes_n.size(0)
            for st in range(0, Ntot, CHUNK):
                ed = min(st + CHUNK, Ntot)
                yp = model(nodes_n[st:ed], g_batch[st:ed],
                           ancS_all[st:ed], ancL_all[st:ed])  # (m,1)
                preds.append(yp)
    pred_n = torch.cat(preds, dim=0)                # (K,1)
    pred   = inverse_transform(pred_n, y_mean, y_std).squeeze().float().cpu().numpy()

    # 放回 500x500 網格
    Z = np.full(U.shape, np.nan, dtype=np.float32)
    Z[mask] = pred

    # 存圖
    plt.figure(figsize=(6.4, 5.6))
    cs = plt.contourf(U, V, Z, levels=40, cmap="viridis")
    plt.colorbar(cs, label=ckpt["target"])
    plt.scatter([zs[0].item()], [zs[1].item()], c="red", s=40, label="ZS anchor")
    plt.xlabel("Re Γl"); plt.ylabel("Im Γl")
    plt.legend(loc="lower right")
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()

    fn_png = f"out/contours_500/ZS_{zs[0].item():+0.3f}_{zs[1].item():+0.3f}.png"
    plt.savefig(fn_png, dpi=180); plt.close()

    # 也存成 .npy，方便後處理
    fn_npy = f"out/contours_500/ZS_{zs[0].item():+0.3f}_{zs[1].item():+0.3f}.npy"
    np.save(fn_npy, Z)

    print("Saved:", fn_png, "|", fn_npy)

print("All contours (500x500) done.")

# plot_from_file.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def parse_lpwave_file(file_path: str):
    """
    從 .lpwave 文字檔讀取數據列，回傳 ndarray shape=(N,5)
    欄序假設為: [Vlow, RFfreq, imag_indexs11, real_indexs11, PAE]
    做法：掃到第一行「像 0.700,0.750,-0.958,-0.249,37.690」的逗號數字後，
    之後所有「逗號分隔且轉得成 5 個 float」的行都收集。
    """
    rows = []
    started = False
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if "," not in ln:
                continue
            parts = [p.strip() for p in ln.split(",")]
            # 只收長度為 5 的行
            if len(parts) != 5:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                # 這行有非數字，略過
                continue
            # 第一次遇到就啟動
            started = True
            rows.append(vals)

    if not started or not rows:
        raise RuntimeError("檔案中沒有找到任何 5 欄逗號分隔的數據列。")

    arr = np.asarray(rows, dtype=np.float64)
    # arr 欄序: [Vlow, RFfreq, imag, real, PAE]
    return arr

def polar_contour_from_rows(rows,
                            grid_mag=100, grid_ang=200,
                            levels=50, cmap='jet',
                            outdir='out_lpwave'):
    """
    rows: ndarray (N,5) with columns [Vlow, RFfreq, imag, real, PAE]
    以 (Vlow, RFfreq) 分組，畫 polar 等高線（角度, 幅值）。
    """
    os.makedirs(outdir, exist_ok=True)

    Vlow = rows[:, 0]
    Freq = rows[:, 1]
    imag = rows[:, 2]
    real = rows[:, 3]
    pae  = rows[:, 4]

    uniq_v = np.unique(Vlow)
    uniq_f = np.unique(Freq)

    # 極座標網格（角度×幅值）
    mag_unique   = np.linspace(0.0, 1.0, grid_mag)
    angle_unique = np.linspace(-np.pi, np.pi, grid_ang)
    mag_grid, angle_grid = np.meshgrid(mag_unique, angle_unique)

    # 轉直角座標用來插值
    real_grid = mag_grid * np.cos(angle_grid)
    imag_grid = mag_grid * np.sin(angle_grid)

    count = 0
    for v in uniq_v:
        for f in uniq_f:
            mask = (Vlow == v) & (Freq == f)
            if not np.any(mask):
                continue

            re = real[mask]
            im = imag[mask]
            yy = pae [mask]

            if re.size < 3:
                # 點太少不畫
                continue

            points = np.column_stack([re, im])
            Z = griddata(points, yy, (real_grid, imag_grid), method='linear')

            # 補洞（全是 NaN 就用最近鄰；否則只補 NaN 的格子）
            Z_nn = griddata(points, yy, (real_grid, imag_grid), method='nearest')
            if Z is None or np.isnan(Z).all():
                Z = Z_nn
            else:
                Z = np.where(np.isnan(Z), Z_nn, Z)

            # 畫 polar contour（x=angle, y=mag）
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(2.8, 2.8))
            cs = ax.contour(angle_grid, mag_grid, Z, cmap=cmap, levels=levels)

            ax.axis("scaled")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(0, 1)

            fn = f"contour_v{v}_f{f}.png".replace(".", "p")
            out_png = os.path.join(outdir, fn)
            plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1, dpi=180)
            plt.close(fig)
            count += 1
            print(f"[OK] {out_png}")

    if count == 0:
        print("沒有任何分組成功繪圖（可能有效點數不足？）")

if __name__ == "__main__":
    # 把這裡換成你的 .lpwave 檔案路徑
    FILE_PATH = r"ADS_loadpull_data/load_pull_PAE_sim_data_90nm_sweep_Vlow_freq.csv"

    rows = parse_lpwave_file(FILE_PATH)
    # rows 欄序 = [Vlow, RFfreq, imag_indexs11, real_indexs11, PAE]
    polar_contour_from_rows(
        rows,
        grid_mag=100,      # 想更細可以調到 250/500
        grid_ang=200,
        levels=15,
        cmap='jet',
        outdir='ADS_smith_chart_plots'
    )

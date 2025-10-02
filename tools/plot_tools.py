# tools/plot_tools.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image

def load_lpwave_points(txt_path: str):
    """
    讀取 LPwave 純文字檔，回傳：(cols, data)
      - cols: 欄位名稱 list[str]
      - data: (N, D) np.ndarray[float64]
    """
    cols, rows = None, []
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if cols is None and line.startswith("Point"):
                cols = line.split()
                continue
            if cols is not None:
                parts = line.split()
                if len(parts) < len(cols):
                    continue
                try:
                    values = [float(x) for x in parts[:len(cols)]]
                    rows.append(values)
                except ValueError:
                    pass
    if cols is None or not rows:
        raise RuntimeError("找不到表頭 'Point ...' 或資料列為空。")
    return cols, np.asarray(rows, dtype=np.float64)

def gamma_polar_to_cart(mag, phase_deg):
    phi = np.deg2rad(phase_deg)
    return mag*np.cos(phi), mag*np.sin(phi)

def plot_full_polar_contour_lpwave(
    txt_path: str,
    metric: str = "PAEffWaves[%]",
    mag_n: int = 100,
    ang_n: int = 200,
    cmap: str = "jet",
    levels: int = 50,
    dpi: int = 180,
    out_dir: str = "outputs"
) -> str:
    """
    第一步：先畫完整極座標等高線圖（不做裁切），回傳輸出圖檔路徑。
    """
    os.makedirs(out_dir, exist_ok=True)
    cols, data = load_lpwave_points(txt_path)
    if metric not in cols:
        raise KeyError(f"找不到欄位：{metric}；可用欄位：{cols}")

    idx_gamma = cols.index("Gamma")
    idx_phase = cols.index("Phase[deg]")
    idx_y     = cols.index(metric)

    gamma = data[:, idx_gamma]
    phase = data[:, idx_phase]
    yval  = data[:, idx_y]

    # 轉為 (Re, Im) 當插值點
    re, im = gamma_polar_to_cart(gamma, phase)
    points = np.column_stack((re, im))

    # 建立極座標網格 → 轉直角座標去插值
    mag_unique   = np.linspace(0, 1, mag_n)
    angle_unique = np.linspace(-np.pi, np.pi, ang_n)
    mag_grid, angle_grid = np.meshgrid(mag_unique, angle_unique, indexing="xy")
    real_grid = mag_grid * np.cos(angle_grid)
    imag_grid = mag_grid * np.sin(angle_grid)

    Zi = griddata(points, yval, (real_grid, imag_grid), method="linear")
    if np.isnan(Zi).all():
        Zi = griddata(points, yval, (real_grid, imag_grid), method="nearest")

    base = os.path.splitext(os.path.basename(txt_path))[0]
    out_full_png = os.path.join(out_dir, f"{base}_{metric.replace('%','pct').replace('/','_')}_polar_full.png")

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(2.56, 2.56), dpi=dpi)
    ax.contourf(angle_grid, mag_grid, Zi, cmap=cmap, levels=levels)
    ax.axis("scaled")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_ylim(0, 1)
    plt.savefig(out_full_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return out_full_png

def keep_center_box_only(
    img_path: str,
    center_box_px: int = 100,
    background: str = "white",  # "white" or "transparent"
    out_path: str | None = None
) -> str:
    """
    第二步：影像後處理。只保留影像中央 N×N，其餘白色或透明。
    回傳輸出檔路徑。
    """
    im = Image.open(img_path).convert("RGBA")
    W, H = im.size
    cx, cy = W//2, H//2
    half = center_box_px // 2

    if background == "transparent":
        bg = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    else:
        bg = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    left = max(cx - half, 0)
    upper = max(cy - half, 0)
    right = min(cx + half, W)
    lower = min(cy + half, H)
    patch = im.crop((left, upper, right, lower))
    bg.paste(patch, (left, upper))

    if out_path is None:
        root, ext = os.path.splitext(img_path)
        suffix = f"_center{center_box_px}px"
        out_path = root + suffix + (".png" if background == "transparent" else ext)

    bg.save(out_path)
    return out_path

def lpwave_to_cropped_contour(
    txt_path: str,
    metric: str = "PAEffWaves[%]",
    mag_n: int = 100,
    ang_n: int = 200,
    cmap: str = "jet",
    levels: int = 50,
    dpi: int = 180,
    center_box_px: int = 100,
    background: str = "white",
    out_dir: str = "outputs"
) -> tuple[str, str]:
    """
    便利函式：一步呼叫就同時得到
      - 完整圖 full_png
      - 中央裁切圖 cropped_png
    """
    full_png = plot_full_polar_contour_lpwave(
        txt_path, metric, mag_n, ang_n, cmap, levels, dpi, out_dir
    )
    cropped_png = keep_center_box_only(full_png, center_box_px, background)
    return full_png, cropped_png

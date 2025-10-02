# app.py
import os
import io
import uvicorn
import zipfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

from tools.plot_tools import lpwave_to_cropped_contour
from tools.completion_tools import load_generator, mc_dropout_inpaint, default_device

APP_TITLE = "LPwave Contour & Image Completion (Return Binary)"
OUTPUT_DIR = "outputs"
MODEL_WEIGHTS = os.environ.get("GEN_WEIGHTS", "model/generator_UNET256.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title=APP_TITLE)

GEN = None
@app.on_event("startup")
def _load_model():
    global GEN
    device = default_device()
    if os.path.exists(MODEL_WEIGHTS):
        GEN = load_generator(MODEL_WEIGHTS, device=device, base=64, drop2d=0.2, trans_depth=12, trans_heads=8)
        print(f"[OK] Generator loaded on {device}: {MODEL_WEIGHTS}")
    else:
        GEN = None
        print(f"[WARN] Generator weights not found at {MODEL_WEIGHTS}. /complete-image will return 503.")

@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <html>
    <head><title>{APP_TITLE}</title></head>
    <body>
      <h2>LPwave → Contour（直接回傳圖片或ZIP）</h2>
      <form action="/make-contour" method="post" enctype="multipart/form-data">
        <p>LPwave 檔案: <input type="file" name="file" required></p>
        <p>Metric 欄位 (預設 PAEffWaves[%]): <input type="text" name="metric" value="PAEffWaves[%]"></p>
        <p>mag_n: <input type="number" name="mag_n" value="100"></p>
        <p>ang_n: <input type="number" name="ang_n" value="200"></p>
        <p>levels: <input type="number" name="levels" value="50"></p>
        <p>center_box_px: <input type="number" name="center_box_px" value="100"></p>
        <p>background:
           <select name="background">
             <option value="white">white</option>
             <option value="transparent">transparent</option>
           </select>
        </p>
        <p>return_type:
           <select name="return_type">
             <option value="full">full</option>
             <option value="cropped">cropped</option>
             <option value="both">both (zip)</option>
           </select>
        </p>
        <button type="submit">產生</button>
      </form>

      <hr/>
      <h2>Image Completion（直接回傳圖片或ZIP）</h2>
      <form action="/complete-image" method="post" enctype="multipart/form-data">
        <p>上傳圖片: <input type="file" name="image" required></p>
        <p>mask_size (px): <input type="number" name="mask_size" value="100"></p>
        <p>mc_samples: <input type="number" name="mc_samples" value="30"></p>
        <p>which:
           <select name="which">
             <option value="mean">mean (補全結果)</option>
             <option value="masked">masked</option>
             <option value="uncertainty">uncertainty</option>
             <option value="residual">residual</option>
             <option value="all">all (zip)</option>
           </select>
        </p>
        <button type="submit">補全</button>
      </form>
    </body>
    </html>
    """

# ---------- helpers ----------
def _bytes_response(path: str, filename: str | None = None):
    """讀檔為位元流並以 StreamingResponse 回圖。"""
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": f"file not found: {path}"}, status_code=404)
    ext = os.path.splitext(path)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg" if ext in [".jpg", ".jpeg"] else "application/octet-stream"
    with open(path, "rb") as f:
        data = f.read()
    headers = {}
    if filename:
        headers["Content-Disposition"] = f'inline; filename="{filename}"'
    return StreamingResponse(io.BytesIO(data), media_type=media_type, headers=headers)

def _zip_response(name_to_path: dict, zip_name: str):
    """把多檔打包為 ZIP 回傳。"""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, p in name_to_path.items():
            if os.path.exists(p):
                zf.write(p, arcname=arcname)
    mem.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{zip_name}"'}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)

# ---------- API: LPwave → contour ----------
@app.post("/make-contour")
async def make_contour(
    file: UploadFile = File(...),
    metric: str = Form("PAEffWaves[%]"),
    mag_n: int = Form(100),
    ang_n: int = Form(200),
    levels: int = Form(50),
    center_box_px: int = Form(100),
    background: str = Form("white"),
    return_type: str = Form("full")  # "full" | "cropped" | "both"
):
    # 存上傳檔
    src_path = os.path.join(OUTPUT_DIR, file.filename)
    with open(src_path, "wb") as f:
        f.write(await file.read())

    try:
        full_png, cropped_png = lpwave_to_cropped_contour(
            src_path, metric=metric, mag_n=mag_n, ang_n=ang_n,
            levels=levels, center_box_px=center_box_px, background=background,
            out_dir=OUTPUT_DIR
        )
        base = os.path.splitext(os.path.basename(src_path))[0]
        if return_type == "full":
            return _bytes_response(full_png, filename=f"{base}_full.png")
        elif return_type == "cropped":
            return _bytes_response(cropped_png, filename=f"{base}_cropped.png")
        else:
            # both -> zip
            return _zip_response(
                {
                    f"{base}_full.png": full_png,
                    f"{base}_cropped.png": cropped_png
                },
                zip_name=f"{base}_contours.zip"
            )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

# ---------- API: Image completion ----------
@app.post("/complete-image")
async def complete_image(
    image: UploadFile = File(...),
    mask_size: int = Form(100),
    mc_samples: int = Form(30),
    which: str = Form("mean")  # "mean" | "masked" | "uncertainty" | "residual" | "all"
):
    if GEN is None:
        return JSONResponse({"ok": False, "error": "Generator not loaded (weights missing)."}, status_code=503)

    from PIL import Image
    img_pil = Image.open(image.file).convert("RGB")
    try:
        result = mc_dropout_inpaint(
            GEN, img_pil,
            mask_size=mask_size,
            n_samples=mc_samples,
            save_dir=OUTPUT_DIR
        )
        base = os.path.splitext(image.filename or "image")[0]

        if which == "all":
            # 打包四張圖 + metrics.json
            import json, tempfile
            metrics_json = os.path.join(OUTPUT_DIR, f"{base}_metrics.json")
            with open(metrics_json, "w", encoding="utf-8") as f:
                json.dump(result["metrics"], f, ensure_ascii=False, indent=2)
            return _zip_response(
                {
                    f"{base}_masked.png":     result["masked"],
                    f"{base}_inpaint.png":    result["mean"],
                    f"{base}_uncert.png":     result["uncertainty"],
                    f"{base}_residual.png":   result["residual"],
                    f"{base}_metrics.json":   metrics_json,
                },
                zip_name=f"{base}_inpaint_bundle.zip"
            )
        else:
            key_map = {
                "mean": "mean",
                "masked": "masked",
                "uncertainty": "uncertainty",
                "residual": "residual"
            }
            if which not in key_map:
                return JSONResponse({"ok": False, "error": f"invalid 'which': {which}"}, status_code=400)
            path = result[key_map[which]]
            filename = f"{base}_{which}.png"
            return _bytes_response(path, filename=filename)

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

if __name__ == "__main__":
    # uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

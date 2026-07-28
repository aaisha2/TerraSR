"""TerraSR web-app backend (stage 10).

FastAPI service that loads a trained checkpoint and super-resolves an uploaded
image, returning the (upscaled) input and the SR output for a before/after
comparison. Because TerraSR is terrain-conditioned, the caller can pass the
terrain so the terrain embedding is used; "auto" falls back to the model's
unknown-terrain slot.

If no trained checkpoint is available yet (e.g. before the RESOLVE run), the
service degrades gracefully to a bicubic upscaler and says so, so the UI is
demonstrable end-to-end regardless.

Run:
    uvicorn app:app --app-dir webapp/backend --reload
    # or: python webapp/backend/app.py
"""
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import models  # noqa: E402
from terrasr_data import load_terrain_index  # noqa: E402

SCALE_FACTOR = 2
MAX_INPUT_SIDE = int(os.environ.get("TERRASR_MAX_SIDE", "512"))  # CPU-demo guard
CKPT_PATH = os.environ.get("TERRASR_CKPT", "checkpoints/terrasr/best.pth")
TERRAIN_MODELS = {"terrasr", "terrasr_swinir"}
FRONTEND = REPO_ROOT / "webapp" / "frontend" / "index.html"

app = FastAPI(title="TerraSR")

STATE = {"model": None, "model_name": "bicubic", "is_terrain": False,
         "terrain_index": load_terrain_index(str(REPO_ROOT / "configs" / "terrain_classes.yaml")),
         "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}


def _load_model():
    ckpt_path = REPO_ROOT / CKPT_PATH if not Path(CKPT_PATH).is_absolute() else Path(CKPT_PATH)
    if not ckpt_path.exists():
        print(f"[TerraSR] no checkpoint at {ckpt_path} — using bicubic fallback")
        return
    # weights_only=False: our own checkpoint carries a config dict + RNG state
    # the PyTorch 2.6+ safe loader rejects. Trusted (self-produced).
    ckpt = torch.load(ckpt_path, map_location=STATE["device"], weights_only=False)
    name = ckpt["model_name"]
    model = models.build(name, ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(STATE["device"])
    STATE.update(model=model, model_name=name, is_terrain=(name in TERRAIN_MODELS))
    print(f"[TerraSR] loaded {name} from {ckpt_path} on {STATE['device']}")


@app.on_event("startup")
def startup():
    _load_model()


def _to_data_url(arr01: np.ndarray) -> str:
    img = Image.fromarray(np.clip(arr01 * 255, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _read_gray01(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L")
    w, h = img.size
    if max(w, h) > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


@torch.no_grad()
def _super_resolve(lr01: np.ndarray, terrain: str) -> np.ndarray:
    lr = torch.from_numpy(lr01).unsqueeze(0).unsqueeze(0).to(STATE["device"])
    if STATE["model"] is None:
        sr = F.interpolate(lr, scale_factor=SCALE_FACTOR, mode="bicubic", align_corners=False)
    elif STATE["is_terrain"]:
        idx = STATE["terrain_index"].get(terrain, -1)
        sr = STATE["model"](lr, torch.tensor([idx], device=STATE["device"]))
    else:
        sr = STATE["model"](lr)
    return sr.clamp(0, 1).squeeze().cpu().numpy()


@app.get("/api/health")
def health():
    return {
        "model": STATE["model_name"],
        "terrain_conditioned": STATE["is_terrain"],
        "device": str(STATE["device"]),
        "scale_factor": SCALE_FACTOR,
        "terrains": list(STATE["terrain_index"].keys()),
        "using_fallback": STATE["model"] is None,
    }


@app.post("/api/infer")
async def infer(image: UploadFile = File(...), terrain: str = Form("auto")):
    try:
        lr01 = _read_gray01(await image.read())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"could not read image: {e}"})

    sr01 = _super_resolve(lr01, terrain)
    # upscale the input to SR size (nearest) so the before/after slider aligns
    before = np.asarray(
        Image.fromarray((lr01 * 255).astype(np.uint8)).resize(
            (sr01.shape[1], sr01.shape[0]), Image.NEAREST), dtype=np.float32) / 255.0

    return {
        "model": STATE["model_name"],
        "terrain_used": terrain if STATE["is_terrain"] else "n/a (model not terrain-conditioned)",
        "input_size": [int(lr01.shape[1]), int(lr01.shape[0])],
        "output_size": [int(sr01.shape[1]), int(sr01.shape[0])],
        "before_url": _to_data_url(before),
        "after_url": _to_data_url(sr01),
        "using_fallback": STATE["model"] is None,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    if FRONTEND.exists():
        return FRONTEND.read_text(encoding="utf-8")
    return "<h1>TerraSR</h1><p>frontend not found</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# TerraSR web app (stage 10)

Upload a low-resolution satellite image, pick its terrain, and get a 2×
super-resolved result with a before/after comparison slider. The backend runs
the trained terrain-conditioned TerraSR model (or any baseline checkpoint).

## Run

```bash
pip install -r webapp/requirements.txt          # fastapi, uvicorn, python-multipart

# point at a trained checkpoint (defaults to checkpoints/terrasr/best.pth)
export TERRASR_CKPT=checkpoints/terrasr/best.pth       # Windows: set TERRASR_CKPT=...
python webapp/backend/app.py                    # serves http://127.0.0.1:8000
# or: uvicorn app:app --app-dir webapp/backend --reload
```

Open http://127.0.0.1:8000 — the page (`webapp/frontend/index.html`) is served
by the backend, so there's no separate frontend build/toolchain.

## Endpoints

- `GET  /`            — the UI
- `GET  /api/health`  — loaded model, device, terrain list, whether it's the fallback
- `POST /api/infer`   — multipart `image` (+ optional `terrain`); returns the
  upscaled input and SR output as PNG data URLs

## Notes

- **Terrain conditioning:** the UI's terrain dropdown feeds the TerraSR terrain
  embedding, so the user picks the terrain (`auto` uses the model's
  unknown-terrain slot). Automatic terrain classification from the LR image is
  future work — the proposal's pipeline lists it as a stage, but the current
  system labels via ESA WorldCover at dataset-build time, not from the image at
  inference.
- **Graceful fallback:** if no checkpoint is present (e.g. before the RESOLVE
  training run), the service upscales with bicubic and the UI shows a "bicubic
  fallback" note, so the app is demonstrable end-to-end regardless.
- **CPU-demo guard:** uploads larger than `TERRASR_MAX_SIDE` (default 512 px)
  are downscaled before SR to stay responsive on CPU. Raise it (or remove the
  guard) when running on the RESOLVE GPU.
- The frontend is intentionally a single self-contained HTML file (vanilla
  JS). It can be replaced by the React front end from the proposal later; the
  JSON API is unchanged by that swap.

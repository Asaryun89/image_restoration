"""SRHCnet / ESRGAN x4 super-resolution webapp.

FastAPI server around ``infer.SRUpscaler``: every model is loaded once at
startup and shared across requests, serialized by a lock (the upscalers
are not thread-safe). ``POST /api/upscale`` takes a ``model`` form field
(``srhcnet`` | ``esrgan``) to pick the network per request. The frontend
lives in ``static/index.html``.

Run:
    python app.py                       # http://127.0.0.1:8000
    python app.py --device cpu --port 8080

CPU note: inference is minutes per megapixel on CPU, so ``--max-pixels``
(default 1.5 MP on CPU, 16 MP on CUDA) rejects oversized uploads with a
clear error instead of hanging for an hour.
"""
from __future__ import annotations

import argparse
import io
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

from infer import SRUpscaler

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "checkpoints/best_psnr.pth"
ESRGAN_CHECKPOINT = ROOT / "checkpoints/RRDB_ESRGAN_x4.pth"

MODEL_LABELS = {
    "srhcnet": "SRHCnet (HAT-L + gated NAFNet, NTIRE-2025 winner)",
    "esrgan": "ESRGAN (RRDBNet baseline, ECCV-W 2018)",
}

app = FastAPI(title="SRHCnet / ESRGAN x4 upscaler")

state: dict = {"models": {}, "device": None, "max_pixels": 0}
gpu_lock = threading.Lock()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/info")
def info() -> dict:
    return {
        "device": state["device"],
        "scale": 4,
        "max_pixels": state["max_pixels"],
        "models": {name: MODEL_LABELS[name] for name in state["models"]},
        "default_model": "srhcnet" if "srhcnet" in state["models"] else
                         next(iter(state["models"]), None),
    }


@app.post("/api/upscale")
def upscale(file: UploadFile = File(...), model: str = Form("srhcnet")) -> Response:
    sr = state["models"].get(model)
    if sr is None:
        raise HTTPException(
            400,
            f"Unknown model {model!r}; available: {sorted(state['models'])}.",
        )

    try:
        img = Image.open(io.BytesIO(file.file.read()))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(400, "Not a readable image file.")

    w, h = img.size
    if w * h > state["max_pixels"]:
        raise HTTPException(
            413,
            f"Image is {w}x{h} ({w * h / 1e6:.1f} MP); the limit on this "
            f"server ({state['device']}) is {state['max_pixels'] / 1e6:.1f} MP. "
            "Downscale the image or run the server with --max-pixels.",
        )

    with gpu_lock:
        t0 = time.perf_counter()
        out = sr.upscale(img)
        elapsed = time.perf_counter() - t0

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Elapsed-Seconds": f"{elapsed:.2f}",
            "X-Model": model,
            "Content-Disposition": f'inline; filename="{model}_x4.png"',
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="SRHCnet / ESRGAN x4 upscaling webapp")
    ap.add_argument("--checkpoint", default=str(CHECKPOINT),
                    help="SRHCnet checkpoint path")
    ap.add_argument("--esrgan-checkpoint", default=str(ESRGAN_CHECKPOINT),
                    help="ESRGAN checkpoint path (skipped if the file is missing)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="reject inputs above this many pixels "
                         "(default: 1.5M on CPU, 16M on CUDA)")
    args = ap.parse_args()

    if args.max_pixels is None:
        args.max_pixels = 16_000_000 if args.device.startswith("cuda") else 1_500_000

    to_load = [("srhcnet", args.checkpoint), ("esrgan", args.esrgan_checkpoint)]
    for name, ckpt in to_load:
        if not Path(ckpt).is_file():
            print(f"Skipping {name}: no checkpoint at {ckpt}")
            continue
        print(f"Loading {name} on {args.device} ...")
        try:
            state["models"][name] = SRUpscaler(
                ckpt, device=args.device, tile=args.tile, arch=name
            )
        except RuntimeError as e:
            if "PytorchStreamReader" in str(e) or "central directory" in str(e):
                raise SystemExit(
                    f"Checkpoint {ckpt!r} is corrupted (truncated zip "
                    "archive) — re-download or re-copy the weights file, then "
                    "restart the server."
                ) from e
            raise
    if not state["models"]:
        raise SystemExit("No model checkpoints found; nothing to serve.")

    state["device"] = args.device
    state["max_pixels"] = args.max_pixels
    print(f"Ready: {', '.join(state['models'])}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

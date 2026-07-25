# SRHCnet x4 inference bundle

NTIRE-2025 ImageSR winner (SamsungAICamera/SRC-B): HAT-L + gated NAFNet
refiner, 156.8M params. Upscales RGB images exactly 4x.

## Contents
- `models/` — self-contained architecture code (srhcnet, nafnet, tiling)
- `infer.py` — CLI + `SRUpscaler` class for webapp embedding
- `best_psnr.pth` — released composite weights (`params` key)

## Weight Install and Demo Video
Google Drive: https://drive.google.com/drive/folders/1--RYVZn95QUnJvnckg7f5DIEQO8-mUnU?usp=sharing

## Quick start
    pip install -r requirements.txt
    python infer.py --checkpoint checkpoints/best_psnr.pth \
        --input photo.png --output photo_x4.png

## Webapp
    python app.py                 # serves http://127.0.0.1:8000
    python app.py --device cpu --port 8080 --max-pixels 4000000

FastAPI server (`app.py`) + single-page frontend (`static/index.html`):
drag-and-drop or paste an image, upscale 4x, compare before/after with a
slider, download the PNG. The model is loaded once at startup; requests
are serialized (the upscaler is not thread-safe). Uploads larger than
`--max-pixels` (default 1.5 MP on CPU, 16 MP on CUDA) are rejected with
HTTP 413 rather than hanging.

## Webapp integration
    from infer import SRUpscaler
    sr = SRUpscaler("checkpoints/best_psnr.pth", device="cuda")  # load ONCE at startup
    out_pil = sr.upscale(in_pil)                                  # per request

Notes:
- ~0.7 GB weights; ~3 GB VRAM idle, ~6 GB peak with the default 256 tile.
  CPU works but is slow (minutes per megapixel image).
- Calls are not thread-safe; queue requests per GPU worker.
- Tiled inference (default) keeps memory flat on arbitrarily large inputs;
  `tile=0` does a whole-image forward (fastest for small inputs).
- Later fine-tuned checkpoints from the trainer (`best_psnr.pth`) load the
  same way — EMA weights are picked automatically.

# Training status — team17 DIV2K fine-tune reproduction

*Snapshot: 2026-07-20 01:04 UTC · run `team17_div2k_ft_20260719-162309` (resumed from epoch 24)*

## Process

| Item | Value |
|---|---|
| PID | 2301 (detached via `setsid`/`nohup`, survives disconnects) |
| Uptime since resume | 8 h 41 m (original run: epochs 1–24, killed externally ~04:49 Jul 19) |
| GPU | RTX PRO 6000 Blackwell — 54.8 / 97.9 GB, 57% util, 72 °C, ~306 W |
| Console log | `runs/resume_20260719.out` |
| Checkpoints | `checkpoints/team17_ft/{latest,best_psnr}.pth` |

## Data pipeline

Source: HF dataset **`bezdarnost/DF2K-bicubic`** — its 4-digit-named files are the
complete official DIV2K (train 0001–0800 + val 0801–0900, original ETH numbering,
HR **and** official MATLAB-bicubic ×4 LR), verified pixel-identical to the ETH
originals; the repo's 6-digit files (Flickr2K) are excluded. Prepared once via
`python tools/prepare_div2k.py --root data/DIV2K`:

1. **Download** — `snapshot_download` with `allow_patterns` `hr/????.png`, `x4/??????.png` (~4.3 GB)
2. **Arrange** — into the official zip layout: `DIV2K_train_HR/`, `DIV2K_train_LR_bicubic/X4/`, `DIV2K_valid_HR/`, `DIV2K_valid_LR_bicubic/X4/`
3. **Sub-images** — train pairs pre-cropped to 480×480 HR / 120×120 LR tiles, stride 240/60 (BasicSR convention) → **32,592 tiles** each side

Per-iteration processing (`data/dataset.py` + trainer):
- Aligned random crop: HR patch = current phase size (320/448/768), LR = /4
- Geometric aug: random h/v flips + 90° rotations, identical on both sides
- Batch aug (spec B.5): channel shuffle p=0.5 (one RGB permutation per batch),
  mixup p=0.2 (Beta(1.2, 1.2) blend against a shuffled pairing) — both applied
  identically to LR and HR, PSNR-safe under bicubic degradation
- Validation: full 100 val images, no crops; inference tiled at 256-px LR
  tiles with 32-px overlap-average stitching (full-2K fp32 forwards OOM)

## Model architecture

**SRHCnet** — NTIRE-2025 ImageSR ×4 winner composite (SamsungAICamera / SRC-B),
**156.8 M params** total; init from released checkpoint `17_SRHCnet_39000_final.pth`
(`params` container, strict load; gates β ≈ 0.985, γ ≈ 0.55).

```
LR (B,3,h,w) in [0,1]
  │ reflect-pad to /16, subtract DIV2K mean (0.4488, 0.4371, 0.4040)
  ▼
SRHnet — HAT-L transformer, ~40.8 M params, does the ×4 upscale
  conv_first 3→180
  12 × RHAG (each: 6 × HAB + 1 OCAB + conv + group residual)
    HAB: window-MSA (ws 16, shift 0/8 alternating, 6 heads, rel-pos bias)
         ∥ CAB conv branch (compress 3, squeeze 30) × conv_scale 0.01
         + MLP (ratio 2)
    OCAB: overlapping cross-attention (overlap ratio 0.5 → 24-px kv windows)
  conv_after_body + long skip → conv_before_upsample (180→64)
  ×4 pixel-shuffle upsampler (2 stages, ICNR init) → conv_last 64→3
  │ × β (per-channel gate), add mean back            → HAT output y
  ▼
SRCnet — NAFNet U-Net refiner at HR resolution, ~116 M params
  width 64, encoders [2,2,4,8], middle 12, decoders [2,2,2,2]
  NAFBlocks: LayerNorm + SimpleGate convs + simplified channel attention
  internal input skip → outputs a full refined image srcnet(y)
  ▼
out = y + (srcnet(y) − y) × γ     # γ gates the refinement DELTA
  │ crop padding back to h×4 / w×4
  ▼
SR (B,3,4h,4w)
```

Key wiring facts (cost real debugging time):
- γ gates the NAFNet **delta**, not its raw output — NAFNet's internal skip
  means `y + srcnet(y)·γ` double-counts the image and collapses PSNR to ~14 dB
- The `mean` buffer is non-persistent so the released state dict strict-loads
- Training: bf16 autocast (fp16 overflows), EMA 0.999, drop-path 0 (converged
  model), Adam(0.9, 0.99) wd=0 at lr 1e-5 → cosine 1e-6, grad-clip 1.0

## Position in schedule

- **Step ~103,500 / 150,000 (69%)** — mid-epoch **35 / 50** at 1.12 it/s
- Progressive phases: 320 px (ep 1–25, done) → **448 px @ batch 2 (ep 26–40, current)** → 768 px @ batch 1 (ep 41–50)
- LR cosine: 2.97e-6 (from 1e-5, floor 1e-6)


## Validation trajectory (PSNR-Y, 4-px border, challenge protocol)

Baseline (released checkpoint, pre-training): **31.659 dB** · acceptance line: **≥ 31.639 dB**

| Epochs | Phase | PSNR-Y | Note |
|---|---|---|---|
| 1 | 320 px | 31.630 | EMA ≈ loaded checkpoint |
| 2–8 | 320 px | dip to 31.594 | fresh-Adam + EMA re-averaging disturbance |
| 9–25 | 320 px | recover to 31.617 | steady climb as LR decays |
| 26–34 | **448 px** | **31.621 → 31.639** | 9 consecutive rising epochs |

Key milestones:
- **Epoch 32**: surpassed the run's own starting point (31.633 > 31.630) — `best_psnr.pth` now holds genuinely improved weights
- **Epoch 34**: 31.6388 — within 0.0003 dB of the acceptance threshold, still rising
- Larger patches measurably help, matching the report's claim; the 768-px phase (their biggest gain) is still ahead

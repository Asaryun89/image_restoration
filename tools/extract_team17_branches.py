"""Dissect the team17 composite checkpoint into its branches (spec §A.3).

Produces:
- ``hatl_2M_branch.pth``  — ``{'params': <bare HAT-L state>, 'beta': ...}``
- ``nafnet_refiner.pth``  — ``{'params': <bare NAFNet state>, 'gamma': ...}``

Per spec §A.4 the HAT branch was always consumed through ``beta`` on the
zero-mean signal, so ``beta`` ships with the branch file (standalone use:
``out = m(x - mean) * beta + mean``). Sanity check: the extracted branch
state must strict-load into a fresh ``SRHnet``.

Usage:
    python tools/extract_team17_branches.py \
        --checkpoint checkpoints/17_SRHCnet_39000_final.pth \
        --out-dir checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.nafnet import NAFNet  # noqa: E402
from models.srhcnet import SRHnet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="checkpoints/17_SRHCnet_39000_final.pth")
    parser.add_argument("--out-dir", default="checkpoints")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["params"]

    hat_sd = {k[len("srhnet."):]: v for k, v in ckpt.items() if k.startswith("srhnet.")}
    naf_sd = {k[len("srcnet."):]: v for k, v in ckpt.items() if k.startswith("srcnet.")}
    beta = ckpt["beta"]
    gamma = ckpt["gamma"]

    # Sanity (spec A.3): strict-load both branch states into fresh modules.
    hat = SRHnet(
        embed_dim=180, depths=(6,) * 12, num_heads=(6,) * 12, window_size=16,
        compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
        mlp_ratio=2, upscale=4,
    )
    hat.load_state_dict(hat_sd, strict=True)
    naf = NAFNet(width=64, enc_blk_nums=(2, 2, 4, 8), middle_blk_num=12,
                 dec_blk_nums=(2, 2, 2, 2))
    naf.load_state_dict(naf_sd, strict=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"params": hat_sd, "beta": beta}, out_dir / "hatl_2M_branch.pth")
    torch.save({"params": naf_sd, "gamma": gamma}, out_dir / "nafnet_refiner.pth")
    print(f"HAT-L branch: {len(hat_sd)} keys, beta {beta.flatten().tolist()}")
    print(f"NAFNet refiner: {len(naf_sd)} keys, gamma {gamma.flatten().tolist()}")
    print(f"wrote {out_dir / 'hatl_2M_branch.pth'} and {out_dir / 'nafnet_refiner.pth'}")


if __name__ == "__main__":
    main()

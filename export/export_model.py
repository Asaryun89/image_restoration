"""Export the best (EMA) generator to TorchScript and/or ONNX.

Usage:
    python export/export_model.py --checkpoint checkpoints/stage2/best_psnr.pth --formats onnx torchscript

Note: the exported graphs bake in the "input already padded" branch, so
inputs to the exported models should have height/width that are multiples
of 16 (the NAFNet padder size). The ONNX graph has dynamic batch/height/width axes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.checkpoint import CheckpointManager  # noqa: E402
from models import build_generator  # noqa: E402


def export_torchscript(model: torch.nn.Module, dummy: torch.Tensor, out_dir: Path) -> None:
    """Trace the model and verify traced ~= eager (max abs diff < 1e-4)."""
    traced = torch.jit.trace(model, dummy)
    path = out_dir / "model_ts.pt"
    traced.save(str(path))
    with torch.no_grad():
        diff = (traced(dummy) - model(dummy)).abs().max().item()
    if diff >= 1e-4:
        raise RuntimeError(f"TorchScript verification failed: max abs diff {diff:.2e}")
    print(f"TorchScript -> {path} (max abs diff {diff:.2e})")


def export_onnx(model: torch.nn.Module, dummy: torch.Tensor, out_dir: Path, opset: int) -> None:
    """Export with dynamic batch/height/width axes and verify with onnxruntime."""
    path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"ONNX -> {path} (opset {opset})")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping ONNX verification")
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for size in (dummy.shape[-1], dummy.shape[-1] + 64):  # also test a dynamic size
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"input": x.numpy()})[0]
        diff = abs(out - ref).max()
        if diff >= 1e-3:
            raise RuntimeError(f"ONNX verification failed at {size}px: max abs diff {diff:.2e}")
        print(f"  verified {size}x{size}: max abs diff {diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the EMA generator")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--formats", nargs="+", default=["onnx", "torchscript"],
        choices=["onnx", "torchscript"],
    )
    parser.add_argument("--size", type=int, default=256, help="dummy input side (multiple of 16)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-dir", default=None, help="defaults to the checkpoint directory")
    args = parser.parse_args()

    state = CheckpointManager.load(args.checkpoint)
    cfg = state["cfg"]
    model = build_generator(cfg["model"])
    model.load_state_dict(state.get("ema") or state["generator"])
    model.eval()
    if cfg["model"].get("arch", "nafnet") in ("hmanet", "srhmcnet"):
        # The HMANet shifted-window mask is recomputed per input size in
        # Python, so traced/ONNX graphs are only valid at the traced size.
        print("note: HMANet-based exports are fixed to the traced input size")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size)

    if "torchscript" in args.formats:
        export_torchscript(model, dummy, out_dir)
    if "onnx" in args.formats:
        export_onnx(model, dummy, out_dir, args.opset)


if __name__ == "__main__":
    main()

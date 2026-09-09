"""
ISBD Auto-Export & Optimization Engine
Exports trained models to:
1. TorchScript (.pt) — fast JIT execution, zero python overhead
2. ONNX format (.onnx) — universal format for web, mobile, TensorRT, OpenVINO
"""
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def export_all_formats(model: torch.nn.Module, step: int = 0, sample_size: int = 64) -> dict:
    """
    Exports the model to TorchScript and ONNX (if available).
    Returns dict of exported paths and benchmark results.
    """
    CKPT.mkdir(exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, sample_size, sample_size, device=device)

    results = {"step": step, "exports": {}}

    # 1. TorchScript Export (Traced JIT)
    try:
        traced = torch.jit.trace(model, dummy_input)
        scripted_path = CKPT / "best_traced.pt"
        traced.save(str(scripted_path))
        results["exports"]["torchscript"] = str(scripted_path)
    except Exception as e:
        results["exports"]["torchscript_error"] = str(e)

    # 2. ONNX Export
    try:
        onnx_path = CKPT / "best.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input_image"],
            output_names=["output_image"],
            dynamic_axes={"input_image": {0: "batch_size"}, "output_image": {0: "batch_size"}},
            opset_version=14,
        )
        results["exports"]["onnx"] = str(onnx_path)
    except Exception as e:
        results["exports"]["onnx_error"] = str(e)

    return results

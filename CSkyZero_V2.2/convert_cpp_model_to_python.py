from pathlib import Path
from typing import Dict, Optional

import torch


BASE_DIR = Path(__file__).resolve().parent
# Edit these relative paths directly before running.
INPUT_PATH = Path("data/gomoku/models/gomoku_model_2026-03-30_09-40-27.pth")
OUTPUT_PATH = Path("SkyZero_V2.1-main/gomoku_model_from_cpp_state_dict.pth")
AS_CHECKPOINT = False


def _extract_state_dict_from_dict(obj: dict) -> Optional[Dict[str, torch.Tensor]]:
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = obj.get(key)
        if isinstance(candidate, dict):
            tensor_sd = {k: v for k, v in candidate.items() if torch.is_tensor(v)}
            if tensor_sd:
                return tensor_sd

    tensor_items = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    if not tensor_items:
        return None

    if any(k.startswith("model.") for k in tensor_items):
        return {k[6:] if k.startswith("model.") else k: v for k, v in tensor_items.items()}

    return tensor_items


def _try_load_torchscript_state_dict(input_path: str) -> Optional[Dict[str, torch.Tensor]]:
    try:
        module = torch.jit.load(input_path, map_location="cpu")
    except Exception:
        return None

    state_dict = module.state_dict()
    tensor_sd = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    return tensor_sd or None


def load_state_dict(input_path: str):
    jit_sd = _try_load_torchscript_state_dict(input_path)
    if jit_sd:
        return jit_sd, "torchscript"

    obj = torch.load(input_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        extracted = _extract_state_dict_from_dict(obj)
        if extracted:
            return extracted, "python"

    state_dict_fn = getattr(obj, "state_dict", None)
    if callable(state_dict_fn):
        try:
            state_dict = state_dict_fn()
            if isinstance(state_dict, dict):
                tensor_sd = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
                if tensor_sd:
                    return tensor_sd, "module"
        except Exception:
            pass

    raise ValueError(
        "Unsupported input format: expected torchscript, state_dict, or checkpoint containing model weights"
    )


def convert(input_path: str, output_path: str, as_checkpoint: bool = False):
    state_dict, source = load_state_dict(input_path)
    py_sd = {k: v.detach().cpu() for k, v in state_dict.items() if torch.is_tensor(v)}

    if not py_sd:
        raise ValueError("No tensor weights found in input")

    if as_checkpoint:
        torch.save({"model_state_dict": py_sd}, output_path, _use_new_zipfile_serialization=True)
        print(f"[cpp->py] source={source}")
        print(f"Saved Python checkpoint (model_state_dict) to: {output_path}")
    else:
        torch.save(py_sd, output_path, _use_new_zipfile_serialization=True)
        print(f"[cpp->py] source={source}")
        print(f"Saved Python state_dict model to: {output_path}")
    print(f"Tensor count: {len(py_sd)}")


def main():
    input_path = BASE_DIR / INPUT_PATH
    output_path = BASE_DIR / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert(str(input_path), str(output_path), as_checkpoint=AS_CHECKPOINT)


if __name__ == "__main__":
    main()

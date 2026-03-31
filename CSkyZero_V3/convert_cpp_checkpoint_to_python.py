from pathlib import Path
from typing import Any, Dict

import torch

from convert_cpp_model_to_python import load_state_dict


BASE_DIR = Path(__file__).resolve().parent
# Edit these relative paths directly before running.
INPUT_PATH = Path("data/gomoku/checkpoints/gomoku_checkpoint_2026-03-30_09-40-27.ckpt")
OUTPUT_PATH = Path("SkyZero_V2.1-main/gomoku_checkpoint_from_cpp.ckpt")


def convert(input_path: str, output_path: str):
    state_dict, source = load_state_dict(input_path)
    py_sd = {k: v.detach().cpu() for k, v in state_dict.items() if torch.is_tensor(v)}

    if not py_sd:
        raise ValueError("No tensor weights found in input")

    out_ckpt: Dict[str, Any] = {"model_state_dict": py_sd}
    torch.save(out_ckpt, output_path, _use_new_zipfile_serialization=True)
    print(f"[cpp->py-checkpoint] source={source}")
    print(f"Saved Python checkpoint to: {output_path}")
    print(f"Tensor count: {len(py_sd)}")


def main():
    input_path = BASE_DIR / INPUT_PATH
    output_path = BASE_DIR / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert(str(input_path), str(output_path))


if __name__ == "__main__":
    main()

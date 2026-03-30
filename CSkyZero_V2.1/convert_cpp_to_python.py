from pathlib import Path

from convert_cpp_checkpoint_to_python import convert as convert_cpp_to_py_checkpoint
from convert_cpp_model_to_python import convert as convert_cpp_to_py_model


BASE_DIR = Path(__file__).resolve().parent
# Edit these relative paths directly before running.
INPUT_PATH = Path("data/gomoku/models/gomoku_model_2026-03-30_09-40-27.pth")
OUTPUT_PATH = Path("SkyZero_V2.1-main/gomoku_model_from_cpp_state_dict.pth")
KIND = "model"  # model | checkpoint
AS_CHECKPOINT = False


def main():
    input_path = BASE_DIR / INPUT_PATH
    output_path = BASE_DIR / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if KIND == "checkpoint":
        convert_cpp_to_py_checkpoint(str(input_path), str(output_path))
        return

    convert_cpp_to_py_model(str(input_path), str(output_path), as_checkpoint=AS_CHECKPOINT)


if __name__ == "__main__":
    main()

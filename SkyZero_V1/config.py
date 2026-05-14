"""
Configuration loader for SkyZero training scripts.
Reads TOML .cfg files and returns a flat dict compatible with train_args.

Usage:
    from config import load_config
    train_args = load_config("tictactoe/tictactoe.cfg")

Supports: Python 3.11+ (built-in tomllib) or pip install toml.
"""

import sys
import os

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore
    except ImportError:
        print("tomllib not available; try: pip install toml")
        sys.exit(1)


def load_config(cfg_path: str) -> dict:
    """Load a TOML config file and return a flat dict of train_args."""
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.getcwd(), cfg_path)

    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)

    # Flatten: merge all [section] tables into a single dict
    flat = {}
    for section_name, section in raw.items():
        if not isinstance(section, dict):
            flat[section_name] = section
            continue
        for key, value in section.items():
            if key in flat:
                print(f"Warning: duplicate key '{key}' in [{section_name}]")
            flat[key] = value

    return flat

"""Centered, color-tagged log prefixes for the training pipeline.

Mirrors scripts/internal/log_common.sh: title-case stage names centered in an
8-wide field. Color auto-disables when the target stream is not a TTY or
NO_COLOR is set, so redirected logs (tee / nohup) stay plain text.

    from log_util import tag
    TAG = tag("Train")
    print(f"{TAG} step=...")
"""
import os
import sys

_COLORS = {
    "Run": "1;36",
    "SelfPlay": "32",
    "Daemon": "92",
    "Shuffle": "34",
    "Target": "94",
    "Schedule": "35",
    "Train": "36",
    "Export": "95",
    "Probe": "33",
    "GameInfo": "33",
}


def tag(name: str, stream=sys.stdout) -> str:
    """Bracketed stage tag, colored if `stream` is a TTY."""
    plain = f"[{name}]"
    use_color = (
        getattr(stream, "isatty", None)
        and stream.isatty()
        and not os.environ.get("NO_COLOR")
    )
    if use_color:
        return f"\033[{_COLORS.get(name, '0')}m{plain}\033[0m"
    return plain

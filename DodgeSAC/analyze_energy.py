"""Quantify the 'energy' behavior from a replay: does the agent rest when safe and
move when threatened?

Reads a <name>_play.js (window.REPLAY) and reports, over all frames:
  - mean speed |mv| and rest-fraction (|mv| < 0.3)
  - mean speed split by nearest-threat distance: SAFE (>180px) vs DANGER (<90px)

The energy principle holds when speed_SAFE << speed_DANGER (rests when safe,
sprints when threatened).

    python analyze_energy.py play_sac_sp02.js [play_sac_base.js ...]
"""

import json
import math
import sys

SAFE_D, DANGER_D = 180.0, 90.0


def load(path):
    t = open(path).read()
    return json.loads(t[t.index("{"):t.rstrip().rstrip(";").rindex("}") + 1])


def nearest_threat(f):
    px, py = f["px"], f["py"]
    best = 1e9
    for p in f["pr"]:
        d = math.hypot(p["x"] - px, p["y"] - py)
        if d < best:
            best = d
    for b in f["bo"]:                      # bombs count as threats (blast radius)
        d = math.hypot(b["x"] - px, b["y"] - py) - b["br"]
        best = min(best, max(0.0, d))
    return best


def analyze(path):
    d = load(path)
    speeds, safe_sp, danger_sp = [], [], []
    for ep in d["episodes"]:
        for f in ep["frames"]:
            s = math.hypot(f["ax"], f["ay"])
            speeds.append(s)
            nd = nearest_threat(f)
            if nd > SAFE_D:
                safe_sp.append(s)
            elif nd < DANGER_D:
                danger_sp.append(s)
    mean = lambda x: sum(x) / len(x) if x else float("nan")
    rest = sum(1 for s in speeds if s < 0.3) / len(speeds)
    print(f"{path}:")
    print(f"  mean|v|={mean(speeds):.2f}  rest%(<0.3)={rest:.0%}  "
          f"|  speed SAFE(>{SAFE_D:.0f}px)={mean(safe_sp):.2f} ({len(safe_sp)}f)  "
          f"DANGER(<{DANGER_D:.0f}px)={mean(danger_sp):.2f} ({len(danger_sp)}f)")
    if safe_sp and danger_sp:
        print(f"  -> energy principle: speed_safe/speed_danger = {mean(safe_sp)/mean(danger_sp):.2f} "
              f"({'GOOD: rests when safe' if mean(safe_sp) < 0.7 * mean(danger_sp) else 'weak: moves regardless'})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)

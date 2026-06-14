#!/usr/bin/env bash
# ============================================================================
# autoexp.sh — run every sub-experiment under an umbrella CONFIG_DIR, in turn.
#
# An "umbrella" config dir holds one subdir per experiment, each a full
# CONFIG_DIR (run.cfg + paths.cfg + ...). This driver runs run.sh once per
# subdir, sequentially, so a hyper-parameter sweep finishes unattended.
#
#   Launch:   CONFIG_DIR=configs/GumbelVSPUCT bash scripts/autoexp.sh
#   Optional: MAX_ITERS=<n>  cap every run at n iters (passed as run.sh's CLI
#             arg). This is the ONLY stop you can inject from outside, because
#             run.cfg's plain `MAX_TIME_SECONDS=...` clobbers any env value.
#
# Each run MUST terminate on its own, or the sweep would hang on experiment #1
# forever — run.sh stops only on MAX_TIME_SECONDS>0 (run.cfg) or a max_iters CLI
# arg. So before launching ANYTHING we PROBE each sub-experiment's *effective*
# MAX_TIME_SECONDS by sourcing its run.cfg exactly as run.sh does (later
# assignments win, so a stray `MAX_TIME_SECONDS=0` further down the file
# silently overrides one set at the top). If a sub-experiment has neither a
# positive MAX_TIME_SECONDS nor a MAX_ITERS override, we refuse to start and
# say which one — better than discovering the hang three hours in.
#
# Resumable: a finished run drops EXPERIMENT_DONE in its DATA_DIR; re-launching
# skips it. A Ctrl+C'd / failed run leaves no marker, so re-launching RESUMES it
# (run.sh itself resumes from state.json; the MAX_TIME_SECONDS budget is
# cumulative across sessions). To force a redo, delete that experiment's
# DATA_DIR.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

CONFIG_DIR="${CONFIG_DIR:?set CONFIG_DIR to an umbrella config dir, e.g. CONFIG_DIR=configs/GumbelVSPUCT}"
[[ "$CONFIG_DIR" = /* ]] || CONFIG_DIR="$ROOT/$CONFIG_DIR"
[[ -d "$CONFIG_DIR" ]] || { echo "[autoexp] no config dir at $CONFIG_DIR" >&2; exit 1; }
MAX_ITERS="${MAX_ITERS:-}"

# ---- discover sub-experiments: immediate subdirs that contain a run.cfg -----
mapfile -t EXP_DIRS < <(
    find "$CONFIG_DIR" -mindepth 1 -maxdepth 1 -type d \
        -exec test -f '{}/run.cfg' ';' -print | sort
)
if [[ "${#EXP_DIRS[@]}" -eq 0 ]]; then
    echo "[autoexp] no sub-experiments under $CONFIG_DIR" >&2
    echo "[autoexp] expected one subdir per experiment, each containing run.cfg." >&2
    exit 1
fi

# ---- probe each sub-experiment: effective MAX_TIME_SECONDS + DATA_DIR --------
# Source run.cfg(+.local)+paths.cfg in a subshell, exactly like run.sh, so the
# value we read is the one run.sh will actually use.
probe() {  # <cfgdir> -> "<max_time_seconds>\t<data_dir>"
    ( set -a
      ROOT="$ROOT"
      # shellcheck disable=SC1090
      source "$1/run.cfg" >/dev/null 2>&1
      [[ -f "$1/run.cfg.local" ]] && source "$1/run.cfg.local" >/dev/null 2>&1
      # shellcheck disable=SC1090
      source "$1/paths.cfg" >/dev/null 2>&1
      printf '%s\t%s\n' "${MAX_TIME_SECONDS:-0}" "${DATA_DIR:-}" )
}

echo "[autoexp] umbrella: $CONFIG_DIR"
echo "[autoexp] ${#EXP_DIRS[@]} sub-experiment(s); MAX_ITERS=${MAX_ITERS:-<none>}"
EXP_NAMES=(); EXP_DATA=(); bad=0
for dir in "${EXP_DIRS[@]}"; do
    name="$(basename "$dir")"
    IFS=$'\t' read -r maxt data < <(probe "$dir") || true
    [[ "$maxt" =~ ^[0-9]+$ ]] || maxt=0
    EXP_NAMES+=("$name"); EXP_DATA+=("$data")
    if [[ "$maxt" -gt 0 ]]; then
        echo "  [ok]   $name  stop=MAX_TIME_SECONDS=${maxt}s  DATA_DIR=$data"
    elif [[ -n "$MAX_ITERS" ]]; then
        echo "  [ok]   $name  stop=MAX_ITERS=$MAX_ITERS (MAX_TIME_SECONDS=0)  DATA_DIR=$data"
    else
        echo "  [BAD]  $name  NO stop condition: MAX_TIME_SECONDS=0 and no MAX_ITERS"
        bad=1
    fi
done

if [[ "$bad" -ne 0 ]]; then
    cat >&2 <<EOF

[autoexp] aborting: the [BAD] experiment(s) above would run forever (run.sh
          only stops on MAX_TIME_SECONDS>0 or a max_iters CLI arg). Fix either:
            * set a positive MAX_TIME_SECONDS in that experiment's run.cfg
              (and make sure no LATER line in the file resets it to 0), or
            * re-run with a cap:
                MAX_ITERS=<n> CONFIG_DIR=$CONFIG_DIR bash scripts/autoexp.sh
EOF
    exit 1
fi

# ---- run each sub-experiment sequentially -----------------------------------
LOG_ROOT="$ROOT/autoexp/$(basename "$CONFIG_DIR")"
mkdir -p "$LOG_ROOT"
total=${#EXP_NAMES[@]}
overall_start=$(date +%s)

for i in "${!EXP_NAMES[@]}"; do
    name="${EXP_NAMES[$i]}"; dir="${EXP_DIRS[$i]}"; data="${EXP_DATA[$i]}"
    marker="$data/EXPERIMENT_DONE"
    log="$LOG_ROOT/${name}.runner.log"
    n=$(( i + 1 ))

    if [[ -f "$marker" ]]; then
        echo "[autoexp] ($n/$total) SKIP  $name — already done ($marker)"
        continue
    fi

    echo "[autoexp] ($n/$total) START $name"
    echo "[autoexp]        CONFIG_DIR=$dir"
    echo "[autoexp]        DATA_DIR=$data  log=$log"
    run_start=$(date +%s)

    # run.sh self-terminates on its MAX_TIME_SECONDS / MAX_ITERS. Don't let one
    # failed run abort the whole sweep; Ctrl+C (rc 130) DOES stop the sweep.
    set +e
    CONFIG_DIR="$dir" bash "$SCRIPT_DIR/run.sh" ${MAX_ITERS:+"$MAX_ITERS"} 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -eq 0 ]]; then
        mkdir -p "$data"; touch "$marker"
        echo "[autoexp] ($n/$total) DONE  $name in $(( $(date +%s) - run_start ))s"
    elif [[ "$rc" -eq 130 ]]; then
        echo "[autoexp] ($n/$total) INTERRUPTED $name (rc=130) — stopping sweep; re-launch to resume." >&2
        exit 130
    else
        echo "[autoexp] ($n/$total) FAILED $name rc=$rc — see $log; continuing to next." >&2
    fi
done

echo "[autoexp] sweep finished in $(( $(date +%s) - overall_start ))s."
echo "[autoexp] per-experiment logs under: $LOG_ROOT/"

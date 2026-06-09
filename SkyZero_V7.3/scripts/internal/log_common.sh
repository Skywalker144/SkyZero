#!/usr/bin/env bash
# Shared logging helpers: centered, color-tagged stage prefixes for the
# training pipeline. Sourced by run.sh and internal/*.sh.
#
#   echo "$(_tag Run) message"   ->  [  Run   ] message
#
# Color auto-disables when stdout is not a TTY or NO_COLOR is set, so
# redirected logs (tee / nohup) stay plain text.

if [[ -z "${_LOG_COMMON_LOADED:-}" ]]; then
_LOG_COMMON_LOADED=1

_LOG_COLOR=1
if [[ -n "${NO_COLOR:-}" || ! -t 1 ]]; then _LOG_COLOR=0; fi

declare -gA _TAG_COLOR=(
    [Run]='1;36'      # bold cyan  — loop banners
    [SelfPlay]='32'   # green
    [Daemon]='92'     # bright green
    [Shuffle]='34'    # blue
    [Bucket]='94'     # bright blue
    [Schedule]='35'   # magenta
    [Train]='36'      # cyan
    [Export]='95'     # bright magenta
    [Probe]='33'      # yellow
    [GameInfo]='33'   # yellow
)

# Bracketed stage tag, colored if stdout is a TTY. Usage: echo "$(_tag Run) msg".
_tag() {
    local name="$1"
    if [[ "$_LOG_COLOR" == 1 ]]; then
        printf '\033[%sm[%s]\033[0m' "${_TAG_COLOR[$name]:-0}" "$name"
    else
        printf '[%s]' "$name"
    fi
}
fi

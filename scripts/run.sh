#!/usr/bin/env bash
# Canonical way to run anything long, so its output is actually visible.
#
#   ./scripts/run.sh <logname> python scripts/whatever.py --args
#
# Three separate things buffer output and each one hid a healthy job from us
# today:
#   * Python buffers stdout when it is not a terminal   -> PYTHONUNBUFFERED
#   * grep buffers when writing to a pipe               -> --line-buffered
#   * ssh and nohup add their own buffering             -> tee to a real file
#
# The log file is the contract: tail it from anywhere, any time, and it is
# current. Never pipe a long job straight through grep again.
set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <logname> <command...>" >&2
  exit 2
fi

NAME=$1; shift
LOGDIR=${LOGDIR:-logs}
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${NAME}.log"

export PYTHONUNBUFFERED=1
export PYTHONPATH=${PYTHONPATH:-.}

{
  echo "=== $NAME started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=== $*"
} | tee -a "$LOG"

# stdbuf keeps the command line-buffered; grep --line-buffered keeps the filter
# from holding lines back; tee puts them on disk immediately.
stdbuf -oL -eL "$@" 2>&1 \
  | grep --line-buffered -v -e "UserWarning" -e "warnings.warn" -e "sparse_csr" \
  | tee -a "$LOG"

STATUS=${PIPESTATUS[0]}
echo "=== $NAME finished $(date -u +%H:%M:%SZ) exit $STATUS" | tee -a "$LOG"
exit "$STATUS"

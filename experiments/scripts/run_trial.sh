#!/usr/bin/env bash
# Launch one training_run.py trial against an existing (already-provisioned/adopted) Colab
# session, capturing output to experiments/logs/<trial-name>.log.
#
# Robustness: the underlying colab exec connection has been observed to die mid-run
# unpredictably (mid-epoch or between epochs), independent of wall-clock length. Rather than
# treat that as a hard failure, this script retries automatically: after each attempt it scans
# the log for the highest completed epoch and, if short of the target, launches the next
# attempt with --resume-from-revision/--start-epoch picking up from there (checkpoints are
# pushed after every epoch, so at most one epoch of progress is ever repeated). Before each
# retry it first checks the session is still alive; if the runtime itself is gone, it stops
# retrying immediately (further attempts would just fail on upload) rather than burning the
# full retry budget on a dead session.
#
# Usage:
#   run_trial.sh <trial-name> <session-name> [--timeout SECONDS] [--max-attempts N] -- <training_run.py args...>
#
# Example:
#   run_trial.sh trial-016 claude_cafa6 -- \
#     --model-name facebook/esm2_t6_8M_UR50D --epochs 10 --unfreeze-last-n 2 \
#     --learning-rate 2e-4 --weight-decay 0.5 --max-train-samples 12000 --max-val-samples 2000 \
#     --seed 42
#
# --run-name, --checkpoint-branch, and --hf-token-file are always appended automatically;
# don't pass them (or --epochs/--start-epoch/--resume-from-revision) again after --.
set -uo pipefail  # no -e: a failed attempt should trigger our own retry logic, not abort the script

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <trial-name> <session-name> [--timeout SECONDS] [--max-attempts N] -- <training_run.py args...>" >&2
  exit 1
fi

TRIAL_NAME="$1"
SESSION_NAME="$2"
shift 2

TIMEOUT=3600
MAX_ATTEMPTS=6
while [ "$1" = "--timeout" ] || [ "$1" = "--max-attempts" ]; do
  if [ "$1" = "--timeout" ]; then TIMEOUT="$2"; else MAX_ATTEMPTS="$2"; fi
  shift 2
done

if [ "$1" != "--" ]; then
  echo "Expected -- before training_run.py args, got: $1" >&2
  exit 1
fi
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/share/google-cloud-sdk/bin:$PATH"

mkdir -p .podman/tmp experiments/logs
LOG="experiments/logs/${TRIAL_NAME}.log"
: > "$LOG"

TARGET_EPOCHS=1
prev=""
for i in "$@"; do
  if [ "$prev" = "--epochs" ]; then TARGET_EPOCHS="$i"; fi
  prev="$i"
done

completed=0
attempt=1

while [ "$attempt" -le "$MAX_ATTEMPTS" ] && [ "$completed" -lt "$TARGET_EPOCHS" ]; do
  echo "[run_trial] === attempt $attempt/$MAX_ATTEMPTS for $TRIAL_NAME (completed=$completed/$TARGET_EPOCHS) ===" | tee -a "$LOG"

  if [ "$attempt" -gt 1 ]; then
    if ! colab --auth=adc status -s "$SESSION_NAME" 2>&1 | grep -q "IDLE\|BUSY"; then
      echo "[run_trial] session '$SESSION_NAME' is not alive; stopping retries for $TRIAL_NAME (needs reconnection)." | tee -a "$LOG"
      break
    fi
  fi

  WRAPPER=".podman/tmp/${TRIAL_NAME}_attempt${attempt}_wrapper.py"
  python3 experiments/scripts/make_wrapper.py "$WRAPPER" "$TRIAL_NAME" "$TARGET_EPOCHS" "$completed" "$@" | tee -a "$LOG"

  colab --auth=adc upload .podman/secrets/huggingface_token /content/.hf_token -s "$SESSION_NAME" >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[run_trial] token upload failed; session likely dead. Stopping retries for $TRIAL_NAME." | tee -a "$LOG"
    break
  fi
  colab --auth=adc upload training_run.py /content/training_run.py -s "$SESSION_NAME" >>"$LOG" 2>&1

  colab --auth=adc exec -s "$SESSION_NAME" --timeout "$TIMEOUT" -f "$WRAPPER" >>"$LOG" 2>&1

  new_completed=$(grep -oE '^Epoch [0-9]+/[0-9]+' "$LOG" | grep -oE '[0-9]+' | awk 'NR%2==1' | sort -n | tail -1)
  new_completed=${new_completed:-0}
  if [ "$new_completed" -gt "$completed" ]; then
    completed=$new_completed
  else
    echo "[run_trial] attempt $attempt made no epoch progress (still at $completed)." | tee -a "$LOG"
  fi

  attempt=$((attempt + 1))
done

if [ "$completed" -ge "$TARGET_EPOCHS" ]; then
  echo "[run_trial] $TRIAL_NAME COMPLETE ($completed/$TARGET_EPOCHS epochs, $((attempt - 1)) attempt(s))" | tee -a "$LOG"
  exit 0
else
  echo "[run_trial] $TRIAL_NAME INCOMPLETE ($completed/$TARGET_EPOCHS epochs after $((attempt - 1)) attempt(s))" | tee -a "$LOG"
  exit 1
fi

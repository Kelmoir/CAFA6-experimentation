#!/usr/bin/env bash
# Thin retry wrapper for local mf-span trials: relaunches train_mf_span_model.py automatically if
# it exits non-zero (e.g. the CUDA Xid 13 GPU exceptions documented in
# experiments/mf_span_sweep_log.md's trial 16/18 crashes), resuming from the latest checkpoint
# instead of restarting the whole trial from scratch. Mirrors the Colab CLI self-healing retry
# pattern (run_trial.sh/make_wrapper.py, see CLAUDE.md's "Colab regularization sweep"), adapted
# for a local bash retry loop instead of a colab exec relaunch -- there's no session/VM to
# provision here, just a process to relaunch with --resume-from-checkpoint added.
#
# NOTE ON LIMITS: this only helps if the GPU/driver recovers on its own after the crash. A Xid
# fault that wedges the device (nvidia-smi stops responding, as seen once already -- see the
# session's earlier diagnosis) needs an actual reboot before ANY CUDA process, retried or not,
# will work again. The wrapper still retries in that case -- it's cheap and harmless -- but if
# every attempt fails within a few seconds rather than a few hundred, that's the likely reason;
# check `nvidia-smi` by hand before assuming the wrapper itself is broken.
#
# Usage: pass the exact same arguments you'd give train_mf_span_model.py directly, with one
# addition -- --run-name is REQUIRED (used to locate models/<run-name>/latest_checkpoint for
# resume, same path train_mf_span_model.py itself defaults to when --resume-from-checkpoint is
# unset -- see latest_checkpoint_dir in main()). Do not pass --resume-from-checkpoint yourself;
# the wrapper adds it on retries.
#
# Example:
#   experiments/scripts/run_mf_span_trial.sh --run-name mf-span-reg-trial-19 \
#     --batch-size 4 --grad-accum-steps 8 --learning-rate 1e-4 --label-smoothing 0.3 \
#     --epochs 30 --early-stop-patience 5 --push-to-hub --hub-branch trial-19
#
# MAX_RETRIES env var overrides the default retry cap (5).

set -uo pipefail

MAX_RETRIES=${MAX_RETRIES:-5}
ARGS=("$@")

RUN_NAME=""
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--run-name" ]]; then
    RUN_NAME="${ARGS[$((i + 1))]}"
  fi
done
if [[ -z "$RUN_NAME" ]]; then
  echo "[WRAPPER] --run-name is required (used to locate the latest checkpoint for resume)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_mf_span_model.py"
LATEST_CKPT="models/${RUN_NAME}/latest_checkpoint"

attempt=0
while true; do
  attempt=$((attempt + 1))
  if [[ $attempt -gt 1 && -f "${LATEST_CKPT}/training_state.pt" ]]; then
    echo "[WRAPPER] attempt ${attempt}/${MAX_RETRIES}: resuming from ${LATEST_CKPT}"
    python "$TRAIN_SCRIPT" "${ARGS[@]}" --resume-from-checkpoint "$LATEST_CKPT"
  else
    if [[ $attempt -gt 1 ]]; then
      echo "[WRAPPER] attempt ${attempt}/${MAX_RETRIES}: no ${LATEST_CKPT}/training_state.pt yet" \
           "(crashed before epoch 1 finished?) -- restarting from scratch"
    else
      echo "[WRAPPER] attempt ${attempt}/${MAX_RETRIES}: starting fresh"
    fi
    python "$TRAIN_SCRIPT" "${ARGS[@]}"
  fi
  status=$?

  if [[ $status -eq 0 ]]; then
    echo "[WRAPPER] exited 0 -- done."
    exit 0
  fi

  echo "[WRAPPER] exited with status ${status} (attempt ${attempt}/${MAX_RETRIES})"
  if [[ $attempt -ge $MAX_RETRIES ]]; then
    echo "[WRAPPER] giving up after ${MAX_RETRIES} attempts." >&2
    exit "$status"
  fi
  sleep 5
done

#!/usr/bin/env bash
# Thin retry wrapper for local mf-span trials: relaunches train_mf_span_model.py automatically if
# it exits non-zero (e.g. the CUDA Xid 13 GPU exceptions documented in
# experiments/mf_span/mf_span_sweep_log.md's trial 16/18 crashes), resuming from the latest checkpoint
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
# RESUMING A *COMPLETED* RUN TO TRAIN MORE EPOCHS: the same models/<run-name>/latest_checkpoint
# this wrapper uses for crash retries also works to extend a run that already finished, past its
# original --epochs -- e.g. trial 20 ran its full 30 epochs still improving (see
# experiments/mf_span/mf_span_sweep_log.md); to train 10 more instead of redoing the first 30, rerun this
# wrapper with trial 20's exact original args, --run-name mf-span-reg-trial-20 (unchanged, so it
# finds that same latest_checkpoint), --epochs 40, and --reset-early-stop-on-resume (see
# train_mf_span_model.py's --help -- without it, --early-stop-patience keeps the streak count from
# the old, shorter target, which can stop the extension after just a couple more epochs). The
# wrapper checks for an existing checkpoint on EVERY attempt, including the first -- not just
# retries -- so this "first attempt, but a checkpoint already exists" case resumes automatically,
# the same as a genuine crash retry would.
#
# Usage: pass the exact same arguments you'd give train_mf_span_model.py directly, with one
# addition -- --run-name is REQUIRED (used to locate models/<run-name>/latest_checkpoint for
# resume, same path train_mf_span_model.py itself defaults to when --resume-from-checkpoint is
# unset -- see latest_checkpoint_dir in main()). Do not pass --resume-from-checkpoint yourself;
# the wrapper adds it whenever a checkpoint for --run-name already exists, whether that's a crash's
# leftover or a prior completed run's.
#
# Example (fresh trial):
#   experiments/mf_span/scripts/run_mf_span_trial.sh --run-name mf-span-reg-trial-19 \
#     --batch-size 4 --grad-accum-steps 8 --learning-rate 1e-4 --label-smoothing 0.3 \
#     --epochs 30 --early-stop-patience 5 --push-to-hub --hub-branch trial-19
#
# Example (extend a completed run):
#   experiments/mf_span/scripts/run_mf_span_trial.sh --run-name mf-span-reg-trial-20 \
#     --batch-size 2 --grad-accum-steps 16 --learning-rate 1e-4 --unfreeze-last-n 3 \
#     --label-smoothing 0.3 --epochs 40 --early-stop-patience 5 --reset-early-stop-on-resume \
#     --push-to-hub --hub-branch trial-20
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

# HF_HUB_DISABLE_XET: CLAUDE.md's documented hf_xet/huggingface_hub version-mismatch gotcha on this
# host -- exported (not embedded in the command string) so it actually reaches the Hub push at the
# end of a --push-to-hub run, in either branch below.
export HF_HUB_DISABLE_XET=1

# No hardcoded venv: the trials in mf_span_sweep_log.md are run from a `(base)` conda shell, not
# the env/ virtualenv CLAUDE.md documents for the main pipeline -- env/ doesn't even have torch
# installed on this host (verified directly). PYTHON_BIN defaults to whatever `python3` resolves
# to in the shell that invokes this wrapper, i.e. whatever's already active when you run it -- set
# PYTHON_BIN explicitly only if you need something other than that.
PYTHON_BIN="env/bin/python"

attempt=0
while true; do
  attempt=$((attempt + 1))
  # Checked on EVERY attempt, not just retries -- a checkpoint already sitting at ${LATEST_CKPT}
  # on the very first attempt means --run-name is being reused deliberately to continue that run
  # (either a prior completed trial being extended to more epochs, or -- less likely, since a
  # fresh invocation normally starts with no checkpoint at all -- a retry of this same wrapper
  # after it was killed and relaunched). Either way, resuming is the right call; there's no
  # "attempt 1 must be fresh" case worth special-casing separately from a mid-loop retry.
  if [[ -f "${LATEST_CKPT}/training_state.pt" ]]; then
    echo "[WRAPPER] attempt ${attempt}/${MAX_RETRIES}: resuming from ${LATEST_CKPT}"
    "$PYTHON_BIN" "$TRAIN_SCRIPT" "${ARGS[@]}" --resume-from-checkpoint "$LATEST_CKPT"
  else
    echo "[WRAPPER] attempt ${attempt}/${MAX_RETRIES}: no ${LATEST_CKPT}/training_state.pt yet" \
         "-- starting fresh"
    "$PYTHON_BIN" "$TRAIN_SCRIPT" "${ARGS[@]}"
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

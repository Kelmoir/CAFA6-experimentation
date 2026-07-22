#!/usr/bin/env bash
# Launch one training_run.py trial against an existing (already-provisioned/adopted) Colab
# session, capturing output to experiments/logs/<trial-name>.log.
#
# Usage:
#   run_trial.sh <trial-name> <session-name> [--timeout SECONDS] -- <training_run.py args...>
#
# Example:
#   run_trial.sh trial-016 claude_cafa6 -- \
#     --model-name facebook/esm2_t6_8M_UR50D --epochs 10 --unfreeze-last-n 2 \
#     --learning-rate 2e-4 --weight-decay 0.5 --max-train-samples 12000 --max-val-samples 2000 \
#     --seed 42
#
# --run-name, --checkpoint-branch, and --hf-token-file are always appended automatically;
# don't pass them again in the args after --.
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <trial-name> <session-name> [--timeout SECONDS] -- <training_run.py args...>" >&2
  exit 1
fi

TRIAL_NAME="$1"
SESSION_NAME="$2"
shift 2

TIMEOUT=3600
if [ "$1" = "--timeout" ]; then
  TIMEOUT="$2"
  shift 2
fi

if [ "$1" != "--" ]; then
  echo "Expected -- before training_run.py args, got: $1" >&2
  exit 1
fi
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/share/google-cloud-sdk/bin:$PATH"

mkdir -p .podman/tmp experiments/logs
WRAPPER=".podman/tmp/${TRIAL_NAME}_wrapper.py"

python3 - "$WRAPPER" "$TRIAL_NAME" "$@" <<'PYEOF'
import json
import sys

wrapper_path, trial_name, *extra_args = sys.argv[1:]

argv = ["training_run.py", *extra_args,
        "--run-name", trial_name,
        "--checkpoint-branch", trial_name,
        "--hf-token-file", "/content/.hf_token"]

with open(wrapper_path, "w") as f:
    f.write("import sys\n\n")
    f.write(f"sys.argv = {json.dumps(argv, indent=4)}\n\n")
    f.write(
        'exec(\n'
        '    compile(open("/content/training_run.py").read(), "/content/training_run.py", "exec"),\n'
        '    {"__name__": "__main__"},\n'
        ')\n'
    )
PYEOF

echo "[run_trial] wrapper written to $WRAPPER"
colab --auth=adc upload .podman/secrets/huggingface_token /content/.hf_token -s "$SESSION_NAME"
colab --auth=adc upload training_run.py /content/training_run.py -s "$SESSION_NAME"

LOG="experiments/logs/${TRIAL_NAME}.log"
echo "[run_trial] running trial '$TRIAL_NAME' on session '$SESSION_NAME' (timeout ${TIMEOUT}s), logging to $LOG"
set +e
colab --auth=adc exec -s "$SESSION_NAME" --timeout "$TIMEOUT" -f "$WRAPPER" > "$LOG" 2>&1
EXIT_CODE=$?
set -e
echo "EXEC_EXIT=$EXIT_CODE"
exit "$EXIT_CODE"

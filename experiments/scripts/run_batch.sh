#!/usr/bin/env bash
# Run a batch of trials back-to-back against one already-adopted session, then always
# stop the session at the end (success or failure) rather than leaving it idle — the
# runtime disconnects quickly once it notices nothing is happening anyway.
#
# Usage: run_batch.sh <session-name> <manifest.json>
#
# manifest.json: a JSON array of {"trial_id": "...", "args": ["--flag", "value", ...]}
set -uo pipefail  # no -e: one experiment failing shouldn't abort the rest of the batch

SESSION_NAME="$1"
MANIFEST="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/share/google-cloud-sdk/bin:$PATH"

cleanup() {
  echo "[run_batch] stopping session '$SESSION_NAME'..."
  colab --auth=adc stop -s "$SESSION_NAME" || true
}
trap cleanup EXIT

N=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
echo "[run_batch] running $N experiment(s) from $MANIFEST on session '$SESSION_NAME'"

for i in $(seq 0 $((N - 1))); do
  TRIAL_ID=$(python3 -c "import json; print(json.load(open('$MANIFEST'))[$i]['trial_id'])")
  mapfile -t ARGS < <(python3 -c "import json; [print(a) for a in json.load(open('$MANIFEST'))[$i]['args']]")
  echo "[run_batch] ($((i + 1))/$N) launching $TRIAL_ID"
  bash experiments/scripts/run_trial.sh "$TRIAL_ID" "$SESSION_NAME" -- "${ARGS[@]}"
  status=$?
  echo "[run_batch] ($((i + 1))/$N) $TRIAL_ID finished (exit $status)"
done

echo "[run_batch] batch complete."

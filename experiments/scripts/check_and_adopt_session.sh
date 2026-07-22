#!/usr/bin/env bash
# Verify a Colab runtime is reachable under <session-name>; if not, but exactly one
# unadopted assignment exists on the server, adopt it under that name and smoke-test it.
#
# Usage: check_and_adopt_session.sh <session-name>
set -euo pipefail

SESSION_NAME="${1:-claude_cafa6}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.local/share/google-cloud-sdk/bin:$PATH"

if colab --auth=adc status -s "$SESSION_NAME" 2>&1 | grep -q "IDLE\|BUSY"; then
  echo "[check_and_adopt] session '$SESSION_NAME' is already alive and adopted."
  exit 0
fi

echo "[check_and_adopt] session '$SESSION_NAME' not found locally; attempting to adopt from server assignments..."
"$HOME/.local/share/uv/tools/google-colab-cli/bin/python3" experiments/scripts/adopt_session.py "$SESSION_NAME"

echo "[check_and_adopt] running GPU smoke test..."
cat > .podman/tmp/smoke_check_adopted.py <<'PYEOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PYEOF
colab --auth=adc exec -s "$SESSION_NAME" --timeout 60 -f .podman/tmp/smoke_check_adopted.py

"""Adopt a browser-provisioned Colab runtime into the CLI's local session store.

CLI-created sessions hit a known upstream bug (googlecolab/google-colab-cli#14): the
keep-alive RPC returns 403 USER_PROJECT_DENIED for ordinary accounts, so Colab idle-prunes
the VM within minutes regardless of activity. A browser-owned runtime is kept alive by the
browser tab itself, sidestepping this entirely. This script finds the (single) active
assignment on the account and registers it locally under a given name so it can be driven
via the normal `colab -s <name> ...` commands.

Usage: run with the colab-cli tool's own interpreter (it needs the colab_cli package):
    "$HOME/.local/share/uv/tools/google-colab-cli/bin/python3" adopt_session.py <session-name>
"""

import sys

from colab_cli.auth import AuthProvider
from colab_cli.common import state
from colab_cli.state import SessionState

state.auth_provider = AuthProvider.ADC

target_name = sys.argv[1] if len(sys.argv) > 1 else "claude_cafa6"

assignments = state.client.list_assignments()
print(f"Found {len(assignments)} assignment(s) on the server.")
for a in assignments:
    print(f"  endpoint={a.endpoint} accelerator={a.accelerator.name} variant={a.variant.name}")

if not assignments:
    print("No assignments found - is the browser runtime still active?")
    sys.exit(1)

if len(assignments) > 1:
    print("Multiple assignments found; refusing to guess. Inspect manually.")
    sys.exit(1)

a = assignments[0]
rpi = a.runtime_proxy_info

session = SessionState(
    name=target_name,
    token=rpi.token,
    url=rpi.url,
    endpoint=a.endpoint,
    variant=a.variant.name,
    accelerator=a.accelerator.name,
)

state.store.add(session)
print(f"Adopted browser runtime as local session '{target_name}' (endpoint={a.endpoint}).")

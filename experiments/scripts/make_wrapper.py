"""Write a colab-exec wrapper for one attempt of a trial, resuming from a given epoch.

Usage:
    make_wrapper.py <wrapper_path> <trial_name> <target_epochs> <completed_epochs> <base_arg>...

Strips any --epochs/--start-epoch/--resume-from-revision from <base_arg>... and replaces
them with the correct values for this attempt: --epochs (target - completed), plus
--start-epoch/--resume-from-revision if completed > 0.
"""

import json
import sys

wrapper_path, trial_name, target_epochs_s, completed_s, *base_args = sys.argv[1:]
target_epochs = int(target_epochs_s)
completed = int(completed_s)
remaining = target_epochs - completed

filtered = []
skip_next = False
for arg in base_args:
    if skip_next:
        skip_next = False
        continue
    if arg in ("--epochs", "--start-epoch", "--resume-from-revision"):
        skip_next = True
        continue
    filtered.append(arg)

filtered += ["--epochs", str(remaining)]
if completed > 0:
    filtered += ["--start-epoch", str(completed), "--resume-from-revision", trial_name]

argv = ["training_run.py", *filtered,
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

print(f"[make_wrapper] {wrapper_path}: attempt targets {remaining} more epoch(s) "
      f"(completed={completed}, target={target_epochs})")

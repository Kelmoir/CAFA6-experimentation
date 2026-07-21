"""Throwaway infra smoke test for the Colab regularization-sweep pipeline.

Run via a persistent session (exec does not forward argv, unlike `colab run`):
    colab new -s cafa6-smoke --gpu T4
    colab exec -s cafa6-smoke -f experiments/smoke_test.py --timeout 120
    colab stop -s cafa6-smoke

Checks, in order: GPU availability, and whether a Colab Secret named HF_TOKEN
is reachable from a CLI-provisioned session. Never prints the token value
itself, only whether it resolved.

trackio-Space tracking is currently dropped from the sweep plan (HF free-Space
policy changes made it a non-option) - no trackio check here; per-trial logs
and Hub checkpoint branches are the source of truth instead.
"""

import subprocess
import sys
import traceback

if "google.colab" in sys.modules:
    for package in ("huggingface_hub",):
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_gpu():
    try:
        import torch

        available = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if available else None
        print(f"[SMOKE] cuda_available={available} device={name}")
    except Exception:
        print("[SMOKE] cuda_available=ERROR")
        traceback.print_exc()


def check_hf_secret():
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        print(f"[SMOKE] colab_secret_hf_token_found={bool(token)}")
    except Exception as exc:
        print(f"[SMOKE] colab_secret_hf_token_found=False reason={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    check_gpu()
    check_hf_secret()

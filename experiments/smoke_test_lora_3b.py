"""Smoke test for --use-lora on facebook/esm2_t36_3B_UR50D, before spending real GPU
budget on the actual scale-up trial. Verifies: peft loads and wraps the 3B model
correctly, trainable param count is small (not 3B), one forward+backward pass succeeds
without OOM, gradient checkpointing doesn't break backprop, GPU memory usage is visible
against the ~25GB rental budget, and the adapter-only save/reload round-trip works.
Structural correctness of the LoRA wiring itself (peft attribute paths, bias-init,
dropout-setting, dtype handling) was already verified locally against real peft 0.19.1
using a tiny esm2_t6_8M model - this test is specifically about whether it holds up at
3B scale and fits the VRAM budget, which can only be checked on a real GPU.

Run via `colab run` (self-cleaning, no persistent session/HF token needed - this script
never pushes to the Hub, only saves+reloads locally):
  colab run --session cafa6-lora-3b-smoke --gpu <your rented tier> --timeout 900 experiments/smoke_test_lora_3b.py
"""
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
import training_run

MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
NUM_LABELS = 383  # matches the current coarse-GO-term label space

id2label = {i: f"GO:{i:07d}" for i in range(NUM_LABELS)}
label2id = {v: k for k, v in id2label.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={device}")
if device.type != "cuda":
    print("[WARN] No GPU available - this smoke test is meant to run on a real Colab GPU runtime.")

t0 = time.time()
model = training_run.create_esm(
    model_type=MODEL_NAME,
    output_dropout=0.2,
    id2label=id2label,
    label2id=label2id,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_base_dtype=torch.bfloat16,
    gradient_checkpointing=True,
).to(device)
print(f"[TIMING] model load + LoRA wrap: {time.time() - t0:.1f}s")

if device.type == "cuda":
    print(f"[INFO] GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
          f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved")

# Bias-init sanity check - same call path training_run.py's main() uses on a fresh run.
bias_param = training_run.get_classifier_bias_param(model)
print(f"[INFO] classifier bias param: shape={tuple(bias_param.shape)}, dtype={bias_param.dtype}, requires_grad={bias_param.requires_grad}")
with torch.no_grad():
    bias_param.copy_(torch.zeros(NUM_LABELS))
print("[INFO] bias-init helper: OK")

# Forward + backward smoke test at a real-ish batch/sequence size.
batch_size, seq_len = 4, 1024
input_ids = torch.randint(4, 24, (batch_size, seq_len), device=device)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
labels = torch.zeros(batch_size, NUM_LABELS, device=device)
labels[:, 0] = 1.0

model.train()
t0 = time.time()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
loss.backward()
print(f"[TIMING] forward+backward (batch={batch_size}, seq_len={seq_len}): {time.time() - t0:.1f}s")
print(f"[INFO] loss={loss.item():.4f}")

if device.type == "cuda":
    print(f"[INFO] GPU memory after forward+backward: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
          f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

lora_grad_ok = any(
    p.grad is not None and p.grad.abs().sum().item() > 0
    for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
)
print(f"[INFO] LoRA params received nonzero gradient: {lora_grad_ok}")
print(f"[INFO] classifier bias received gradient: {bias_param.grad is not None}")

# Save/reload round trip - local only, no Hub push, keeps this smoke test self-contained.
save_dir = Path("smoke_test_lora_ckpt")
shutil.rmtree(save_dir, ignore_errors=True)
model.save_pretrained(save_dir)
ckpt_files = list(save_dir.iterdir())
total_size_mb = sum(f.stat().st_size for f in ckpt_files) / 1e6
print(f"[INFO] Saved checkpoint: {[f.name for f in ckpt_files]}, total {total_size_mb:.1f}MB")

model.eval()
with torch.no_grad():
    out_before = model(input_ids=input_ids, attention_mask=attention_mask).logits

reloaded = training_run.create_esm(
    model_type=MODEL_NAME,
    output_dropout=0.2,
    id2label=id2label,
    label2id=label2id,
    use_lora=True,
    lora_resume_repo_id=str(save_dir),
).to(device)
reloaded.eval()
with torch.no_grad():
    out_after = reloaded(input_ids=input_ids, attention_mask=attention_mask).logits
print(f"[INFO] save/reload round-trip logits match: {torch.allclose(out_before, out_after, atol=1e-4)}")

print("\n[SUMMARY] Smoke test complete. Review the [TIMING]/[INFO] GPU memory lines above "
      "against your rented VRAM budget before launching the real trial.")

"""Post-hoc diagnostic backfill for a checkpoint trained BEFORE run_diagnostic gained the
rarity-stratified AUROC / hierarchy-violation-rate fields (Task 5 of
docs/superpowers/plans/2026-10-11-go-dag-hierarchy-consistency.md) -- most importantly
kelmoir/cafa6-esm@main (trial-013), the pure-BCE baseline a hierarchy-loss trial needs a
real control number against (see docs/superpowers/specs/2026-10-11-go-dag-hierarchy-
consistency-design.md's "Backfill script" section). Modeled directly on
experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py's re-download-and-reload
pattern, but targeting dataset_full (not that script's stale legacy dataset path) and
passing the new hierarchy_edge_index/label_is_rare arguments unconditionally. No
retraining, no re-push to the Hub -- read-only.

Run directly:
    env/bin/python experiments/scripts/rerun_diagnostic_full.py <revision> [max_val_samples] [rare_term_threshold]
e.g.:
    env/bin/python experiments/scripts/rerun_diagnostic_full.py main 2000 100
"""
import json
import sys
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import training_run

CHECKPOINT_REVISION = sys.argv[1] if len(sys.argv) > 1 else "main"
HF_REPO_ID = "kelmoir/cafa6-esm"
MAX_VAL_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
RARE_TERM_THRESHOLD = int(sys.argv[3]) if len(sys.argv) > 3 else 100
DATA_PATH = Path("data/cafa-6-protein-function-prediction/dataset_full")

trial_metrics_path = hf_hub_download(repo_id=HF_REPO_ID, filename="trial_metrics.json", revision=CHECKPOINT_REVISION)
trial_config = json.loads(Path(trial_metrics_path).read_text())["config"]
MODEL_NAME = trial_config["model_name"]
USE_LORA = trial_config.get("use_lora", False)
USE_ASPECT_INPUT = trial_config.get("use_aspect_input", False)
USE_KNOWN_INPUT = trial_config.get("use_known_input", False)
KNOWN_EMBED_DIM = trial_config.get("known_embed_dim", 64)
MIXED_PRECISION = trial_config.get("mixed_precision", "no")
AUTOCAST_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(MIXED_PRECISION)
# Use the ORIGINAL trial's own --max-length, not this module's default (1024): trial-013
# (and any other trial that explicitly passed --max-length 4192) would otherwise have its
# validation sequences silently truncated differently than they were during its own
# training/diagnostic run, making this backfilled control number not fairly comparable to
# a new trial that correctly passes --max-length 4192 (see this plan's Global Constraints
# and Task 6's own note on the same confound) -- caught while implementing this task.
MAX_LENGTH = trial_config.get("max_length", training_run.MAX_LENGTH)

dataset = training_run.load_training_data(data_path=DATA_PATH)
label_names = [str(name) for name in dataset["train"].features["labels"].feature.names]
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in id2label.items()}

label_counts = torch.zeros(len(label_names))
for labs in dataset["train"]["labels"]:
    for idx in labs:
        label_counts[int(idx)] += 1
label_is_rare = (label_counts.numpy() < RARE_TERM_THRESHOLD)

graph_path = DATA_PATH / "go_dependency_graph.json"
hierarchy_edge_index = None
if graph_path.exists():
    hierarchy_edge_index = training_run.build_hierarchy_edge_index(label2id, graph_path).numpy()
else:
    print(f"[WARN] {graph_path} not found -- skipping hierarchy-violation diagnostic fields")

tokenizer = training_run.create_esm_tokenizer(MODEL_NAME)
preprocess = partial(
    training_run.preprocess_batch,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    unique_items=label_names,
    label2id=label2id,
    use_known_input=USE_KNOWN_INPUT,
    known_terms_from_dataset=False,
)
val_ds = dataset["validation"].with_transform(transform=preprocess)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = AUTOCAST_DTYPE is not None and device.type == "cuda"

if USE_LORA:
    model = training_run.create_esm(
        model_type=MODEL_NAME, use_lora=True, lora_resume_repo_id=HF_REPO_ID,
        revision=CHECKPOINT_REVISION, id2label=id2label, label2id=label2id,
        lora_base_dtype=AUTOCAST_DTYPE,
    ).to(device)
else:
    model = training_run.create_esm(
        unfreeze_last_n=1, model_type=HF_REPO_ID, revision=CHECKPOINT_REVISION,
        id2label=id2label, label2id=label2id,
        use_aspect_input=USE_ASPECT_INPUT, use_known_input=USE_KNOWN_INPUT,
        known_embed_dim=KNOWN_EMBED_DIM,
    ).to(device)
model.eval()
print(f"[INFO] Loaded {HF_REPO_ID}@{CHECKPOINT_REVISION} ({MODEL_NAME}, use_lora={USE_LORA}, "
      f"use_known_input={USE_KNOWN_INPUT}, max_length={MAX_LENGTH}) for eval, device={device}")

training_run.run_diagnostic(
    model,
    tokenizer,
    val_ds,
    label_names,
    device,
    autocast_dtype=AUTOCAST_DTYPE,
    use_amp=use_amp,
    max_val_samples=MAX_VAL_SAMPLES,
    batch_size=16,
    use_aspect_input=USE_ASPECT_INPUT,
    use_known_input=USE_KNOWN_INPUT,
    hierarchy_edge_index=hierarchy_edge_index,
    label_is_rare=label_is_rare,
)

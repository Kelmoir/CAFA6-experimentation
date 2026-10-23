import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

# Known env gotcha (see CLAUDE.md): huggingface_hub/hf_xet version mismatches on this host
# can raise "ImportError: cannot import name 'XetSession' from 'hf_xet'" on any push/upload.
# huggingface_hub.constants reads this env var at IMPORT time (not lazily at push time), so
# it must be set before `import huggingface_hub` runs, not just before the push call itself.
# setdefault (not a direct assignment) leaves an explicit user override alone.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import networkx as nx
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, get_token
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, EsmTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import EsmForSequenceClassification
from transformers.trainer_pt_utils import LengthGroupedSampler

if "google.colab" in sys.modules:
    required_packages = ["torch", "transformers", "datasets", "huggingface_hub", "peft", "scikit-learn"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import trackio
except ImportError:  # pragma: no cover - optional dependency
    trackio = None

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency, only required for --use-lora
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None

MAX_LENGTH = 1024
ESM_MODEL_NAMES = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
    "facebook/esm2_t48_15B_UR50D",
]
SELECTED_MODEL = "facebook/esm2_t12_35M_UR50D"
HF_REPO_ID = os.getenv("HF_REPO_ID", "kelmoir/cafa6-esm")


def get_lr_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    schedule_type: str = "constant",
):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return (step + 1) / max(1, num_warmup_steps)
        if schedule_type == "constant":
            return 1.0
        progress = min(1.0, (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps))
        if schedule_type == "linear_decay":
            return max(0.0, 1.0 - progress)
        if schedule_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_esm_tokenizer(model_name: str = "facebook/esm2_t30_150M_UR50D") -> EsmTokenizer:
    return EsmTokenizer.from_pretrained(model_name)


def set_classifier_dropout(model, dropout_p: float) -> None:
    """Sets dropout on the classification head. Under plain EsmForSequenceClassification
    the module lives at `classifier.dropout`. Under peft's `modules_to_save=["classifier"]`
    wrapping (used by --use-lora), peft replaces `classifier` with a ModulesToSaveWrapper
    holding TWO copies - a frozen `classifier.original_module.dropout` and the active,
    trainable `classifier.modules_to_save.default.dropout` - verified directly against
    peft 0.19.1 rather than assumed, since the exact wrapping structure isn't
    guessable/stable across peft versions. Matching on "classifier" appearing anywhere
    in the name plus a "dropout" suffix catches both copies in the wrapped case (setting
    the frozen copy too is harmless - it's never invoked in the forward pass while the
    adapter is active) and the single copy in the unwrapped case, without depending on
    the exact intermediate path.
    """
    matched = 0
    for name, module in model.named_modules():
        if "classifier" in name and name.endswith("dropout") and isinstance(module, torch.nn.Dropout):
            module.p = dropout_p
            matched += 1
    if matched == 0:
        raise RuntimeError("Could not find classifier dropout module(s) on model - unexpected model structure")


def get_classifier_bias_param(model) -> torch.nn.Parameter:
    """Locates the trainable classifier output-projection bias. Under plain
    EsmForSequenceClassification this is `classifier.out_proj.bias`. Under peft's
    modules_to_save wrapping (--use-lora) it's `classifier.modules_to_save.default.out_proj.bias`,
    with a frozen `classifier.original_module.out_proj.bias` sibling (requires_grad=False) -
    verified directly against peft 0.19.1. Matching on the "out_proj.bias" suffix
    (unambiguous: nothing else in ESM's architecture is named out_proj) plus
    requires_grad correctly isolates the one live copy in both cases without depending
    on the exact intermediate path peft inserts.
    """
    candidates = [
        (name, param)
        for name, param in model.named_parameters()
        if name.endswith("out_proj.bias") and param.requires_grad
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one trainable out_proj.bias parameter, found "
            f"{len(candidates)}: {[n for n, _ in candidates]}"
        )
    return candidates[0][1]


class AspectAwareClassificationHead(torch.nn.Module):
    """Classifier head for the aspect-conditioning ablation (trial-078 vs trial-077,
    see experiments/ancestor_sweep/sweep_log.md): identical to transformers' EsmClassificationHead
    (CLS-token pooling -> dropout -> dense -> tanh -> dropout -> out_proj) except the
    CLS-token representation is concatenated with a ground-truth aspect multi-hot
    vector ([C, F, P], see generate_dataset_depth1.py's aspect_labels column) before
    the dense layer. The ESM backbone's own forward pass is untouched by this - only
    the classifier head differs from the unconditioned baseline - so any quality delta
    between the two experiments is attributable to the aspect signal alone, not to a
    changed encoder.
    """

    def __init__(self, config, num_aspects: int = 3):
        super().__init__()
        self.num_aspects = num_aspects
        self.dense = torch.nn.Linear(config.hidden_size + num_aspects, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, aspect_input: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if aspect_input is None:
            raise ValueError("AspectAwareClassificationHead requires aspect_input (a batch x num_aspects tensor)")
        x = features[:, 0, :]  # <s>/CLS token, matching EsmClassificationHead
        x = self.dropout(x)
        x = torch.cat([x, aspect_input.to(dtype=x.dtype, device=x.device)], dim=-1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AspectConditionedEsmForSequenceClassification(EsmForSequenceClassification):
    """EsmForSequenceClassification with its classifier head swapped for
    AspectAwareClassificationHead. Loss is intentionally not computed here (unlike the
    base class's forward) - training_run.py always computes it externally via its own
    criterion for every model variant, so this override only needs to produce logits.
    """

    def __init__(self, config, num_aspects: int = 3):
        super().__init__(config)
        self.classifier = AspectAwareClassificationHead(config, num_aspects=num_aspects)
        # The base __init__'s post_init() already ran (and random-initialized the
        # original classifier it just got replaced with) before this line, so this
        # head's freshly-created Linear layers need their own init - mirrors
        # PreTrainedModel._init_weights' standard scheme (normal_(std=initializer_range),
        # zeroed bias) rather than leaving them at PyTorch's default Linear init.
        std = getattr(config, "initializer_range", None) or 0.02
        with torch.no_grad():
            for module in self.classifier.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, input_ids=None, attention_mask=None, aspect_input=None, **kwargs) -> SequenceClassifierOutput:
        outputs = self.esm(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, aspect_input=aspect_input)
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class KnownAwareClassificationHead(torch.nn.Module):
    """Like AspectAwareClassificationHead (above), but the auxiliary input is a
    variable-size set of known GO term ids rather than a fixed 3-dim aspect vector:
    pooled via a learned per-term embedding + sum (nn.EmbeddingBag), not concatenated
    directly. A raw one-hot concat of the full 27,191-dim label space was rejected --
    see docs/superpowers/specs/2026-10-08-known-annotation-conditioning-design.md's
    "Alternatives considered": it would add ~27191*hidden_size parameters to one dense
    layer and tends to memorize specific known-term combinations rather than
    generalize across related GO terms. The embedding table's vocabulary size is
    config.num_labels -- every term is both a known-input candidate and a prediction
    target, the same vocabulary either way.
    """

    def __init__(self, config, known_embed_dim: int = 64):
        super().__init__()
        self.known_embed_dim = known_embed_dim
        self.known_embedding = torch.nn.EmbeddingBag(config.num_labels, known_embed_dim, mode="sum")
        self.dense = torch.nn.Linear(config.hidden_size + known_embed_dim, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor, known_indices: Optional[torch.Tensor] = None,
                known_offsets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if known_indices is None or known_offsets is None:
            raise ValueError("KnownAwareClassificationHead requires known_indices/known_offsets "
                              "(flattened EmbeddingBag inputs -- see data_collate_function)")
        x = features[:, 0, :]  # <s>/CLS token, matching EsmClassificationHead
        x = self.dropout(x)
        known_vec = self.known_embedding(known_indices, known_offsets)
        x = torch.cat([x, known_vec.to(dtype=x.dtype, device=x.device)], dim=-1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class KnownConditionedEsmForSequenceClassification(EsmForSequenceClassification):
    """EsmForSequenceClassification with its classifier head swapped for
    KnownAwareClassificationHead. Mirrors AspectConditionedEsmForSequenceClassification
    exactly (above) -- loss is computed externally by training_run.py, not here.
    """

    def __init__(self, config, known_embed_dim: int = 64):
        super().__init__(config)
        self.classifier = KnownAwareClassificationHead(config, known_embed_dim=known_embed_dim)
        std = getattr(config, "initializer_range", None) or 0.02
        with torch.no_grad():
            for module in self.classifier.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, input_ids=None, attention_mask=None, known_indices=None,
                known_offsets=None, **kwargs) -> SequenceClassifierOutput:
        outputs = self.esm(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, known_indices=known_indices, known_offsets=known_offsets)
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LabelEmbeddingClassificationHead(torch.nn.Module):
    """Item 8's compatibility-score head (see experiments/cafa6_next_steps_plan.md): replaces
    the flat per-term classifier weight ([hidden_size, num_labels]) with a small learned
    projection into a precomputed, frozen GO-term-embedding space ([hidden_size, embed_dim]),
    scored by cosine similarity against every term's embedding. A new term just needs its own
    embedding row (see experiments/label_embedding/build_term_embeddings.py), not a newly
    trained output weight -- the premise item 9 needs to be worth building.

    term_embeddings must already be L2-normalized and in exactly the label2id row order (the
    feasibility spike/salvage follow-up found a generic sentence encoder alone isn't
    distinctive enough for antonym term pairs -- build_term_embeddings.py's definition
    embeddings are augmented with a direction-antonym flag to fix this; this head doesn't care
    how the embeddings were produced, only that they're already fixed and normalized).
    """

    def __init__(self, config, term_embeddings: torch.Tensor):
        super().__init__()
        if term_embeddings.shape[0] != config.num_labels:
            raise ValueError(
                f"term_embeddings has {term_embeddings.shape[0]} rows but config.num_labels="
                f"{config.num_labels} -- regenerate experiments/label_embedding/"
                "build_term_embeddings.py against the dataset actually in use, the cached "
                "term order must exactly match label2id"
            )
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.proj = torch.nn.Linear(config.hidden_size, term_embeddings.shape[1])
        self.register_buffer("term_embeddings", term_embeddings, persistent=False)
        self.scale = torch.nn.Parameter(torch.tensor(10.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = features[:, 0, :]  # <s>/CLS token, matching EsmClassificationHead
        x = self.dropout(x)
        x = self.proj(x)
        x = torch.nn.functional.normalize(x, dim=-1)
        sims = x @ self.term_embeddings.to(dtype=x.dtype).T
        return self.scale * sims + self.bias


class LabelEmbeddingEsmForSequenceClassification(EsmForSequenceClassification):
    """EsmForSequenceClassification with its classifier head swapped for
    LabelEmbeddingClassificationHead. Mirrors AspectConditionedEsmForSequenceClassification
    exactly (above) -- loss is computed externally by training_run.py, not here.
    """

    def __init__(self, config, term_embeddings: Optional[torch.Tensor] = None):
        super().__init__(config)
        if term_embeddings is None:
            raise ValueError("LabelEmbeddingEsmForSequenceClassification requires term_embeddings")
        self.classifier = LabelEmbeddingClassificationHead(config, term_embeddings)
        std = getattr(config, "initializer_range", None) or 0.02
        with torch.no_grad():
            self.classifier.proj.weight.normal_(mean=0.0, std=std)
            self.classifier.proj.bias.zero_()

    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> SequenceClassifierOutput:
        outputs = self.esm(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_esm(
    unfreeze_last_n: int = 2,
    model_type: str = "facebook/esm2_t30_150M_UR50D",
    output_dropout: float = 0.0,
    attention_dropout: float = 0.0,
    esm_dropout: float = 0.0,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    revision: Optional[str] = None,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    lora_resume_repo_id: Optional[str] = None,
    lora_base_dtype: Optional[torch.dtype] = None,
    lora_layers_to_transform: Optional[int] = None,
    gradient_checkpointing: bool = False,
    use_aspect_input: bool = False,
    use_known_input: bool = False,
    known_embed_dim: int = 64,
    use_label_embedding_head: bool = False,
    term_embeddings: Optional[torch.Tensor] = None,
):
    if id2label is None or label2id is None:
        raise ValueError("id2label and label2id must be provided")

    if use_aspect_input and use_lora:
        raise NotImplementedError(
            "--use-aspect-input + --use-lora is not implemented - AspectConditionedEsmForSequenceClassification "
            "only replaces the plain (non-LoRA) classifier head."
        )

    if use_known_input and use_lora:
        raise NotImplementedError(
            "--use-known-input + --use-lora is not implemented - KnownConditionedEsmForSequenceClassification "
            "only replaces the plain (non-LoRA) classifier head."
        )
    if use_known_input and use_aspect_input:
        raise NotImplementedError(
            "--use-known-input + --use-aspect-input is not implemented - no classifier head "
            "concatenates both auxiliary inputs today (see the design spec's non-goals)."
        )
    if use_label_embedding_head and use_lora:
        raise NotImplementedError(
            "--use-label-embedding-head + --use-lora is not implemented - "
            "LabelEmbeddingEsmForSequenceClassification only replaces the plain (non-LoRA) classifier head."
        )
    if use_label_embedding_head and (use_known_input or use_aspect_input):
        raise NotImplementedError(
            "--use-label-embedding-head is mutually exclusive with --use-known-input/--use-aspect-input - "
            "only one classifier-head swap mechanism is implemented at a time."
        )
    if use_label_embedding_head and term_embeddings is None:
        raise ValueError("--use-label-embedding-head requires term_embeddings (see --label-embeddings-path)")

    if use_lora:
        if get_peft_model is None:
            raise RuntimeError("--use-lora requires the 'peft' package; install it or drop --use-lora")

        # The LoRA base architecture always loads from the original pretrained
        # model_type, never from --hf-repo-id: a LoRA checkpoint on the Hub is
        # adapter-only (a few MB of low-rank deltas), not a full standalone model, so
        # it can't itself serve as the base the way a full-fine-tune checkpoint can
        # (see the non-LoRA branch below, and the model_type selection in main()).
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_type,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            problem_type="multi_label_classification",
            torch_dtype=lora_base_dtype,
        )
        for param in base_model.esm.parameters():
            param.requires_grad = False
        if esm_dropout or attention_dropout:
            # By default every layer gets a LoRA adapter (no "last N unfrozen"
            # boundary the way full fine-tuning has), so these apply uniformly across
            # all layers rather than only the last unfreeze_last_n - unless
            # lora_layers_to_transform restricts which layers get adapters below, in
            # which case dropout on the untouched early layers has no effect anyway
            # (their backward pass is skipped entirely, see below).
            for layer in base_model.esm.encoder.layer:
                layer.output.dropout.p = esm_dropout
                layer.attention.output.dropout.p = attention_dropout

        layers_to_transform = None
        if lora_layers_to_transform is not None:
            # Restricting which layers get LoRA adapters isn't just about adapter
            # count - it changes what autograd actually has to compute. Verified
            # directly (peft 0.19.1): with adapters only on the last N layers, the
            # earlier layers' output activations have requires_grad=False (since
            # neither their frozen weights nor their input - unless
            # enable_input_require_grads() forces it, see gradient_checkpointing
            # below - require grad), so PyTorch's autograd never builds or traverses
            # a backward graph for them at all. Forward compute still runs for every
            # layer (the final classification depends on the full forward pass), but
            # the backward pass - typically the more expensive half - only runs for
            # the last N layers. No manual graph-splitting/no_grad wrapping needed;
            # this is standard autograd behavior once trainable params are absent
            # from a prefix of the network.
            num_layers = base_model.esm.config.num_hidden_layers
            n = min(lora_layers_to_transform, num_layers)
            layers_to_transform = list(range(num_layers - n, num_layers))
            print(f"[INFO] --lora-layers-to-transform={lora_layers_to_transform}: LoRA adapters "
                  f"restricted to layers {layers_to_transform} - earlier layers skip backward entirely.")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or ["query", "value"],
            modules_to_save=["classifier"],
            layers_to_transform=layers_to_transform,
            task_type=TaskType.SEQ_CLS,
        )
        if lora_resume_repo_id:
            model = PeftModel.from_pretrained(base_model, lora_resume_repo_id, revision=revision, is_trainable=True)
            print(f"[INFO] Resumed LoRA adapter weights from {lora_resume_repo_id}@{revision}")
        else:
            model = get_peft_model(base_model, lora_config)

        if lora_base_dtype is not None:
            # peft explicitly upcasts LoRA A/B matrices to fp32 regardless of the frozen
            # base's dtype (verified directly against peft 0.19.1), but modules_to_save's
            # classifier copy has no such special-casing - it's a plain deep-copy of the
            # original module, so it silently inherited lora_base_dtype (bf16/fp16) from
            # the base load above. Training a classifier head purely in reduced precision
            # risks AdamW's small per-step updates getting rounded away with no fp32
            # master copy; the VRAM cost of keeping it fp32 is negligible (a few hundred
            # KB against a multi-GB frozen backbone), so there's no real tradeoff here.
            for name, param in model.named_parameters():
                if "classifier" in name and param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.float()

        model.print_trainable_parameters()
        print("[INFO] --use-lora set: --unfreeze-last-n is ignored (LoRA adapters apply "
              "per --lora-target-modules across all backbone layers).")

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            # Required with LoRA: the input embeddings are frozen, so without this the
            # activation entering the first checkpointed layer has requires_grad=False
            # and the backward graph never reaches the trainable LoRA/classifier
            # params - a well-documented peft + gradient-checkpointing interaction.
            model.enable_input_require_grads()

        set_classifier_dropout(model, output_dropout)
        return model

    model_cls = (
        KnownConditionedEsmForSequenceClassification if use_known_input
        else AspectConditionedEsmForSequenceClassification if use_aspect_input
        else LabelEmbeddingEsmForSequenceClassification if use_label_embedding_head
        else AutoModelForSequenceClassification
    )
    extra_model_kwargs = (
        {"known_embed_dim": known_embed_dim} if use_known_input
        else {"term_embeddings": term_embeddings} if use_label_embedding_head
        else {}
    )
    model = model_cls.from_pretrained(
        model_type,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        revision=revision,
        **extra_model_kwargs,
    )

    if use_label_embedding_head:
        # from_pretrained's fast meta-device init path only properly initializes module
        # types _init_weights recognizes (nn.Linear, etc, matched by isinstance) - verified
        # directly: classifier.proj's Linear weight/bias come out correctly initialized
        # (std=0.02/zeros), but classifier.scale/classifier.bias are bare nn.Parameter
        # scalars with no such isinstance match, so they're left as whatever uninitialized
        # meta-device garbage they happened to contain (~1e-29), silently overwriting the
        # 10.0/0.0 LabelEmbeddingClassificationHead.__init__ already set. Only re-applied on
        # a fresh start (revision is None) - a --resume-from-revision load has real trained
        # scale/bias values in the checkpoint's state dict, not missing/garbage ones.
        #
        # term_embeddings is hit even harder regardless of revision - verified directly it
        # comes back as all zeros, since it's constructed (under __init__) while
        # from_pretrained's meta-device context is active, which silently redirects any
        # tensor creation to a placeholder regardless of the real data passed in. It's also
        # registered non-persistent (never saved to/loaded from a checkpoint's state dict,
        # see LabelEmbeddingClassificationHead), so it always needs recopying from the
        # term_embeddings argument, resume or not.
        if revision is None:
            with torch.no_grad():
                model.classifier.scale.fill_(10.0)
                model.classifier.bias.fill_(0.0)
        with torch.no_grad():
            model.classifier.term_embeddings.copy_(term_embeddings)

    for param in model.esm.parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        num_layers = model.esm.config.num_hidden_layers
        unfreeze_last_n = min(unfreeze_last_n, num_layers)
        print(f"Model layers: {num_layers}, request unfreeze layers: {unfreeze_last_n}")
        for i in range(num_layers - unfreeze_last_n, num_layers):
            for param in model.esm.encoder.layer[i].parameters():
                param.requires_grad = True
            model.esm.encoder.layer[i].output.dropout.p = esm_dropout
            model.esm.encoder.layer[i].attention.output.dropout.p = attention_dropout

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    set_classifier_dropout(model, output_dropout)
    return model


def preprocess_batch(
    examples,
    tokenizer: EsmTokenizer,
    label2id: Dict[str, int],
    unique_items: List[str],
    max_length: int = MAX_LENGTH,
    mask_sparse_negatives_max_total_terms: Optional[int] = None,
    loss_mask_from_dataset: bool = False,
    use_known_input: bool = False,
    known_mix_zero_frac: float = 0.3,
    known_reveal_max: float = 1.0,
    known_terms_from_dataset: bool = False,
    rng: Optional[random.Random] = None,
    known_split_seed: Optional[int] = None,
):
    seq_key = "sequence" if "sequence" in examples else "input_ids"
    if seq_key not in examples:
        return examples

    sequences = examples[seq_key]
    if not isinstance(sequences, list):
        sequences = [sequences]

    tokenized_output = tokenizer(sequences, max_length=max_length, truncation=True)

    input_ids = []
    attention_masks = []
    for item in tokenized_output["input_ids"]:
        if hasattr(item, "tolist"):
            input_ids.append(torch.tensor(item.tolist(), dtype=torch.long))
        else:
            input_ids.append(torch.tensor(item, dtype=torch.long))

    for item in tokenized_output["attention_mask"]:
        if hasattr(item, "tolist"):
            attention_masks.append(torch.tensor(item.tolist(), dtype=torch.long))
        else:
            attention_masks.append(torch.tensor(item, dtype=torch.long))

    output_dict: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }

    if "protein" in examples:
        output_dict["protein"] = examples["protein"]

    if "labels" in examples:
        raw_labels = examples["labels"]
        if not isinstance(raw_labels, list):
            raw_labels = [raw_labels]

        labels_list = []
        for labels_value in raw_labels:
            labels = torch.zeros((len(unique_items),), dtype=torch.float)
            if isinstance(labels_value, str):
                for item in labels_value.split():
                    if item in label2id:
                        labels[label2id[item]] = 1.0
            elif isinstance(labels_value, (list, tuple)):
                # datasets.Sequence(ClassLabel(...)) yields already-encoded integer
                # class ids here, not the string names label2id is keyed on.
                for item in labels_value:
                    if isinstance(item, str):
                        if item in label2id:
                            labels[label2id[item]] = 1.0
                    else:
                        idx = int(item)
                        if 0 <= idx < len(unique_items):
                            labels[idx] = 1.0
            labels_list.append(labels)
        output_dict["labels"] = labels_list

        if mask_sparse_negatives_max_total_terms is not None and "total_terms" in examples:
            # GO's Open World Assumption (absence of annotation != absence of function,
            # see experiments/ancestor_sweep/sweep_log.md's trial-080 investigation): a "negative" label
            # on a sparsely-annotated protein more likely reflects incomplete
            # characterization than a confirmed absence, so it gets excluded from the
            # loss (weight 0) rather than trusted as a hard 0. Positives are always
            # trusted (weight 1) regardless of total_terms - an asserted positive
            # annotation doesn't become less true because the rest of the protein is
            # under-studied; only the interpretation of an absence is in question.
            raw_total_terms = examples["total_terms"]
            if not isinstance(raw_total_terms, list):
                raw_total_terms = [raw_total_terms]
            loss_mask_list = []
            for labels, total_terms in zip(labels_list, raw_total_terms):
                sparse = total_terms <= mask_sparse_negatives_max_total_terms
                mask = torch.ones_like(labels) if not sparse else (labels == 1.0).float()
                loss_mask_list.append(mask)
            output_dict["loss_mask"] = loss_mask_list

        if loss_mask_from_dataset and "excluded_labels" in examples:
            # Two-tier discrete relabeling (see experiments/scripts/build_two_tier_dataset.py):
            # the dataset already carries a sparse per-row list of label indices to exclude from
            # the loss (currently-unlabeled cells a TIcE-calibrated prior + the baseline
            # classifier's own ranking flagged as likely-hidden-positives) -- just materialize it
            # into a dense mask here, no derivation needed (unlike the total_terms path above).
            raw_excluded = examples["excluded_labels"]
            if not isinstance(raw_excluded, list):
                raw_excluded = [raw_excluded]
            loss_mask_list = []
            for labels, excluded in zip(labels_list, raw_excluded):
                mask = torch.ones_like(labels)
                for idx in excluded:
                    mask[int(idx)] = 0.0
                loss_mask_list.append(mask)
            output_dict["loss_mask"] = loss_mask_list

    if "aspect_labels" in examples:
        # generate_dataset_depth1.py's 3-way [C, F, P] multi-hot column - an optional
        # model INPUT feature for the aspect-conditioning ablation (trial-078), not a
        # prediction target. Present regardless of --use-aspect-input whenever the
        # dataset carries the column, so both the with- and without-aspect experiments
        # (trial-077/078) share an identical data pipeline up to the flag; the model
        # simply doesn't consume it unless built with AspectConditionedEsmForSequenceClassification.
        raw_aspects = examples["aspect_labels"]
        if not isinstance(raw_aspects, list):
            raw_aspects = [raw_aspects]
        output_dict["aspect_input"] = [torch.tensor(a, dtype=torch.float) for a in raw_aspects]

    if use_known_input:
        if known_terms_from_dataset:
            if "known_labels" not in examples:
                raise ValueError("known_terms_from_dataset=True requires a known_labels column")
            raw_known = examples["known_labels"]
            if not isinstance(raw_known, list):
                raw_known = [raw_known]
            known_ids_list = [[int(i) for i in ids] for ids in raw_known]
        else:
            if "labels" not in examples:
                raise ValueError("use_known_input (synthetic mode) requires a labels column")
            # labels_list is built above whenever "labels" in examples - reused here
            # rather than re-decoding raw_labels, since it's already a dense
            # per-example tensor over unique_items.
            known_ids_list = []
            for seq_idx, labels in enumerate(labels_list):
                # known_split_seed keys a per-example rng off the example's own
                # sequence, not call order/global state -- without this, validation/
                # test's known/novel split (same preprocess_batch, re-evaluated lazily
                # by with_transform on every epoch) would re-randomize every epoch too,
                # making val_loss/best-checkpoint-selection and the diagnostic AUROC
                # noisy across epochs instead of a fixed holdout (code review finding).
                if known_split_seed is not None:
                    active_rng = random.Random(f"{known_split_seed}:{sequences[seq_idx]}")
                else:
                    active_rng = rng if rng is not None else random
                positive_ids = torch.nonzero(labels, as_tuple=True)[0].tolist()
                if not positive_ids or active_rng.random() < known_mix_zero_frac:
                    known_ids_list.append([])
                    continue
                reveal_frac = active_rng.uniform(0.0, known_reveal_max)
                n_reveal = int(reveal_frac * len(positive_ids))
                known_ids_list.append(active_rng.sample(positive_ids, n_reveal))

        output_dict["known_term_ids"] = known_ids_list
        loss_mask_list = []
        for known_ids in known_ids_list:
            mask = torch.ones((len(unique_items),), dtype=torch.float)
            for idx in known_ids:
                mask[idx] = 0.0
            loss_mask_list.append(mask)
        output_dict["loss_mask"] = loss_mask_list

    return output_dict


def data_collate_function(preprocessed_batch: List[Dict[str, Any]], tokenizer: EsmTokenizer) -> Dict[str, Any]:
    collated_data: Dict[str, Any] = {}

    input_ids = [sample["input_ids"] for sample in preprocessed_batch]
    attention_mask = [sample["attention_mask"] for sample in preprocessed_batch]

    collated_data["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=int(tokenizer.pad_token_id)
    )
    collated_data["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    labels = [sample["labels"] for sample in preprocessed_batch if "labels" in sample]
    if labels:
        collated_data["labels"] = torch.stack(labels)

    aspect_input = [sample["aspect_input"] for sample in preprocessed_batch if "aspect_input" in sample]
    if aspect_input:
        collated_data["aspect_input"] = torch.stack(aspect_input)

    loss_mask = [sample["loss_mask"] for sample in preprocessed_batch if "loss_mask" in sample]
    if loss_mask:
        collated_data["loss_mask"] = torch.stack(loss_mask)

    known_term_ids = [sample["known_term_ids"] for sample in preprocessed_batch if "known_term_ids" in sample]
    if known_term_ids:
        flat_indices: List[int] = []
        offsets: List[int] = []
        cursor = 0
        for ids in known_term_ids:
            offsets.append(cursor)
            flat_indices.extend(ids)
            cursor += len(ids)
        collated_data["known_indices"] = torch.tensor(flat_indices, dtype=torch.long)
        collated_data["known_offsets"] = torch.tensor(offsets, dtype=torch.long)

    # preprocess_batch attaches "protein" per-sample but this function never gathered it into the
    # batch -- added for experiments/scripts/dataset_refinement_pilot.py, which needs to know
    # which batch row belongs to which protein to patch in its per-run label overrides. A plain
    # list, not a tensor (protein ids are strings).
    proteins = [sample["protein"] for sample in preprocessed_batch if "protein" in sample]
    if proteins:
        collated_data["protein"] = proteins

    return collated_data


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    """Per-batch correct/tp/fn counts, meant to be summed across an epoch and turned
    into accuracy/recall via finalize_metrics() rather than averaged as per-batch
    ratios. Averaging per-batch recall directly (the previous approach) systematically
    underestimates it once label count is high and positive rate is low relative to
    batch size (e.g. 383 labels at ~3.6% positive with batch_size=8: most (label,
    batch) pairs then have zero true positives, so tp/(tp+fn+eps) is spuriously 0 for
    the overwhelming majority of a batch's labels regardless of true model quality,
    dragging every batch's - and therefore the epoch's - averaged recall toward 0).
    Confirmed directly on trial-050: training reported val_recall=3.55% at epoch 10,
    while a proper pooled-over-the-full-validation-set recall was 45.8% for the same
    checkpoint.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == labels).float().sum().item()
    total = labels.numel()
    tp = ((preds == 1) & (labels == 1)).sum(dim=0)
    fp = ((preds == 1) & (labels == 0)).sum(dim=0)
    fn = ((preds == 0) & (labels == 1)).sum(dim=0)
    return {"correct": correct, "total": total, "tp": tp, "fp": fp, "fn": fn}


def finalize_metrics(accumulated: Dict[str, Any]) -> Dict[str, float]:
    """recall is macro (mean of each label's own tp/(tp+fn) ratio) -- this is the
    quantity --checkpoint-selection recall watches, and the one trial logs across this
    repo call "val_recall". precision/f1 are pooled/micro instead (summed tp/fp/fn
    across every label before taking one ratio, same as run_diagnostic()'s
    'precision_at_0.5'/'f1_at_0.5') -- deliberately, so a per-epoch f1 trend is directly
    comparable to --run-diagnostic's post-training number, and so f1 doesn't inherit
    macro-recall's known failure mode against severe label imbalance (see
    compute_metrics's docstring / trial-050) at the precision side too.
    """
    accuracy = accumulated["correct"] / max(1, accumulated["total"])
    recall = (accumulated["tp"] / (accumulated["tp"] + accumulated["fn"] + 1e-8)).mean().item()
    tp_sum = accumulated["tp"].sum()
    fp_sum = accumulated["fp"].sum()
    fn_sum = accumulated["fn"].sum()
    precision = (tp_sum / (tp_sum + fp_sum + 1e-8)).item()
    recall_pooled = (tp_sum / (tp_sum + fn_sum + 1e-8)).item()
    f1 = 2 * precision * recall_pooled / (precision + recall_pooled + 1e-8)
    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}


def compute_loss(criterion, logits: torch.Tensor, labels: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """criterion must have reduction="none" whenever loss_mask is used (--mask-sparse-negatives-max-total-terms) -
    see main()'s criterion construction. Masked-mean over unmasked elements only, so
    excluded (untrustworthy) negative labels contribute zero gradient rather than being
    pulled toward any value - both to training and to the validation loss used for
    best-checkpoint selection, so the two stay consistent with each other.
    """
    if loss_mask is None:
        return criterion(logits, labels)
    per_element_loss = criterion(logits, labels)
    return (per_element_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)


def build_hierarchy_edge_index(label2id: Dict[str, int], graph_path: Path) -> torch.LongTensor:
    """Restrict generate_dataset_full.py's exported GO graph (go_dependency_graph.json)
    to direct is_a/part_of edges where both endpoints are in the current trainable
    vocabulary (label2id) -- the transitive ancestor closure is NOT used here (see the
    design spec's "Why direct edges, not full ancestor closure"): if every direct edge
    satisfies parent_score >= child_score, transitive consistency over any longer chain
    follows automatically, so penalizing the full closure would roughly square the edge
    count for no additional constraint power. Returns a (E, 2) LongTensor of
    (child_label_idx, parent_label_idx) pairs, or a (0, 2) tensor if nothing matches --
    cheap enough (well under a second, ~47k edges at the current 27,191-term vocabulary)
    to build once per run and keep resident on device for the whole training loop.
    """
    payload = json.loads(graph_path.read_text())
    graph = nx.readwrite.json_graph.node_link_graph(payload["graph"], edges="links")
    true_path_relationships = set(payload["true_path_relationships"])
    edges = [
        (label2id[u], label2id[v])
        for u, v, key in graph.edges(keys=True)
        if key in true_path_relationships and u in label2id and v in label2id
    ]
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


def hierarchy_consistency_penalty(logits: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean hinge violation of GO's true-path rule over edge_index's direct
    (child_idx, parent_idx) pairs: relu(child_logit - parent_logit), averaged over every
    edge and every example in the batch. Zero when every edge already satisfies
    parent_logit >= child_logit for every example; monotonic with sigmoid, so this
    constrains predicted probabilities (parent_prob >= child_prob) without needing to
    compute probabilities at all.
    """
    if edge_index.numel() == 0:
        return torch.zeros((), device=logits.device)
    # Upcast to fp32 before subtracting: two fp16 logits near the dtype's extremes
    # (e.g. +-60000) can subtract to a value past fp16's max (65504), producing inf
    # that would otherwise feed straight into GradScaler under --mixed-precision fp16.
    logits32 = logits.float()
    child_logits = logits32[:, edge_index[:, 0]]
    parent_logits = logits32[:, edge_index[:, 1]]
    return torch.relu(child_logits - parent_logits).mean()


def load_ia_weights(ia_file: Path, term_names: List[str]) -> np.ndarray:
    """Relocated from evaluate_cafa_score.py (Tier 2 item 4): that module imports
    training_run at module level, so training_run importing FROM it would be circular --
    this is now the one home for both callers, and evaluate_cafa_score.py imports it
    back from here.
    """
    weights = {}
    for line in ia_file.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2:
            weights[parts[0]] = float(parts[1])
    missing = [name for name in term_names if name not in weights]
    if missing:
        print(f"[WARN] {len(missing)} term(s) missing from {ia_file}, defaulting IA=0.0 "
              f"(contributes nothing to weighted precision/recall): {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")
    return np.array([weights.get(name, 0.0) for name in term_names], dtype=np.float64)


def soft_f1_loss(logits: torch.Tensor, labels: torch.Tensor, ia_weights: torch.Tensor,
                  eps: float = 1e-7) -> torch.Tensor:
    """Differentiable, per-protein (row-wise) surrogate for the official CAFA weighted-Fmax
    metric (see evaluate_cafa_score.py's cafa_weighted_fmax): IA-weighted soft TP/FP/FN
    summed per example, not pooled across the batch -- works at any batch size, unlike a
    term-level variant would at this project's --batch-size 2 (a 2-example micro-batch has
    almost no positives for any given one of 27,191 terms, making per-micro-batch term-level
    precision/recall mostly noise). Two independent CAFA-5 top-6 teams (Evans, Freddolino
    Lab) report this class of loss beating plain BCE at this label cardinality/imbalance.

    Rows with no IA-weighted true annotation among the scored labels are excluded from the
    mean (true_ia_sum == 0 -> soft-F1 is undefined, same convention cafa_weighted_fmax uses
    for its eval_mask), rather than contributing a meaningless eps-dominated loss of ~1.
    """
    logits32 = logits.float()
    probs = torch.sigmoid(logits32)
    weighted_tp = (ia_weights * probs * labels).sum(dim=1)
    weighted_fp = (ia_weights * probs * (1 - labels)).sum(dim=1)
    weighted_fn = (ia_weights * (1 - probs) * labels).sum(dim=1)
    soft_f1 = (2 * weighted_tp) / (2 * weighted_tp + weighted_fp + weighted_fn + eps)
    scored = (ia_weights * labels).sum(dim=1) > 0
    if not scored.any():
        return torch.zeros((), device=logits.device)
    return (1 - soft_f1[scored]).mean()


def nnpu_loss(
    logits_u: torch.Tensor,
    hard_labels_u: torch.Tensor,
    logits_p: torch.Tensor,
    hard_labels_p: torch.Tensor,
    priors: torch.Tensor,
    eligible_mask: torch.Tensor,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Per-label nnPU risk (Kiryo, Niu, du Plessis & Sugiyama 2017, Algorithm 1's non-negative
    correction), generalized here to a shared-backbone multi-label network: each eligible label j
    gets its own class prior priors[j] and its own risk terms, computed from two separate
    forward-passed batches -- logits_u/hard_labels_u (the ordinary training batch, standing in for
    the "unlabeled" sample U) and logits_p/hard_labels_p (a dedicated positive-anchor batch,
    oversampled so every eligible label has real positive rows to compute R_p+/R_p- from even
    though true positives are far too rare -- 0.556% observed rate on the raw label space -- to
    reliably land in an ordinary batch). This dual-batch, per-label generalization is NOT
    demonstrated in the source papers (they cover a single binary task with separate P/U streams,
    not many simultaneous per-label priors sharing one backbone) -- see pu_learning_plan.md's "Key
    risks" section; treat it as unreviewed until validated per-term against a --run-diagnostic
    AUROC/F1 comparison against the plain-BCE baseline.

    Uses the logistic (softplus) surrogate loss -- the same one BCEWithLogitsLoss implicitly uses
    -- so risk magnitudes stay comparable to the rest of this script's loss reporting.

    Only returns the mean risk over eligible columns; combining it with the plain-BCE loss for
    non-eligible columns is the caller's responsibility (see main()'s training loop).
    """
    eligible_idx = eligible_mask.nonzero(as_tuple=True)[0]
    if eligible_idx.numel() == 0:
        return torch.zeros((), device=logits_u.device)

    softplus_pos_u = torch.nn.functional.softplus(-logits_u[:, eligible_idx])
    softplus_neg_u = torch.nn.functional.softplus(logits_u[:, eligible_idx])
    u_mask = (hard_labels_u[:, eligible_idx] == 0).float()
    n_u = u_mask.sum(dim=0).clamp(min=1)
    r_u_neg = (softplus_neg_u * u_mask).sum(dim=0) / n_u

    softplus_pos_p = torch.nn.functional.softplus(-logits_p[:, eligible_idx])
    softplus_neg_p = torch.nn.functional.softplus(logits_p[:, eligible_idx])
    p_mask = (hard_labels_p[:, eligible_idx] == 1).float()
    n_p = p_mask.sum(dim=0).clamp(min=1)
    r_p_pos = (softplus_pos_p * p_mask).sum(dim=0) / n_p
    r_p_neg = (softplus_neg_p * p_mask).sum(dim=0) / n_p

    pi = priors[eligible_idx]
    neg_risk = r_u_neg - pi * r_p_neg
    positive_part = pi * r_p_pos
    # Non-negative correction: once a label's empirical negative risk drops below -beta (which
    # can't happen in the population limit -- it's finite-sample noise from too few positives --
    # but does happen routinely with flexible/deep models, per Kiryo et al.'s uPU-overfitting
    # finding), switch to ascending on it instead of continuing to trust it.
    corrected = torch.where(neg_risk < -beta, positive_part - gamma * neg_risk, positive_part + neg_risk)
    return corrected.mean()


def init_trackio_run(run_name: str, space_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    if trackio is None:
        print("[INFO] trackio is not installed; skipping experiment tracking.")
        return None

    try:
        trackio.init(project="cafa6", name=run_name, space_id=space_id, config=config)
        return trackio
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"[WARN] Could not initialize trackio: {exc}")
        return None


def get_hf_token(hf_token_file: Optional[str] = None) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
        if token:
            print("[INFO] HF token resolved via Colab Secrets")
            return token
    except Exception:
        pass

    if hf_token_file:
        try:
            token = Path(hf_token_file).read_text().strip()
            os.remove(hf_token_file)
            if token:
                print("[INFO] HF token resolved via --hf-token-file")
                return token
        except Exception as exc:
            print(f"[WARN] Could not read --hf-token-file {hf_token_file}: {exc}")

    env_token = os.getenv("HF_TOKEN")
    if env_token:
        print("[INFO] HF token resolved via HF_TOKEN env var")
        return env_token

    cached_token = get_token()
    if cached_token:
        print("[INFO] HF token resolved via huggingface_hub cache")
        return cached_token

    print("[WARN] No HF token found from any source")
    return None


def push_checkpoint(
    model,
    tokenizer,
    repo_id: str,
    branch: str,
    token: str,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    local_dir: Path,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    (local_dir / "trial_metrics.json").write_text(
        json.dumps({"epoch": epoch, **metrics, "config": config}, default=str)
    )

    # A branch name can be reused across runs with a different --use-lora setting (e.g.
    # promoting a LoRA trial to the main branch after a previous full-fine-tune trial
    # held that branch, or vice versa) - upload_folder() only adds/overwrites files
    # present locally, it doesn't clear the branch first, so the *other* checkpoint
    # shape's files silently persist alongside the new ones (a real incident: the main
    # branch of kelmoir/cafa6-esm ended up with both a full model.safetensors from a
    # non-LoRA push and a stale adapter_config.json/adapter_model.safetensors left over
    # from an earlier LoRA promotion, describing two different, incompatible base
    # models). Explicitly deleting the other shape's filenames on every push keeps a
    # branch always representing exactly one coherent checkpoint.
    delete_patterns = (
        ["model.safetensors", "config.json"] if config.get("use_lora")
        else ["adapter_config.json", "adapter_model.safetensors"]
    )

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    api.create_branch(repo_id, branch=branch, exist_ok=True, token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        revision=branch,
        token=token,
        commit_message=f"epoch {epoch}: val_loss={metrics.get('val_loss', float('nan')):.4f}",
        delete_patterns=delete_patterns,
    )


def compute_hierarchy_violations(probs: np.ndarray, edge_index: np.ndarray,
                                  chunk_size: int = 256) -> Dict[str, float]:
    """Direct measurement of what hierarchy_consistency_penalty targets, computed on
    predicted PROBABILITIES (one level removed from the logit-space loss itself) over
    every (validation example, edge) pair: a violation is child_prob > parent_prob,
    regardless of margin. Processed in example-chunks rather than materializing the
    full (n_examples, n_edges) array at once -- negligible at today's ~2000x47068 scale
    but keeps this safe if --diagnostic-max-val-samples is raised well beyond that.
    """
    if len(edge_index) == 0:
        return {"violation_rate": 0.0, "mean_violation_magnitude": 0.0, "n_edges": 0}
    child_idx, parent_idx = edge_index[:, 0], edge_index[:, 1]
    n_violations, violation_sum, total_pairs = 0, 0.0, 0
    for start in range(0, len(probs), chunk_size):
        chunk = probs[start:start + chunk_size]
        diff = chunk[:, child_idx] - chunk[:, parent_idx]
        violated = diff > 0
        n_violations += int(violated.sum())
        violation_sum += float(diff[violated].sum())
        total_pairs += diff.size
    return {
        "violation_rate": n_violations / total_pairs if total_pairs else 0.0,
        "mean_violation_magnitude": violation_sum / n_violations if n_violations else 0.0,
        "n_edges": int(len(edge_index)),
    }


def clamp_hierarchy_consistency(probs: np.ndarray, edge_index: np.ndarray,
                                 max_iterations: int = 32) -> np.ndarray:
    """Post-hoc inference-time fix for GO's true-path rule (Tier 2 item 5 of
    experiments/cafa6_next_steps_plan.md): for every direct (child, parent) edge, raise
    parent_prob to at least child_prob. Complementary to the training-time
    hierarchy_consistency_penalty (item 3) -- this works on ANY model's output
    probabilities, including one with no hierarchy-aware training at all.

    edge_index lists only DIRECT edges, so one sweep over them isn't enough to fully
    propagate a leaf's high score up a multi-hop chain (e.g. grandchild -> child ->
    parent): after one sweep, "child" has absorbed its own children's scores, but
    "parent" was updated from child's PRE-sweep value. Repeating the sweep lets each
    hop's update ripple one level further up per iteration; since a full sweep with no
    change means every edge is already consistent (a fixed point), iterating until that
    happens -- or until max_iterations, a generous bound well above GO's real max
    depth -- guarantees full transitive propagation without needing an explicit
    topological sort over the vocabulary.
    """
    clamped = probs.copy()
    if len(edge_index) == 0:
        return clamped
    child_idx, parent_idx = edge_index[:, 0], edge_index[:, 1]
    for _ in range(max_iterations):
        before = clamped[:, parent_idx].copy()
        # np.maximum.at, not a plain `clamped[:, parent_idx] = ...` assignment: when two
        # edges share the same parent (a term with multiple direct children -- the
        # common case, unlike the rarer multi-parent case), plain fancy-index assignment
        # is last-write-wins per duplicate target and silently drops every child but the
        # last one's contribution. ufunc.at applies each edge's update in turn against
        # the same underlying buffer, so duplicate parents correctly accumulate the max
        # over every one of their children.
        np.maximum.at(clamped, (slice(None), parent_idx), clamped[:, child_idx])
        if np.array_equal(clamped[:, parent_idx], before):
            break
    return clamped


def _diagnostic_metrics_from_arrays(probs: np.ndarray, labels: np.ndarray,
                                     loss_mask: Optional[np.ndarray] = None,
                                     label_is_rare: Optional[np.ndarray] = None):
    """Shared AUROC/F1 computation for run_diagnostic. loss_mask follows the same
    convention as compute_loss's loss_mask (1 = score this cell, 0 = exclude) and
    defaults to all-ones (every cell scored, the original behavior). With
    --use-known-input, revealed/known cells are visible to the model as INPUT, not
    something it predicted -- scoring them too would make the diagnostic trivially
    "predict what you were just handed" rather than measuring the same novel-term
    prediction task the training loss is restricted to (final-review finding: an
    earlier version scored every cell regardless of loss_mask, making its AUROC
    incomparable to a pure-NK run like trial-013's).

    Returns (result_dict, per_label_auroc_array) -- the array is kept separate from
    the dict since it isn't JSON-serializable and run_diagnostic's own print output
    needs its raw min/max.
    """
    if loss_mask is None:
        loss_mask = np.ones_like(labels, dtype=bool)
    else:
        loss_mask = loss_mask.astype(bool)

    per_label_auroc = []
    rare_auroc, common_auroc = [], []
    for j in range(labels.shape[1]):
        col_mask = loss_mask[:, j]
        y_true = labels[col_mask, j]
        if y_true.size == 0 or y_true.min() == y_true.max():
            continue  # not evaluable: no unmasked examples, or single-class
        auroc_j = roc_auc_score(y_true, probs[col_mask, j])
        per_label_auroc.append(auroc_j)
        if label_is_rare is not None:
            (rare_auroc if label_is_rare[j] else common_auroc).append(auroc_j)
    per_label_auroc = np.array(per_label_auroc)

    flat_mask = loss_mask.ravel()
    flat_labels = labels.ravel()[flat_mask]
    flat_probs = probs.ravel()[flat_mask]
    preds_at_half = (flat_probs >= 0.5).astype(float)

    prior = float(flat_labels.mean()) if flat_labels.size else 0.0
    tp = ((preds_at_half == 1) & (flat_labels == 1)).sum()
    fp = ((preds_at_half == 1) & (flat_labels == 0)).sum()
    fn = ((preds_at_half == 0) & (flat_labels == 1)).sum()
    accuracy = float((preds_at_half == flat_labels).mean()) if flat_labels.size else 0.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    pooled_auroc = (roc_auc_score(flat_labels, flat_probs)
                    if flat_labels.size and flat_labels.min() != flat_labels.max() else None)

    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.02, 0.51, 0.02):
        f1_t = f1_score(flat_labels, (flat_probs >= t).astype(float), zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t

    result = {
        "n_val_examples": int(labels.shape[0]),
        "n_labels_evaluable": int(len(per_label_auroc)),
        "n_labels_total": int(labels.shape[1]),
        "mean_auroc": float(per_label_auroc.mean()) if per_label_auroc.size else None,
        "median_auroc": float(np.median(per_label_auroc)) if per_label_auroc.size else None,
        "pooled_auroc": float(pooled_auroc) if pooled_auroc is not None else None,
        "accuracy_at_0.5": accuracy,
        "trivial_baseline_accuracy": float(1 - prior),
        "precision_at_0.5": float(precision),
        "recall_at_0.5": float(recall),
        "f1_at_0.5": float(f1),
        "best_f1": float(best_f1),
        "best_f1_threshold": float(best_thresh),
    }
    if label_is_rare is not None:
        rare_auroc = np.array(rare_auroc)
        common_auroc = np.array(common_auroc)
        result["rare_mean_auroc"] = float(rare_auroc.mean()) if rare_auroc.size else None
        result["rare_median_auroc"] = float(np.median(rare_auroc)) if rare_auroc.size else None
        result["rare_n_labels_evaluable"] = int(rare_auroc.size)
        result["common_mean_auroc"] = float(common_auroc.mean()) if common_auroc.size else None
        result["common_median_auroc"] = float(np.median(common_auroc)) if common_auroc.size else None
        result["common_n_labels_evaluable"] = int(common_auroc.size)
    return result, per_label_auroc


def run_diagnostic(
    model,
    tokenizer: EsmTokenizer,
    val_dataset,
    label_names: List[str],
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
    use_amp: bool = False,
    max_val_samples: int = 2000,
    batch_size: int = 16,
    use_aspect_input: bool = False,
    use_known_input: bool = False,
    hierarchy_edge_index: Optional[np.ndarray] = None,
    label_is_rare: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Post-hoc AUROC/F1 diagnostic, run against the model already resident in memory
    (GPU, post-training) rather than reloading a checkpoint from the Hub - originally a
    separate script (experiments/ancestor_sweep/scripts/diagnostic_eval_stage2.py) meant to be run
    later, locally; folded in here since re-downloading and reloading a checkpoint just
    to immediately re-run inference on it is wasted work when the trained model is
    right there, and running on CPU (the only option for a separate later invocation
    outside Colab) is dramatically slower than the GPU it just trained on for anything
    above ~100M params.

    Exists because this task's raw threshold-0.5 accuracy/recall can look collapsed
    under severe class imbalance (3.58% average positive rate across 383 labels) even
    when the model has learned real per-label ranking signal - trial-050's false alarm,
    where val_recall stayed pinned near 3% despite a real, well-ranking model
    underneath. AUROC is threshold-independent and reveals that signal directly.
    """
    eval_dataset = val_dataset
    if max_val_samples is not None and len(eval_dataset) > max_val_samples:
        eval_dataset = eval_dataset.select(range(max_val_samples))

    loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )

    was_training = model.training
    model.eval()
    all_probs, all_labels, all_loss_masks = [], [], []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if use_aspect_input:
                model_kwargs["aspect_input"] = batch["aspect_input"].to(device)
            if use_known_input:
                model_kwargs["known_indices"] = batch["known_indices"].to(device)
                model_kwargs["known_offsets"] = batch["known_offsets"].to(device)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                logits = model(**model_kwargs).logits
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(batch["labels"])
            if use_known_input:
                all_loss_masks.append(batch["loss_mask"])
    if was_training:
        model.train()

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    # use_known_input: exclude revealed/known cells from the diagnostic the same way
    # compute_loss already excludes them from training - they're model INPUT, not
    # something it predicted, so scoring them isn't a fair comparison against a
    # pure-NK run (final-review finding).
    loss_mask = torch.cat(all_loss_masks).numpy() if all_loss_masks else None
    result, per_label_auroc = _diagnostic_metrics_from_arrays(probs, labels, loss_mask, label_is_rare)
    if hierarchy_edge_index is not None:
        result.update(compute_hierarchy_violations(probs, hierarchy_edge_index))
        # Tier 2 item 5 (experiments/cafa6_next_steps_plan.md): post-hoc clamping, applied
        # here regardless of whether this checkpoint was trained with --hierarchy-loss-weight
        # (item 3) -- this measures what a free inference-time fix alone is worth on ANY
        # model, and whether it adds anything on top of training-time awareness.
        clamped_probs = clamp_hierarchy_consistency(probs, hierarchy_edge_index)
        clamped_result, _ = _diagnostic_metrics_from_arrays(clamped_probs, labels, loss_mask, label_is_rare)
        clamped_result.update(compute_hierarchy_violations(clamped_probs, hierarchy_edge_index))
        result.update({f"clamped_{key}": value for key, value in clamped_result.items()})
    accuracy = result["accuracy_at_0.5"]
    prior = 1 - result["trivial_baseline_accuracy"]
    precision = result["precision_at_0.5"]
    recall = result["recall_at_0.5"]
    f1 = result["f1_at_0.5"]
    pooled_auroc = result["pooled_auroc"]
    best_f1 = result["best_f1"]
    best_thresh = result["best_f1_threshold"]

    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC: {result['n_labels_evaluable']}/{result['n_labels_total']} labels evaluable "
          f"(both classes present), {result['n_val_examples']} val examples")
    print("=" * 70)
    if per_label_auroc.size:
        print(f"Per-label AUROC: mean={result['mean_auroc']:.4f} median={result['median_auroc']:.4f} "
              f"min={per_label_auroc.min():.4f} max={per_label_auroc.max():.4f}")
        print(f"  labels with AUROC > 0.5: {(per_label_auroc > 0.5).sum()}/{len(per_label_auroc)}")
        print(f"  labels with AUROC > 0.7: {(per_label_auroc > 0.7).sum()}/{len(per_label_auroc)}")
    else:
        print("Per-label AUROC: no evaluable labels (every label single-class in this val slice)")
    print(f"Pooled @ threshold 0.5: accuracy={accuracy:.4f} (trivial baseline={1 - prior:.4f}) "
          f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    pooled_auroc_str = f"{pooled_auroc:.4f}" if pooled_auroc is not None else "n/a (single-class)"
    print(f"Pooled AUROC={pooled_auroc_str}  Best F1={best_f1:.4f}@threshold={best_thresh:.2f}")
    if "rare_mean_auroc" in result:
        rare_str = f"{result['rare_mean_auroc']:.4f}" if result["rare_mean_auroc"] is not None else "n/a"
        common_str = f"{result['common_mean_auroc']:.4f}" if result["common_mean_auroc"] is not None else "n/a"
        print(f"Rarity-stratified AUROC: rare mean={rare_str} (n={result['rare_n_labels_evaluable']})  "
              f"common mean={common_str} (n={result['common_n_labels_evaluable']})")
    if "violation_rate" in result:
        print(f"Hierarchy violations: rate={result['violation_rate']:.4f} "
              f"mean_magnitude={result['mean_violation_magnitude']:.4f} (n_edges={result['n_edges']})")
    if "clamped_violation_rate" in result:
        print(f"After post-hoc clamping: mean_auroc={result['clamped_mean_auroc']:.4f} "
              f"pooled_auroc={result['clamped_pooled_auroc']:.4f} "
              f"violation_rate={result['clamped_violation_rate']:.4f} "
              f"(n_edges={result['clamped_n_edges']})")
    print(f"[DIAGNOSTIC] {json.dumps(result, default=str)}")

    return result


def load_training_data(data_path: Optional[Path] = None, repo_id: str = "kelmoir/CAFA6"):
    if data_path is not None and data_path.exists():
        return load_from_disk(str(data_path))

    try:
        return load_dataset(repo_id)
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(f"Could not load the dataset. Provide a local dataset at {data_path} or ensure the Hugging Face dataset {repo_id} is available. Original error: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight ESM-based CAFA6 classifier")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--model-name", default=SELECTED_MODEL)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--data-path", default="data/cafa-6-protein-function-prediction/dataset",
                         help="Local on-disk dataset (load_from_disk), e.g. "
                              "data/cafa-6-protein-function-prediction/dataset_raw for the raw/"
                              "unpropagated 668-term label space instead of the default coarse "
                              "383-term one. Falls back to --dataset-repo-id from the Hub if this "
                              "path doesn't exist.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the finished model to Hugging Face Hub")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID, help="Hugging Face Hub repository id, e.g. kelmoir/cafa6-esm")
    parser.add_argument("--dataset-repo-id", default="kelmoir/CAFA6", help="Hugging Face Hub dataset repository id to fall back to when no local dataset is found at data/cafa-6-protein-function-prediction/dataset (e.g. kelmoir/cafa6-depth1 for the depth-1-only aspect-conditioning ablation)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dropout", type=float, default=0.2, help="Dropout on the classifier head")
    parser.add_argument("--esm-dropout", type=float, default=0.0, help="Dropout on unfrozen ESM encoder layers")
    parser.add_argument("--attention-dropout", type=float, default=0.0, help="Attention dropout on unfrozen ESM encoder layers")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--max-minutes", type=float, default=None, help="Wall-clock safety cutoff; stop after the current epoch once exceeded")
    parser.add_argument("--run-name", default=None, help="Explicit run/trial id; used for trackio run name and checkpoint branch name")
    parser.add_argument("--trackio-space-id", default=None, help="HF Space id to persist trackio runs to, e.g. kelmoir/cafa6-trackio")
    parser.add_argument("--checkpoint-mode", choices=["none", "final", "every-epoch"], default="final", help="'final' pushes one commit (the last completed epoch) to the trial's Hub branch; 'every-epoch' pushes after each epoch (needed for --resume-from-revision retries); 'none' disables Hub checkpointing entirely")
    parser.add_argument("--checkpoint-branch", default=None, help="Overrides the auto-derived (run-name) branch name for per-epoch checkpoints")
    parser.add_argument("--hf-token-file", default=None, help="Path to a file containing only an HF token; read once, then deleted")
    parser.add_argument("--resume-from-revision", default=None, help="Load model weights from this revision (branch) of --hf-repo-id instead of the base pretrained model; used to continue a multi-epoch trial across separate colab exec invocations after a connection drop")
    parser.add_argument("--start-epoch", type=int, default=0, help="Count of already-completed epochs when resuming; offsets epoch numbering in logs/checkpoints (0 for a fresh run)")
    parser.add_argument("--lr-schedule", choices=["constant", "linear_decay", "cosine"], default="constant", help="Shape of the post-warmup LR schedule")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override the default warmup step count (default: min(100, steps_per_epoch // 10))")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Smooth 0/1 BCE targets toward 0.5 by this amount (0 disables)")
    parser.add_argument("--unknown-target-value", type=float, default=None, help="Replace every currently-0 (i.e. unannotated/unknown, per GO's Open World Assumption -- see experiments/ancestor_sweep/sweep_log.md's trial-080/081 investigation) BCE target with this value instead of hard 0.0; confirmed positives (1.0) are never touched. None (default) keeps today's behavior. Unlike --label-smoothing (which softens both classes symmetrically), this only touches the unknown side -- the gentler, soft-target alternative trial-081's own closing note recommended after hard masking made AUROC measurably worse.")
    parser.add_argument("--decouple-bias-wd", action=argparse.BooleanOptionalAction, default=False, help="Exclude bias and LayerNorm params from weight decay (standard AdamW refinement)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Accumulate gradients over this many steps before each optimizer step (raises effective batch size)")
    parser.add_argument("--pos-weight", action=argparse.BooleanOptionalAction, default=False, help="Weight positive examples in BCE loss by (n_negative/n_positive)**--pos-weight-exponent per label, computed from the training split - helps severely imbalanced multi-label tasks (e.g. the coarse GO-term label space, ~3.6% average positive rate) at the cost of one more axis to tune")
    parser.add_argument("--pos-weight-exponent", type=float, default=0.5, help="Compression exponent for --pos-weight: pos_weight = (n_negative/n_positive)**exponent per label. 0.5=sqrt (default; trial-051 used this), 1.0=raw inverse frequency (not recommended - spans ~12,600x on the coarse-GO-term dataset), lower=gentler correction")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False, help="Suppress per-step [STEP] prints (only needed to keep a colab-cli exec websocket alive; noisy for interactive/notebook use)")
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=False, help="Fine-tune via LoRA adapters (peft) instead of freeze-mostly/unfreeze-last-n: freezes the entire backbone and trains small low-rank adapters plus the classifier head, letting a much bigger model fit in limited VRAM. --unfreeze-last-n is ignored when this is set")
    parser.add_argument("--use-aspect-input", action=argparse.BooleanOptionalAction, default=False, help="Feed the dataset's ground-truth aspect_labels column (a 3-way [C,F,P] multi-hot, see generate_dataset_depth1.py) into the classifier head as an auxiliary input, concatenated to the pooled ESM sequence representation. Requires a dataset with an aspect_labels column (e.g. kelmoir/cafa6-depth1); not implemented with --use-lora")
    parser.add_argument("--use-known-input", action=argparse.BooleanOptionalAction, default=False,
                         help="Synthetically reveal a random subset of each training example's own "
                              "positive GO terms as a 'known' input to the classifier head (see "
                              "KnownAwareClassificationHead / docs/superpowers/specs/"
                              "2026-10-08-known-annotation-conditioning-design.md), excluding revealed "
                              "terms from the loss. Re-randomized every epoch (on-the-fly, no new "
                              "dataset). Mutually exclusive with --use-aspect-input, --use-lora, "
                              "--mask-sparse-negatives-max-total-terms, and --loss-mask-from-dataset.")
    parser.add_argument("--known-mix-zero-frac", type=float, default=0.3,
                         help="Fraction of --use-known-input draws forced to zero known terms (pure "
                              "NK), so the model doesn't forget how to perform the original NK task.")
    parser.add_argument("--known-reveal-max", type=float, default=1.0,
                         help="Upper bound of the Uniform(0, known_reveal_max) reveal-fraction "
                              "sampling range for non-zero --use-known-input draws. Must be in (0, 1].")
    parser.add_argument("--known-embed-dim", type=int, default=64,
                         help="Embedding dimension per GO term for --use-known-input's "
                              "nn.EmbeddingBag known-term pooling.")
    parser.add_argument("--use-label-embedding-head", action=argparse.BooleanOptionalAction, default=False,
                         help="Item 8 (see experiments/cafa6_next_steps_plan.md): replace the flat "
                              "classifier weight with LabelEmbeddingClassificationHead, a small learned "
                              "projection scored by cosine similarity against precomputed GO-term "
                              "embeddings (--label-embeddings-path). Mutually exclusive with --use-lora, "
                              "--use-aspect-input, --use-known-input.")
    parser.add_argument("--label-embeddings-path", default="experiments/label_embedding/term_embeddings.pt",
                         help="Cached term-embedding tensor from experiments/label_embedding/"
                              "build_term_embeddings.py, used only when --use-label-embedding-head is set. "
                              "Its term order must exactly match the dataset's own label2id ordering.")
    parser.add_argument("--hierarchy-loss-weight", type=float, default=0.0,
                         help="Weight on a training-time hinge penalty (relu(child_logit - "
                              "parent_logit), see hierarchy_consistency_penalty) that discourages "
                              "scoring a GO term's child higher than its parent -- GO's true-path "
                              "rule, a logical fact independent of annotation completeness (see "
                              "docs/superpowers/specs/2026-10-11-go-dag-hierarchy-consistency-"
                              "design.md). 0.0 (default) disables the mechanism entirely -- no edge "
                              "index is built, no change to any existing invocation.")
    parser.add_argument("--go-dependency-graph-path", default=None,
                         help="Path to generate_dataset_full.py's exported GO graph (see "
                              "export_dependency_graph in generate_dataset.py). Defaults to "
                              "<--data-path>/go_dependency_graph.json. Required (and must exist) "
                              "when --hierarchy-loss-weight > 0; optional for --run-diagnostic's "
                              "violation-rate reporting, which degrades gracefully (a [WARN] and "
                              "skipped fields) if the file isn't found.")
    parser.add_argument("--soft-f1-loss-weight", type=float, default=0.0,
                         help="Weight on an additive, IA-weighted soft-F1 loss term (see "
                              "soft_f1_loss) computed per protein and added to BCE -- two "
                              "independent CAFA-5 top-6 teams (Evans, Freddolino Lab) found this "
                              "beat plain BCE at this label cardinality/imbalance. 0.0 (default) "
                              "disables the mechanism entirely -- no IA file is read, no change "
                              "to any existing invocation.")
    parser.add_argument("--ia-file", default="data/cafa-6-protein-function-prediction/IA.tsv",
                         help="Path to the CAFA Information Accretion weights file (GO:ID<tab>"
                              "float per line, see load_ia_weights). Only read when "
                              "--soft-f1-loss-weight > 0.")
    parser.add_argument("--mask-sparse-negatives-max-total-terms", type=int, default=None, help="Exclude 'negative' (0) labels from the loss for proteins whose total_terms (raw GO annotation count across all aspects, see generate_dataset_aspect.py) is <= this value - an annotation-completeness proxy for GO's Open World Assumption (absence of annotation != absence of function). Positive (1) labels are always trusted regardless of total_terms. Requires a dataset with a total_terms column (e.g. kelmoir/cafa6-aspect); no-op otherwise")
    parser.add_argument("--loss-mask-from-dataset", action=argparse.BooleanOptionalAction, default=False,
                         help="Two-tier discrete relabeling (see experiments/scripts/build_two_tier_dataset.py, "
                              "the OWA approach revived after --nnpu's training-time mechanism failed three "
                              "separate ways -- see project_pu_learning_nnpu_km2.md): reads a precomputed sparse "
                              "'excluded_labels' column from the dataset and reuses the SAME loss_mask/"
                              "compute_loss machinery --mask-sparse-negatives-max-total-terms uses, instead of "
                              "deriving the mask from total_terms. A purely offline data change -- no nnPU-style "
                              "per-step training code involved. Requires a dataset with an excluded_labels column.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (only used with --use-lora)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor, typically 2x --lora-r (only used with --use-lora)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout on LoRA adapter layers (only used with --use-lora)")
    parser.add_argument("--lora-target-modules", default=None, help="Comma-separated submodule names to attach LoRA adapters to (default: query,value - the original LoRA paper's minimal config). Only used with --use-lora")
    parser.add_argument("--lora-layers-to-transform", type=int, default=None, help="Restrict LoRA adapters to only the last N backbone layers, instead of every layer (the default). This isn't just fewer adapters - PyTorch's autograd naturally skips the backward pass entirely for the untouched earlier layers (verified directly: their output activations have requires_grad=False), since no trainable parameter depends on them. Forward compute still runs for every layer, but backward - typically the more expensive half - only runs for the last N. Trades adaptation breadth for real wall-clock savings on large models. Only used with --use-lora")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no", help="Enable autocast mixed-precision training - needed to fit larger models (e.g. via --use-lora) in limited VRAM, and also controls the dtype frozen LoRA base weights are loaded in. bf16 is preferred on Ampere+ GPUs (A10G, L4, A100); fp16 needs GradScaler (handled automatically) and works on older GPUs (T4) too")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False, help="Trade compute for activation memory by recomputing the forward pass during backward - needed to fit larger models/batches in limited VRAM")
    parser.add_argument("--checkpoint-selection", choices=["val_loss", "f1", "recall"], default="val_loss", help="Metric that decides which epoch's weights get saved as best_checkpoint (and therefore used for --run-diagnostic/--push-to-hub/Hub promotion). 'val_loss' (default, lower is better) preserves prior behavior. 'f1'/'recall' (higher is better) select by the same pooled-f1/macro-recall numbers now printed every epoch (see finalize_metrics()) instead -- val_loss can look flat or even worsen slightly while ranking quality (f1/recall) keeps improving, especially under this task's severe class imbalance (see compute_metrics's docstring / trial-050), so it isn't always the metric that actually reflects which epoch is best. Mirrors experiments/mf_span/scripts/train_mf_span_model.py's --checkpoint-selection.")
    parser.add_argument("--run-diagnostic", action=argparse.BooleanOptionalAction, default=False, help="Run the post-training AUROC/F1 diagnostic (see run_diagnostic()) against the in-memory model right after training finishes, instead of needing a separate later script run against a re-downloaded checkpoint")
    parser.add_argument("--diagnostic-max-val-samples", type=int, default=2000, help="Validation examples to use for --run-diagnostic (independent of --max-val-samples, which controls the training loop's own per-epoch validation pass and may be set much smaller for a quick calibration trial)")
    parser.add_argument("--rare-term-threshold", type=int, default=100,
                         help="Train-split annotation-count boundary between 'rare' and 'common' "
                              "for --run-diagnostic's rarity-stratified AUROC breakdown (a term "
                              "with fewer than this many positive examples in the train split is "
                              "'rare'). Matches this item's roadmap source's own rare-term "
                              "definition (docs/superpowers/specs/2026-10-11-go-dag-hierarchy-"
                              "consistency-design.md).")
    parser.add_argument("--nnpu", action=argparse.BooleanOptionalAction, default=False, help="Replace BCE with the nnPU risk (Kiryo, Niu, du Plessis & Sugiyama 2017) for labels with a cached class prior -- see experiments/pu_learning_plan.md. Labels without a cached, above-floor prior keep today's plain BCE (with --unknown-target-value/--label-smoothing applying to them as before). Requires --class-priors-path.")
    parser.add_argument("--class-priors-path", default=None, help="Path to a class-priors cache (class_priors_km2.json from experiments/scripts/estimate_class_priors_km2.py, or class_priors_spy_heldout.json from estimate_class_priors_spy_heldout.py, the current recommended source -- see --class-priors-min-source-positives). Required by --nnpu.")
    parser.add_argument("--class-priors-min-source-positives", type=int, default=2000,
                         help="A cached prior is only trusted if the ESTIMATOR itself saw at least this many "
                              "positives for that term (the cache's own 'n_positives' field) -- an eligibility "
                              "floor on the prior's reliability, separate from --nnpu-min-term-positives (which "
                              "guards THIS run's positive-anchor batch composition instead). Calibrated against "
                              "the real held-out spy retrain (2026-09-20): pi_spy_heldout/true_prevalence ratio "
                              "was a clean, monotonic function of source positive count, from ~109x median "
                              "overestimation at 100-200 positives down to 1.1x-4.7x at >=2000 -- see "
                              "project_pu_learning_nnpu_km2.md v1.3. Set to 0 to disable (trust every cached, "
                              "above-nnpu-min-term-positives entry, the pre-v1.3 behavior).")
    parser.add_argument("--nnpu-min-term-positives", type=int, default=20, help="A label needs at least this many positives in THIS run's training split to be nnPU-eligible even if it has a cached prior -- protects the positive-anchor batch (see --nnpu-positive-batch-size) from being built on too few real examples in a small/scale-limited run.")
    parser.add_argument("--nnpu-positive-batch-size", type=int, default=32, help="Size of the dedicated positive-anchor batch forwarded once per training step alongside the ordinary batch, so every nnPU-eligible label has real positive rows to compute its R_p+/R_p- risk terms from -- real positives are far too rare (0.556% observed rate) to show up reliably within an ordinary batch. Drawn from all train-split proteins with >=1 eligible-label positive, independent of --max-train-samples.")
    parser.add_argument("--nnpu-beta", type=float, default=0.0, help="Kiryo et al.'s non-negative-risk correction threshold: when a label's empirical (R_u- - pi*R_p-) drops below -beta, gradient ascent is applied to it instead of continuing to descend (prevents the classic uPU overfitting failure mode on flexible/deep models).")
    parser.add_argument("--nnpu-gamma", type=float, default=1.0, help="Scale of the gradient-ascent correction applied when a label's negative risk drops below -beta (see --nnpu-beta).")
    parser.add_argument("--nnpu-detach-anchor-backbone", action=argparse.BooleanOptionalAction, default=True,
                         help="v1.4: run the positive-anchor batch's backbone forward pass under no_grad, so its "
                              "backward pass only updates the classifier head, not the shared unfrozen backbone "
                              "layers. Diagnosed as necessary after a real trial showed --nnpu degrading AUROC on "
                              "the ~650+ NON-eligible labels by roughly the same amount whether 11 or 437 terms "
                              "were nnPU-eligible -- a training-dynamics cost from the anchor batch's gradient "
                              "reshaping the shared backbone, independent of prior calibration. Pass "
                              "--no-nnpu-detach-anchor-backbone to restore the old (pre-v1.4) behavior for an "
                              "ablation comparison.")
    args = parser.parse_args()

    if args.use_known_input:
        if args.mask_sparse_negatives_max_total_terms is not None or args.loss_mask_from_dataset:
            raise ValueError(
                "--use-known-input cannot be combined with --mask-sparse-negatives-max-total-terms "
                "or --loss-mask-from-dataset -- all three write loss_mask and there's no defined "
                "combined semantics (see docs/superpowers/specs/"
                "2026-10-08-known-annotation-conditioning-design.md)")
        if not (0.0 < args.known_reveal_max <= 1.0):
            raise ValueError(f"--known-reveal-max must be in (0, 1], got {args.known_reveal_max}")

    go_dependency_graph_path = (Path(args.go_dependency_graph_path) if args.go_dependency_graph_path
                                 else Path(args.data_path) / "go_dependency_graph.json")
    if args.hierarchy_loss_weight > 0 and not go_dependency_graph_path.exists():
        raise ValueError(
            f"--hierarchy-loss-weight={args.hierarchy_loss_weight} requires the GO dependency "
            f"graph at {go_dependency_graph_path} (see --go-dependency-graph-path) -- not found.")

    ia_file_path = Path(args.ia_file)
    if args.soft_f1_loss_weight > 0 and not ia_file_path.exists():
        raise ValueError(
            f"--soft-f1-loss-weight={args.soft_f1_loss_weight} requires the IA weights file at "
            f"{ia_file_path} (see --ia-file) -- not found.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    config = vars(args)
    print(f"[CONFIG] {json.dumps(config, default=str)}")

    device = torch.device(args.device)
    data_path = Path(args.data_path)
    dataset = load_training_data(data_path=data_path, repo_id=args.dataset_repo_id)

    dataset_train = dataset["train"]
    label_names = [str(name) for name in dataset_train.features["labels"].feature.names]
    id2label = {index: item for index, item in enumerate(label_names)}
    label2id = {value: key for key, value in id2label.items()}

    hierarchy_edge_index = None
    if args.hierarchy_loss_weight > 0 or args.run_diagnostic:
        if go_dependency_graph_path.exists():
            hierarchy_edge_index = build_hierarchy_edge_index(label2id, go_dependency_graph_path)
            print(f"[INFO] Built hierarchy edge index: {hierarchy_edge_index.shape[0]} true-path "
                  f"edges within the {len(label2id)}-term trainable vocabulary")
        elif args.hierarchy_loss_weight == 0:
            print(f"[WARN] --run-diagnostic requested violation-rate reporting but "
                  f"{go_dependency_graph_path} was not found -- skipping hierarchy-violation "
                  f"diagnostic fields (rarity-stratified AUROC is unaffected).")

    ia_weights = None
    if args.soft_f1_loss_weight > 0:
        ia_weights = torch.tensor(load_ia_weights(ia_file_path, label_names), dtype=torch.float32)
        print(f"[INFO] Loaded IA weights for {len(label_names)} terms "
              f"(mean={ia_weights.mean():.4f}, nonzero={int((ia_weights > 0).sum())})")

    term_embeddings = None
    if args.use_label_embedding_head:
        cached = torch.load(args.label_embeddings_path)
        if cached["terms"] != label_names:
            raise ValueError(
                f"{args.label_embeddings_path}'s cached term order does not match this dataset's "
                "label2id ordering -- regenerate experiments/label_embedding/build_term_embeddings.py "
                "against the dataset actually in use at --data-path/--dataset-repo-id."
            )
        term_embeddings = cached["embeddings"]
        print(f"[INFO] Loaded term embeddings for --use-label-embedding-head: "
              f"{term_embeddings.shape} from {args.label_embeddings_path}")

    tokenizer = create_esm_tokenizer(args.model_name)
    preprocess_batch_partial = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        max_length=args.max_length,
        unique_items=label_names,
        label2id=label2id,
        mask_sparse_negatives_max_total_terms=args.mask_sparse_negatives_max_total_terms,
        loss_mask_from_dataset=args.loss_mask_from_dataset,
        use_known_input=args.use_known_input,
        known_mix_zero_frac=args.known_mix_zero_frac,
        known_reveal_max=args.known_reveal_max,
    )
    # Validation/test get a FIXED known/novel split (known_split_seed), not the
    # training split's intentional per-epoch re-randomization: with_transform is
    # lazy, so re-using preprocess_batch_partial as-is would re-randomize val/test on
    # every epoch too, making val_loss (which drives best-checkpoint selection) and
    # --run-diagnostic's AUROC noisy across epochs instead of a fixed holdout (code
    # review finding on this plan's final review).
    eval_preprocess_batch_partial = partial(preprocess_batch_partial, known_split_seed=args.seed)

    processed_dataset = {
        "train": dataset["train"].with_transform(transform=preprocess_batch_partial),
        "validation": dataset["validation"].with_transform(transform=eval_preprocess_batch_partial),
        "test": dataset["test"].with_transform(transform=eval_preprocess_batch_partial),
    }

    # train stays unsliced here (unlike validation) so each epoch can draw a different
    # --max-train-samples-sized subset below, instead of memorizing the same fixed rows
    # every epoch of a multi-epoch run.
    full_validation_dataset = processed_dataset["validation"]  # kept unsliced for --run-diagnostic,
    # independent of --max-val-samples below (the training loop's own per-epoch validation
    # pass, which a quick calibration trial may cap much smaller than the diagnostic wants)
    if args.max_val_samples is not None:
        processed_dataset["validation"] = processed_dataset["validation"].select(range(min(args.max_val_samples, len(processed_dataset["validation"]))))

    # A --use-lora checkpoint on --hf-repo-id is adapter-only (a few MB of low-rank
    # deltas), not a full standalone model - it can never itself serve as the base to
    # load AutoModelForSequenceClassification from the way a full-fine-tune checkpoint
    # can. So the LoRA base architecture always loads fresh from --model-name, and
    # resuming pulls adapter weights separately via lora_resume_repo_id.
    lora_base_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision)
    model_esm = create_esm(
        unfreeze_last_n=args.unfreeze_last_n,
        model_type=(args.hf_repo_id if (args.resume_from_revision and not args.use_lora) else args.model_name),
        output_dropout=args.output_dropout,
        esm_dropout=args.esm_dropout,
        attention_dropout=args.attention_dropout,
        id2label=id2label,
        label2id=label2id,
        revision=args.resume_from_revision,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
        lora_resume_repo_id=(args.hf_repo_id if (args.resume_from_revision and args.use_lora) else None),
        lora_base_dtype=lora_base_dtype,
        lora_layers_to_transform=args.lora_layers_to_transform,
        gradient_checkpointing=args.gradient_checkpointing,
        use_aspect_input=args.use_aspect_input,
        use_known_input=args.use_known_input,
        known_embed_dim=args.known_embed_dim,
        use_label_embedding_head=args.use_label_embedding_head,
        term_embeddings=term_embeddings,
    ).to(device)
    # Computed unconditionally (not just on a fresh start) since --pos-weight needs it
    # regardless of whether we're resuming from a checkpoint.
    raw_train_labels = dataset_train["labels"]
    label_counts = torch.zeros(len(label_names))
    for labs in raw_train_labels:
        for idx in labs:
            label_counts[int(idx)] += 1

    if args.resume_from_revision:
        print(f"[INFO] Resumed model weights from {args.hf_repo_id}@{args.resume_from_revision}")
    elif args.use_label_embedding_head:
        # The per-term prior-bias trick below needs one bias per term (out_proj.bias,
        # shape [num_labels]) to saturate toward each label's own prior. LabelEmbeddingClassificationHead
        # has a single global scale/bias instead (shared across every term by construction --
        # that's the whole point of not having a per-term weight) -- there's no per-term slot
        # to initialize this way, so it's skipped rather than attempted on the wrong shape.
        print("[INFO] --use-label-embedding-head set: skipping per-term prior-bias init "
              "(the head's scale/bias are global, not per-term).")
    else:
        # BCE with an imbalanced label prior otherwise wastes early gradient budget just
        # learning the bias (saturating logits toward the prior), which shrinks per-example
        # gradient magnitude and starves the weight matrix of signal to actually discriminate.
        label_priors = (label_counts / max(1, len(raw_train_labels))).clamp(1e-4, 1 - 1e-4)
        prior_bias = torch.log(label_priors / (1 - label_priors)).to(device)
        with torch.no_grad():
            model_esm.classifier.out_proj.bias.copy_(prior_bias)
        print(
            f"[INFO] Initialized classifier bias to label priors for {len(label_names)} labels "
            f"(mean={label_priors.mean().item():.4f}, min={label_priors.min().item():.4f}, "
            f"max={label_priors.max().item():.4f})"
        )

    class_priors_tensor = torch.zeros(len(label_names), device=device)
    eligible_mask = torch.zeros(len(label_names), dtype=torch.bool, device=device)
    positive_iter = None
    if args.nnpu:
        if args.class_priors_path is None:
            raise ValueError("--nnpu requires --class-priors-path (see experiments/scripts/estimate_class_priors_km2.py)")
        with open(args.class_priors_path) as f:
            cached_priors = json.load(f)["priors"]
        # v1.3 (2026-09-20, see pu_learning_plan.md / project_pu_learning_nnpu_km2.md): KM2 and the
        # old classifier-probability pi_spy (scored by the SAME checkpoint trained on those exact
        # positives -- not a real held-out spy at all) both failed empirically, as did a
        # centroid-based ratio estimator. estimate_class_priors_spy_heldout.py fixes both the
        # held-out-ness (a genuine spy retrain, from a fresh backbone, that never saw the held-out
        # positives) and the formula (the actual Elkan-Noto ratio E[g|unlabeled]/E[g|known positive],
        # not just the numerator alone). Result: NOT uniform garbage like the prior three attempts --
        # a clean, monotonic relationship between a term's source positive count and how close the
        # ratio lands to 1.0 (median ~109x at 100-200 positives, down to 1.1x-4.7x at >=2000). Below
        # --class-priors-min-source-positives the estimate is still too noisy (small-sample variance
        # in a tiny mean, not a lack of underlying signal) to trust, so those terms keep plain BCE
        # rather than getting a wrong-scale nnPU correction the way the v1.2 full trial demonstrated
        # (a badly-scaled pi moves the loss surface's minimum away from the correct answer entirely).
        n_below_floor = 0
        n_below_source_floor = 0
        for term_name, entry in cached_priors.items():
            if term_name not in label2id:
                continue
            idx = label2id[term_name]
            if label_counts[idx].item() < args.nnpu_min_term_positives:
                n_below_floor += 1
                continue
            if entry.get("n_positives", 0) < args.class_priors_min_source_positives:
                n_below_source_floor += 1
                continue
            eligible_mask[idx] = True
            class_priors_tensor[idx] = entry.get("pi_spy_heldout", entry.get("pi_tice", entry.get("pi_spy")))
        n_eligible = int(eligible_mask.sum().item())
        n_uncached = len(label_names) - n_eligible - n_below_floor - n_below_source_floor
        print(f"[INFO] --nnpu: {n_eligible} labels eligible for nnPU ({n_below_floor} cached but below "
              f"--nnpu-min-term-positives={args.nnpu_min_term_positives}, {n_below_source_floor} cached but "
              f"below --class-priors-min-source-positives={args.class_priors_min_source_positives}, "
              f"{n_uncached} not in cache -- all three groups keep plain BCE)")
        if n_eligible == 0:
            raise ValueError("--nnpu: no labels eligible after applying --nnpu-min-term-positives; "
                              "check --class-priors-path or lower the floor")

        eligible_names = {label_names[i] for i in range(len(label_names)) if eligible_mask[i]}
        # Independent of --max-train-samples: real positives are rare enough (0.556% observed
        # rate) that scale-limiting the anchor pool the same way as the main training stream would
        # starve rarer eligible labels of any anchor rows at all.
        positive_pool_indices = [
            row_idx for row_idx, labs in enumerate(raw_train_labels)
            if any(label_names[int(idx)] in eligible_names for idx in labs)
        ]
        print(f"[INFO] --nnpu positive-anchor pool: {len(positive_pool_indices)} proteins (>=1 eligible-label positive)")
        positive_pool_dataset = processed_dataset["train"].select(positive_pool_indices)
        positive_dataloader = torch.utils.data.DataLoader(
            positive_pool_dataset,
            batch_size=args.nnpu_positive_batch_size,
            shuffle=True,
            collate_fn=partial(data_collate_function, tokenizer=tokenizer),
        )

        def _cycle_positive_batches():
            while True:
                for pos_batch in positive_dataloader:
                    yield pos_batch

        positive_iter = _cycle_positive_batches()

    def build_train_dataloader(epoch: int) -> torch.utils.data.DataLoader:
        train_subset = processed_dataset["train"]
        raw_subset = dataset_train
        if args.max_train_samples is not None:
            n = min(args.max_train_samples, len(train_subset))
            train_subset = train_subset.shuffle(seed=args.seed + epoch).select(range(n))
            # dataset_train (pre-with_transform) shuffled with the identical seed/length
            # yields the identical permutation as train_subset above - HF's .shuffle(seed=)
            # depends only on dataset length and seed, not content - so raw_subset's row
            # order lines up with train_subset's without needing to touch processed_dataset
            # itself (which can't be indexed by "sequence": with_transform intercepts
            # column access too, not just row access, and preprocess_batch's output has no
            # "sequence" key to give back).
            raw_subset = raw_subset.shuffle(seed=args.seed + epoch).select(range(n))
        # Batches assembled from a plain random shuffle can by chance cluster several
        # near-max-length sequences together, spiking that one batch's padding/activation
        # memory well above the epoch's typical batch even at a fixed batch_size - this is
        # what made trial-076 OOM at the exact batch_size=16 trial-075 already ran clean
        # with, purely from an unlucky per-epoch resample. Grouping same-length sequences
        # into the same batch (mega-batch sort + shuffle, HF's standard `group_by_length`
        # technique) removes that variance instead of trading it away for a smaller
        # batch_size/--grad-accum-steps workaround. It also deliberately runs the batch
        # containing the single longest sequence first, so a config that still doesn't fit
        # fails in seconds, not deep into a paid multi-hour run.
        lengths = [min(len(seq), args.max_length) for seq in raw_subset["sequence"]]
        length_generator = torch.Generator().manual_seed(args.seed + epoch)
        sampler = LengthGroupedSampler(batch_size=args.batch_size, lengths=lengths, generator=length_generator)
        return torch.utils.data.DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=partial(data_collate_function, tokenizer=tokenizer),
        )

    train_dataloader = build_train_dataloader(args.start_epoch)
    validation_dataloader = torch.utils.data.DataLoader(
        processed_dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=partial(data_collate_function, tokenizer=tokenizer),
    )

    if args.decouple_bias_wd:
        decay_params, no_decay_params = [], []
        for name, param in model_esm.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay_params if param.ndim == 1 else decay_params).append(param)
        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
        )
    else:
        optimizer = AdamW(model_esm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, min(100, len(train_dataloader) // 10))
    scheduler = get_lr_schedule(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, len(train_dataloader) * args.epochs),
        schedule_type=args.lr_schedule,
    )
    pos_weight = None
    if args.pos_weight:
        # Compression exponent, not raw n_negative/n_positive: raw inverse frequency on
        # this dataset spans ~0.4-4800x across labels (the 3 majority-positive aspect
        # roots vs. the rarest depth<=2 terms), which would blow up gradients on the rare
        # occasions a rare-term positive appears in a batch. ratio**exponent compresses
        # that range - trial-051 (exponent=0.5, i.e. sqrt) gave a small AUROC gain but a
        # net-worse best-achievable F1 (0.5727 -> 0.5007), overcorrecting precision away;
        # a lower exponent is a gentler correction, worth testing directly.
        n_negative = len(raw_train_labels) - label_counts
        ratio = n_negative / label_counts.clamp(min=1)
        pos_weight = ratio.pow(args.pos_weight_exponent).to(device)
        print(f"[INFO] Using --pos-weight (exponent={args.pos_weight_exponent}): "
              f"min={pos_weight.min():.2f} median={pos_weight.median():.2f} max={pos_weight.max():.2f}")
    use_loss_mask = (args.mask_sparse_negatives_max_total_terms is not None
                      or args.loss_mask_from_dataset or args.use_known_input)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none" if use_loss_mask else "mean")
    hierarchy_edge_index_device = hierarchy_edge_index.to(device) if hierarchy_edge_index is not None else None
    ia_weights_device = ia_weights.to(device) if ia_weights is not None else None

    run_name = args.run_name or f"{args.model_name.split('/')[-1]}-epochs{args.epochs}-bs{args.batch_size}-device{device.type}"
    tracker = init_trackio_run(run_name, space_id=args.trackio_space_id, config=config)
    token = get_hf_token(args.hf_token_file)
    checkpoint_branch = args.checkpoint_branch or run_name
    checkpoint_dir = Path("models") / run_name / "checkpoint"
    best_checkpoint_dir = Path("models") / run_name / "best_checkpoint"
    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_score = float("inf") if args.checkpoint_selection == "val_loss" else float("-inf")
    start_time = time.time()

    use_amp = args.mixed_precision != "no" and device.type == "cuda"
    if args.mixed_precision != "no" and device.type != "cuda":
        print(f"[WARN] --mixed-precision={args.mixed_precision} requested but device is {device.type}; autocast only applies on CUDA, ignoring.")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision)
    use_grad_scaler = use_amp and args.mixed_precision == "fp16"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        if epoch != args.start_epoch:
            train_dataloader = build_train_dataloader(epoch)
        epoch_start = time.time()
        model_esm.train()
        train_loss = 0.0
        train_hierarchy_penalty_sum = 0.0
        train_soft_f1_sum = 0.0
        train_metrics_acc = {"correct": 0.0, "total": 0, "tp": torch.zeros(len(label_names), device=device), "fp": torch.zeros(len(label_names), device=device), "fn": torch.zeros(len(label_names), device=device)}

        log_every = max(1, len(train_dataloader) // 25)
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["labels"].to(device)
            labels = hard_labels
            if args.unknown_target_value is not None:
                # Only the unknown (currently-0) side moves -- confirmed positives stay at 1.0,
                # unlike --label-smoothing below which softens both classes symmetrically.
                labels = torch.where(hard_labels == 0,
                                      torch.full_like(labels, args.unknown_target_value), labels)
            if args.label_smoothing > 0:
                labels = labels * (1 - args.label_smoothing) + 0.5 * args.label_smoothing

            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if args.use_aspect_input:
                model_kwargs["aspect_input"] = batch["aspect_input"].to(device)
            if args.use_known_input:
                model_kwargs["known_indices"] = batch["known_indices"].to(device)
                model_kwargs["known_offsets"] = batch["known_offsets"].to(device)
            loss_mask = batch["loss_mask"].to(device) if use_loss_mask else None
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                outputs = model_esm(**model_kwargs)
                logits = outputs.logits
                if args.nnpu:
                    # Non-eligible columns (no cached, above-floor prior) keep today's plain-BCE
                    # treatment -- including --unknown-target-value/--label-smoothing, already
                    # applied to `labels` above -- computed only over those columns here.
                    non_eligible = ~eligible_mask
                    if non_eligible.any():
                        standard_loss = compute_loss(
                            criterion, logits[:, non_eligible], labels[:, non_eligible],
                            loss_mask[:, non_eligible] if loss_mask is not None else None,
                        )
                    else:
                        standard_loss = torch.zeros((), device=device)
                    pos_batch = next(positive_iter)
                    pos_input_ids = pos_batch["input_ids"].to(device)
                    pos_attention_mask = pos_batch["attention_mask"].to(device)
                    pos_hard_labels = pos_batch["labels"].to(device)
                    if args.nnpu_detach_anchor_backbone:
                        # v1.4 (see project_pu_learning_nnpu_km2.md): the positive-anchor batch's
                        # backward pass shared the same unfrozen backbone layers as the main batch,
                        # which measurably degraded AUROC on the ~650+ non-eligible labels by
                        # roughly the same amount regardless of how many/which terms were actually
                        # nnPU-eligible (11 vs 437, see the floor2000 trial) -- a training-dynamics
                        # cost paid unconditionally just by enabling --nnpu, independent of prior
                        # calibration quality. Running the anchor batch's backbone forward pass
                        # under no_grad and only letting gradient flow through the classifier head
                        # isolates its influence to the eligible labels' own classifier rows: the
                        # head's parameters still get correct gradients from nnpu_component, only
                        # the path back INTO the shared backbone is cut.
                        with torch.no_grad():
                            pos_hidden = model_esm.esm(input_ids=pos_input_ids, attention_mask=pos_attention_mask).last_hidden_state
                        classifier_kwargs = {"aspect_input": pos_batch["aspect_input"].to(device)} if args.use_aspect_input else {}
                        pos_logits = model_esm.classifier(pos_hidden, **classifier_kwargs)
                    else:
                        pos_kwargs = {"input_ids": pos_input_ids, "attention_mask": pos_attention_mask}
                        if args.use_aspect_input:
                            pos_kwargs["aspect_input"] = pos_batch["aspect_input"].to(device)
                        pos_logits = model_esm(**pos_kwargs).logits
                    nnpu_component = nnpu_loss(
                        logits, hard_labels, pos_logits, pos_hard_labels,
                        class_priors_tensor, eligible_mask, beta=args.nnpu_beta, gamma=args.nnpu_gamma,
                    )
                    # Weighted by label-count share so overall loss scale stays comparable to the
                    # plain-BCE path regardless of how many labels this run's cache made eligible.
                    eligible_frac = eligible_mask.float().mean()
                    loss = eligible_frac * nnpu_component + (1 - eligible_frac) * standard_loss
                else:
                    loss = compute_loss(criterion, logits, labels, loss_mask)
                hierarchy_penalty_contribution = 0.0
                if args.hierarchy_loss_weight > 0:
                    # Tracked separately (not folded silently into train_loss) per final-review
                    # finding: with the flag on, train_loss would otherwise become BCE+penalty
                    # combined, making it misleading to compare against val_loss (pure BCE,
                    # unchanged) or against any earlier trial's train_loss curve.
                    hierarchy_penalty_term = args.hierarchy_loss_weight * hierarchy_consistency_penalty(
                        logits, hierarchy_edge_index_device)
                    loss = loss + hierarchy_penalty_term
                    hierarchy_penalty_contribution = hierarchy_penalty_term.item()
                soft_f1_contribution = 0.0
                if args.soft_f1_loss_weight > 0:
                    # Same separate-tracking rationale as train_hierarchy_penalty above --
                    # don't let train_loss silently become BCE+soft_f1 combined.
                    soft_f1_term = args.soft_f1_loss_weight * soft_f1_loss(logits, labels, ia_weights_device)
                    loss = loss + soft_f1_term
                    soft_f1_contribution = soft_f1_term.item()
            scaled_loss = scaler.scale(loss / args.grad_accum_steps) if use_grad_scaler else (loss / args.grad_accum_steps)
            scaled_loss.backward()

            is_accum_boundary = (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_dataloader)
            if is_accum_boundary:
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_esm.parameters(), max_norm=args.grad_clip_norm)
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not args.quiet and (step + 1) % log_every == 0:
                # Periodic output keeps the colab exec websocket carrying traffic during
                # the otherwise-silent multi-minute training loop; a long silent stretch
                # has been observed to trigger "Connection was lost" mid-epoch.
                print(f"[STEP] epoch={epoch + 1} step={step + 1}/{len(train_dataloader)} loss={loss.item():.4f}")

            batch_metrics = compute_metrics(logits, hard_labels)
            train_metrics_acc["correct"] += batch_metrics["correct"]
            train_metrics_acc["total"] += batch_metrics["total"]
            train_metrics_acc["tp"] += batch_metrics["tp"]
            train_metrics_acc["fp"] += batch_metrics["fp"]
            train_metrics_acc["fn"] += batch_metrics["fn"]
            train_loss += loss.item()
            train_hierarchy_penalty_sum += hierarchy_penalty_contribution
            train_soft_f1_sum += soft_f1_contribution

            if tracker is not None:
                # "step" is a trackio-reserved key (silently renamed to "__step" with a
                # UserWarning if used directly) -- "global_step" avoids the collision.
                tracker.log({"epoch": epoch, "global_step": len(train_dataloader) * epoch + 1, "train_loss": loss.item(), **finalize_metrics(batch_metrics)})

        train_loss /= max(1, len(train_dataloader))
        train_hierarchy_penalty = train_hierarchy_penalty_sum / max(1, len(train_dataloader))
        train_soft_f1 = train_soft_f1_sum / max(1, len(train_dataloader))
        train_metrics = finalize_metrics(train_metrics_acc)

        validation_loss = 0.0
        validation_metrics_acc = {"correct": 0.0, "total": 0, "tp": torch.zeros(len(label_names), device=device), "fp": torch.zeros(len(label_names), device=device), "fn": torch.zeros(len(label_names), device=device)}
        model_esm.eval()
        with torch.inference_mode():
            for batch in validation_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if args.use_aspect_input:
                    model_kwargs["aspect_input"] = batch["aspect_input"].to(device)
                if args.use_known_input:
                    model_kwargs["known_indices"] = batch["known_indices"].to(device)
                    model_kwargs["known_offsets"] = batch["known_offsets"].to(device)
                loss_mask = batch["loss_mask"].to(device) if use_loss_mask else None
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                    outputs = model_esm(**model_kwargs)
                    loss = compute_loss(criterion, outputs.logits, labels, loss_mask)
                validation_loss += loss.item()

                batch_metrics = compute_metrics(outputs.logits, labels)
                validation_metrics_acc["correct"] += batch_metrics["correct"]
                validation_metrics_acc["total"] += batch_metrics["total"]
                validation_metrics_acc["tp"] += batch_metrics["tp"]
                validation_metrics_acc["fp"] += batch_metrics["fp"]
                validation_metrics_acc["fn"] += batch_metrics["fn"]

        validation_loss /= max(1, len(validation_dataloader))
        validation_metrics = finalize_metrics(validation_metrics_acc)

        latest_metrics = {
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": validation_metrics["accuracy"],
            "train_recall": train_metrics["recall"],
            "val_recall": validation_metrics["recall"],
            "train_precision": train_metrics["precision"],
            "val_precision": validation_metrics["precision"],
            "train_f1": train_metrics["f1"],
            "val_f1": validation_metrics["f1"],
        }
        if args.hierarchy_loss_weight > 0:
            # Only added when the flag is actually in use -- keeps every existing invocation's
            # metrics dict/trial_metrics.json byte-for-byte unchanged (Global Constraint).
            latest_metrics["train_hierarchy_penalty"] = train_hierarchy_penalty
        if args.soft_f1_loss_weight > 0:
            latest_metrics["train_soft_f1_loss"] = train_soft_f1

        # --checkpoint-selection picks which epoch's weights get saved as best_checkpoint.
        # val_loss (default) is lower-is-better; f1/recall (val_f1/val_recall, both already
        # computed above every epoch) are higher-is-better. best_val_loss is still tracked
        # unconditionally alongside whatever the selection metric is, purely for reporting/
        # trial_metrics.json backward-compat -- it does not itself gate the save when
        # --checkpoint-selection isn't "val_loss".
        if args.checkpoint_selection == "val_loss":
            current_score = validation_loss
        elif args.checkpoint_selection == "f1":
            current_score = validation_metrics["f1"]
        else:
            current_score = validation_metrics["recall"]
        improved = (current_score < best_score) if args.checkpoint_selection == "val_loss" else (current_score > best_score)

        if improved:
            # Tracked unconditionally (not just under --push-to-hub) since it's cheap -
            # a LoRA checkpoint only saves the small adapter (tens of MB), and even a
            # full-fine-tune save is a normal, standard "best checkpoint" cost, not a
            # VRAM risk (save_pretrained doesn't require a second live GPU copy of the
            # model). Local-disk only here, no Hub upload - --checkpoint-mode's
            # every-epoch/final pushes are a separate, unrelated concern (per-epoch
            # history vs. this trial's single best point).
            best_score = current_score
            best_val_loss = validation_loss
            best_epoch = epoch + 1
            best_metrics = latest_metrics
            model_esm.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)

        hierarchy_penalty_str = (f" | train_hierarchy_penalty={train_hierarchy_penalty:.4f}"
                                 if args.hierarchy_loss_weight > 0 else "")
        soft_f1_str = (f" | train_soft_f1_loss={train_soft_f1:.4f}"
                       if args.soft_f1_loss_weight > 0 else "")
        print(
            f"Epoch {epoch + 1}/{args.start_epoch + args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | val_acc={validation_metrics['accuracy']:.4f} | "
            f"train_recall={train_metrics['recall']:.4f} | val_recall={validation_metrics['recall']:.4f} | "
            f"train_f1={train_metrics['f1']:.4f} | val_f1={validation_metrics['f1']:.4f}"
            f"{hierarchy_penalty_str}{soft_f1_str}"
        )
        print(f"[TIMING] epoch {epoch + 1} elapsed={time.time() - epoch_start:.1f}s")

        if tracker is not None:
            tracker.log(
                {
                    "epoch": epoch,
                    **latest_metrics,
                }
            )

        if args.checkpoint_mode == "every-epoch" and token:
            try:
                push_checkpoint(
                    model_esm,
                    tokenizer,
                    repo_id=args.hf_repo_id,
                    branch=checkpoint_branch,
                    token=token,
                    epoch=epoch + 1,
                    metrics=latest_metrics,
                    config=config,
                    local_dir=checkpoint_dir,
                )
                print(f"[INFO] Pushed checkpoint for epoch {epoch + 1} to {args.hf_repo_id}@{checkpoint_branch}")
            except Exception as exc:  # pragma: no cover - runtime dependency
                print(f"[WARN] Checkpoint push failed for epoch {epoch + 1}: {exc}")
        elif args.checkpoint_mode == "every-epoch" and not token:
            print("[WARN] --checkpoint-mode every-epoch requested but no HF token was found; skipping checkpoint push.")

        if args.max_minutes and (time.time() - start_time) / 60 >= args.max_minutes:
            print(f"[INFO] Reached --max-minutes={args.max_minutes} after epoch {epoch + 1}; stopping early.")
            break

    if args.checkpoint_mode == "final":
        if token:
            try:
                push_checkpoint(
                    model_esm,
                    tokenizer,
                    repo_id=args.hf_repo_id,
                    branch=checkpoint_branch,
                    token=token,
                    epoch=epoch + 1,
                    metrics=latest_metrics,
                    config=config,
                    local_dir=checkpoint_dir,
                )
                print(f"[INFO] Pushed final-epoch checkpoint (epoch {epoch + 1}) to {args.hf_repo_id}@{checkpoint_branch}")
            except Exception as exc:  # pragma: no cover - runtime dependency
                print(f"[WARN] Final checkpoint push failed: {exc}")
        else:
            print("[WARN] --checkpoint-mode final requested but no HF token was found; skipping checkpoint push.")

    summary = {
        "run_name": run_name,
        "epochs_completed": epoch + 1,
        "final_train_loss": train_loss,
        "final_val_loss": validation_loss,
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": validation_metrics["accuracy"],
        "train_recall": train_metrics["recall"],
        "val_recall": validation_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "val_f1": validation_metrics["f1"],
        "checkpoint_selection": args.checkpoint_selection,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
    }
    print(f"[SUMMARY] {json.dumps(summary, default=str)}")

    # Resolved once, shared by --run-diagnostic and --push-to-hub below - the model we
    # actually care about reporting/promoting is the BEST epoch, not necessarily
    # whatever the last epoch happened to be (a real gap found via trial-074: val_loss
    # bottomed out at epoch 7 then crept back up by epoch 10 from mild overfitting -
    # model_esm's in-memory weights had moved past the best point by the time this
    # code runs). Reuses create_esm's existing LoRA-resume code path pointed at a local
    # directory instead of a Hub revision - transformers/peft's from_pretrained both
    # accept a local path transparently, same as a Hub repo id.
    best_model = model_esm
    if best_epoch is not None and best_epoch != epoch + 1:
        print(f"[INFO] Using best checkpoint (epoch {best_epoch}, selected by "
              f"--checkpoint-selection={args.checkpoint_selection}={best_score:.4f}, "
              f"val_loss={best_val_loss:.4f}) for diagnostic/promotion, not the last epoch "
              f"({epoch + 1}, val_loss={validation_loss:.4f})")
        if args.use_lora:
            best_model = create_esm(
                model_type=args.model_name,
                use_lora=True,
                lora_resume_repo_id=str(best_checkpoint_dir),
                id2label=id2label,
                label2id=label2id,
                lora_base_dtype=lora_base_dtype,
            ).to(device)
        elif args.use_known_input:
            best_model = KnownConditionedEsmForSequenceClassification.from_pretrained(
                str(best_checkpoint_dir),
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                problem_type="multi_label_classification",
                known_embed_dim=args.known_embed_dim,
            ).to(device)
        else:
            best_model_cls = AspectConditionedEsmForSequenceClassification if args.use_aspect_input else AutoModelForSequenceClassification
            best_model = best_model_cls.from_pretrained(
                str(best_checkpoint_dir),
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                problem_type="multi_label_classification",
            ).to(device)

    if args.run_diagnostic:
        print("[INFO] Running post-training AUROC/F1 diagnostic on the best-checkpoint model...")
        run_diagnostic(
            best_model,
            tokenizer,
            full_validation_dataset,
            label_names,
            device,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            max_val_samples=args.diagnostic_max_val_samples,
            batch_size=args.batch_size,
            use_aspect_input=args.use_aspect_input,
            use_known_input=args.use_known_input,
            hierarchy_edge_index=hierarchy_edge_index.numpy() if hierarchy_edge_index is not None else None,
            label_is_rare=(label_counts.numpy() < args.rare_term_threshold),
        )

    if args.push_to_hub:
        if token:
            print(f"[INFO] Uploading best checkpoint to Hugging Face Hub as {args.hf_repo_id}@main")
            try:
                push_checkpoint(
                    best_model,
                    tokenizer,
                    repo_id=args.hf_repo_id,
                    branch="main",
                    token=token,
                    epoch=(best_epoch if best_epoch is not None else epoch + 1),
                    metrics=(best_metrics if best_metrics is not None else latest_metrics),
                    config=config,
                    local_dir=best_checkpoint_dir,
                )
                print(f"[INFO] Pushed to {args.hf_repo_id}@main")
            except Exception as exc:  # pragma: no cover - runtime dependency
                print(f"[WARN] Hub promotion push failed: {exc}")
        else:
            print("[WARN] Hugging Face Hub upload requested but no token was found; skipping upload.")


if __name__ == "__main__":
    main()



from transformers import EsmTokenizer
import torch
from typing import Dict, List
from datasets import load_dataset
from functools import partial
from transformers import AutoModelForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

MAX_LENGTH = 1024 # The maximum length of the proteins. 
UNIQUE_ASPECTS = 3 # The labels in the dataset ["C", "F", "P"]
unique_aspects = ["C", "F", "P"]
ESM_MODEL_NAMES = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/esm2_t36_3B_UR50D",
    "facebook/esm2_t48_15B_UR50D"
]
SELECTED_MODEL = "facebook/esm2_t12_35M_UR50D"

def create_esm_tokenizer(model_name: str="facebook/esm2_t30_150M_UR50D"):
  # Instantiate EsmTokenizer using from_pretrained, similar to BertTokenizerFast
  # We'll use the same model as for the EsmForSequenceClassification later for consistency.
  return EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

def create_esm(unfreeze_last_n=2,
               model_type:str="facebook/esm2_t30_150M_UR50D",
               output_dropout:float=0.0, # Dropout in the classifier layer
               attention_dropout:float=0.0, # Dropout in the attention layers
               esm_dropout:float=0.0,   #Dropout after the attention layers
               ):
  model = AutoModelForSequenceClassification.from_pretrained(model_type,
                                    num_labels= len(id2label),
                                    id2label = id2label,
                                    label2id=label2id,
                                    token=userdata.get("Huggingface"))

  # Erst alles einfrieren
  for param in model.esm.parameters():
    param.requires_grad = False

  # Dann die letzten n Layer des Encoders wieder auftauen, limitieren auf max layers
  if unfreeze_last_n > 0:
    num_layers = model.esm.config.num_hidden_layers
    unfreeze_last_n = min(unfreeze_last_n, num_layers)
    print(f"Model layers: {num_layers}, request unfreeze layers: {unfreeze_last_n}")
    for i in range(num_layers - unfreeze_last_n, num_layers):
      for param in model.esm.encoder.layer[i].parameters():
        param.requires_grad = True
      #adjust the esmdropout
      model.esm.encoder.layer[i].output.dropout.p = esm_dropout
      # adjust the attention_dropout
      model.esm.encoder.layer[i].attention.output.dropout.p=attention_dropout

  model.classifier.dropout.p=output_dropout
  return model

def preprocess_batch(examples, 
                     tokenizer:EsmTokenizer,
                     label2id:Dict[str, int], 
                     unique_items:List[str], 
                     transforms = None,
                     max_length=1024):
  """
  Preprocesses a batch of items from the dataset.
  Added safety checks for keys to prevent KeyError during iteration.
  """
  # Handle sequences
  seq_key = "sequence" if "sequence" in examples else "input_ids" # Fallback if already partially processed
  if seq_key not in examples:
      return examples

  sequences_processed = [item for item in examples[seq_key]] if isinstance(examples[seq_key], list) else [examples[seq_key]]

  # Tokenize
  tokenized_output = tokenizer(sequences_processed, max_length=max_length, truncation=True)

  # Process labels
  labels_list = []
  if "labels" in examples:
    go_term_lists_processed = examples["labels"] if isinstance(examples["labels"], list) else [examples["labels"]]

    for go_terms_str in go_term_lists_processed:
      labels = torch.full((len(unique_items),), 0, dtype=torch.float)
      if isinstance(go_terms_str, str):
        for item in go_terms_str.split(" "):
          if item in label2id:
            labels[label2id[item]] = 1
      labels_list.append(labels)

  # Construct output safely
  output_dict = {
      "input_ids": [torch.tensor(ids, dtype=torch.long) for ids in tokenized_output["input_ids"]],
      "attention_mask": [torch.tensor(ids, dtype=torch.long) for ids in tokenized_output["attention_mask"]]
  }

  if "protein" in examples:
      output_dict["protein"] = examples["protein"]

  if "labels" in examples:
      output_dict["labels"] = labels_list

  return output_dict

def data_collate_function(preprocessed_batch: List[Dict[str, Any]])-> Dict[str, Any]:
  """
  Stacks together groups of preprocessed samples into batches for our model,
  applying dynamic padding to the maximum sequence length within the batch.

  Params:
  **preprocessed_batch**: A list of Dicts, where each Dict represents a single processed sample
                          with unpadded 1D tensors for input_ids, token_type_ids, attention_mask, and labels.
  """
  collated_data = {}

  # Determine the maximum sequence length in the current batch
  max_seq_len = MAX_LENGTH

  # Manually pad each item in the batch
  padded_input_ids = []
  padded_token_type_ids = []
  padded_attention_mask = []
  labels = []

  for sample in preprocessed_batch:
    current_len = len(sample["input_ids"])
    padding_len = max_seq_len - current_len

    # Pad input_ids with the tokenizer's pad_token_id
    padded_input_ids.append(torch.cat([sample["input_ids"], torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)]))
    # Pad token_type_ids with 0
    if "token_type_ids" in sample:
      padded_token_type_ids.append(torch.cat([sample["token_type_ids"], torch.full((padding_len,), 0, dtype=torch.long)]))
    # Pad attention_mask with 0
    padded_attention_mask.append(torch.cat([sample["attention_mask"], torch.full((padding_len,), 0, dtype=torch.long)]))
    if "labels" in sample:
      labels.append(sample["labels"])

  # Stack the padded tensors to form the batch
  collated_data["input_ids"] = torch.stack(padded_input_ids)
  if padded_token_type_ids != []:
    collated_data["token_type_ids"] = torch.stack(padded_token_type_ids)
  collated_data["attention_mask"] = torch.stack(padded_attention_mask)
  if labels != []:
    collated_data["labels"] = torch.stack(labels)

  return collated_data


dataset = load_dataset(path="kelmoir/CAFA6")

tokenizer = create_esm_tokenizer()

preprocess_batch_partial = partial(preprocess_batch,
                                   tokenizer = tokenizer,
                                   transforms = None,
                                   max_length = MAX_LENGTH,
                                   unique_items=unique_aspects)

processed_dataset={"train": dataset["train"].with_transform(transform=preprocess_batch_partial),
                  "validation": dataset["validation"].with_transform(transform=preprocess_batch_partial),
                  "test": dataset["test"].with_transform(transform=preprocess_batch_partial)}



model_esm = create_esm(unfreeze_last_n=4,
                       model_type=SELECTED_MODEL,
                       output_dropout=0.2,
                       esm_dropout=0.0,
                       attention_dropout=0.0)

train_dataloader = torch.utils.data.DataLoader(processed_dataset["train"],
                                               batch_size=32,
                                               shuffle=True,
                                               collate_fn=coll)

# 2. Optimizer und Scheduler
optimizer = AdamW(model_esm.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)

# 3. Loss mit Label Smoothing
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# 4. Training Loop
model_esm.train()
for epoch in range(10):
    for batch in train_dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)  # Annahme: Labels für MLM (Masked Language Modeling)

        # Forward Pass
        outputs = model_esm(inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Reshape für CrossEntropyLoss (combines log_softmax + nll_loss)
        logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
        labels = labels.view(-1)  # [batch_size * seq_len]

        # Loss berechnen
        loss = criterion(logits, labels)

        # Backward Pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model_esm.parameters(), max_norm=1.0)

        # Optimizer Step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()



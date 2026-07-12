from Bio import SeqIO
import pandas as pd
import obonet
from pathlib import Path
import numpy as np
from progressbar import progressbar
import datasets
import random

prot_sequences ={}
unique_prot = []


def generate_dataset(input_data: pd.DataFrame,
                     label: str,
                     label2id: dict,
                     protein_ids=None,
                     constraint=lambda: True,
                     prot_sequences=prot_sequences,
                     max_length=1024) -> datasets.Dataset:
  """
  Generates a dataset from the input data with the given constraints.

  The labels are encoded as integer IDs using `label2id`, and the dataset
  features are annotated so Hugging Face tooling can understand the label space.
  """
  label_idx = input_data.columns.get_loc(label)+1
  selected_protein_ids = set(protein_ids) if protein_ids is not None else None
  label_names = []
  sequences = []
  features = []
  last_feature = input_data["EntryID"][0]
  label_string = ""
  for item in progressbar(input_data.itertuples(), total=input_data.shape[0], desc="creating data as aligned lists"):
    if item[1] != last_feature:
      if len(prot_sequences[last_feature]) <= max_length and (selected_protein_ids is None or last_feature in selected_protein_ids):
        label_names.append([token for token in label_string.split() if token in label2id])
        sequences.append(prot_sequences[last_feature])
        features.append(last_feature)
      last_feature = item[1]
      label_string = item[label_idx]
    elif item[label_idx] not in label_string:
      label_string += " " + item[label_idx]
  #handle last item
  if len(prot_sequences[last_feature]) <= max_length and (selected_protein_ids is None or last_feature in selected_protein_ids):
    label_names.append([token for token in label_string.split() if token in label2id])
    sequences.append(prot_sequences[last_feature])
    features.append(last_feature)
  print(f"[INFO] Generating Dataset with {len(features)} items, with a label of {label}.")
  print(f"[INFO] Number of proteins in the dataset: {len(features)}")
  print(f"[INFO] Number of proteins removed from data source: {len(prot_sequences) - len(features)}")
  label_feature = datasets.ClassLabel(
      names=sorted(label2id.keys())
  )
  dataset_dict = {
      "protein": features,
      "labels": label_names,
      "sequence": sequences,
  }
  datset_features = datasets.Features({
                                    "protein": datasets.Value("string"),
                                    "labels": datasets.Sequence(label_feature),
                                    "sequence": datasets.Value("string"),
                                })
  dataset_info = datasets.DatasetInfo(description="CAFA-6 based datasets for training ans exploring purposes",
                                citation= "Iddo Friedberg, Predrag Radivojac, Paul D Thomas, An Phan, M. Clara De Paolis Kaluza, Damiano Piovesan, Parnal Joshi, Chris Mungall, Martyna Plomecka, Walter Reade, and María Cruz. CAFA 6 Protein Function Prediction. https://kaggle.com/competitions/cafa-6-protein-function-prediction, 2025. Kaggle.",
                                features= datset_features,
                                )
  return datasets.Dataset.from_dict(
      dataset_dict,
      datset_features,
      info=dataset_info,
  )

# Create train, test, validation split
# Split the data into train and test splits
def create_train_val_split_from_dataset(dataset):
  # Shuffle first to ensure indices are randomized before splitting
  dataset = dataset.shuffle(seed=42)

  dataset_split = dataset.train_test_split(test_size=0.2, seed=42)

  # split the 80% train into train and validation (70/30 of the 80%)
  dataset_val_split = dataset_split["train"].train_test_split(test_size=0.3, seed=42)

  return datasets.DatasetDict({
      "train": dataset_val_split["train"],
      "validation": dataset_val_split["test"],
      "test": dataset_split["test"]
  })


data_path = Path("./data/cafa-6-protein-function-prediction")
for sequence in SeqIO.parse(data_path/"Train/train_sequences.fasta", "fasta"):
  unique_prot.append(sequence.id.split("|")[1])
  prot_sequences[sequence.id.split("|")[1]] = str(sequence.seq).upper().replace(r"[UZOB]", "X") # Reading the relevant data, and pre-formatting it for ProtBERT
train_df = pd.read_csv(data_path/"Train/train_terms.tsv", delimiter="\t")
obonet_graph = obonet.read_obo(data_path/"Train/go-basic.obo")
unique_go_terms = np.unique(train_df["term"])
unique_aspects = np.unique(train_df["aspect"])
# generate id2label and label2id datasets for the model to use (now for aspects)
id2label = {id: label for id, label in enumerate(unique_aspects)}
label2id = {label: id for id, label in enumerate(unique_aspects)}

# 1. Get unique protein IDs
unique_protein_ids = list(train_df['EntryID'].unique())
random.seed(42)
random.shuffle(unique_protein_ids)

# 2. Split IDs manually (80/20 split)
test_size = int(len(unique_protein_ids) * 0.18)
train_val_ids = unique_protein_ids[:-test_size]
test_ids = unique_protein_ids[-test_size:]

# 3. Split Train/Val IDs (80/20 of the 80%)
val_size = int(len(train_val_ids) * 0.22)
train_ids = train_val_ids[:-val_size]
val_ids = train_val_ids[-val_size:]

print(f"Splitting IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

# 4. Generate disjoint datasets
dataset_dict = datasets.DatasetDict({
    "train": generate_dataset(train_df, "aspect", label2id, train_ids),
    "validation": generate_dataset(train_df, "aspect", label2id, val_ids),
    "test": generate_dataset(train_df, "aspect", label2id, test_ids),
})

dataset_dict.save_to_disk(data_path/"dataset")
dataset_dict.push_to_hub("kelmoir/cafa6", private=False)

print("\nDisjoint datasets created successfully.")

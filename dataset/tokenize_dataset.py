import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer

dataset_path = '../.dataset/hdfs/train.csv'
data = pd.read_csv(dataset_path)

# Prepare dataset for Hugging Face
data = data["Content"].tolist()
dataset_dic = {"text": data}

# Split into train, validation, and test datasets
split = int(len(data) * 0.8)
validation_split = int(len(data) * 0.9)

dataset = DatasetDict({
    "train": Dataset.from_dict({"text": data[:split]}),
    "validation": Dataset.from_dict({"text": data[split:validation_split]}),
    "test": Dataset.from_dict({"text": data[validation_split:]})
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

#Save the tokenized dataset
tokenized_datasets.save_to_disk('../.dataset/hdfs/tokenized')
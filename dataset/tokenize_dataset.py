import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_path = '../.dataset/hdfs/train.csv'
data = pd.read_csv(dataset_path)

# Prepare dataset for Hugging Face
data = data["Content"].tolist()
dataset_dic = {"text": data}

dataset = DatasetDict({
    "train": Dataset.from_dict({"text": data})
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

#Save the tokenized dataset
tokenized_datasets.save_to_disk('../.dataset/hdfs/tokenized')
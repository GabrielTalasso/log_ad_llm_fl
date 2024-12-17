import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
import json
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Step 1: Load Dataset
dataset_path = '../.dataset/hdfs/test.csv'

lora_path = "lora-smol/checkpoint-11162"

nrows = 10000
data = pd.read_csv(dataset_path, nrows=nrows)
print(len(data[data['Label'] == 1]))

# Prepare dataset for Hugging Face

data = data["Content"].tolist()
dataset_dic = {"text": data}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
tokenizer.pad_token = tokenizer.eos_token

#tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenize_dataset = Dataset.from_dict(dataset_dic)
tokenized_datasets = tokenize_dataset.map(tokenize_function, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_dataset, tokenizer):
    model.eval()
    total_loss = 0
    num_batches = 0
    losses = []
    with torch.no_grad():
        for batch in test_dataset['text']:
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            labels = inputs.input_ids.clone()
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            losses.append(outputs.loss.item())
            num_batches += 1
    return total_loss / num_batches, losses

model_pt =  AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M").to(device)

eval_pt, losses_pt = evaluate_model(model_pt, dataset_dic, tokenizer)
print(f"Pre-trained model loss: {eval_pt}")

model_with_adapters = PeftModel.from_pretrained(model_pt, lora_path)
model_with_adapters.to(device)

eval_peft, losses_ft = evaluate_model(model_with_adapters, dataset_dic, tokenizer)

print(f"PEFT model loss: {eval_peft}")

data = pd.read_csv(dataset_path, nrows=nrows)
data['losses_pt'] = losses_pt
data['losses_ft'] = losses_ft

data.to_csv("eval.csv", index=False)
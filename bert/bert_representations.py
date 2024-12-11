import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel 
import numpy as np
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Step 1: Load Dataset
dataset_path = '../.dataset/hdfs/test.csv'

lora_path = "lora-bert/checkpoint-10000"

nrows = 10000
data = pd.read_csv(dataset_path, nrows=nrows)
print(len(data[data['Label'] == 1]))

# Prepare dataset for Hugging Face

data = data["Content"].tolist()
dataset_dic = {"text": data}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings(model, dataset, tokenizer):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in dataset['text']:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Forward pass with output of hidden states enabled
            outputs = model(**inputs, output_hidden_states=True)
            
            # Use the last hidden state (hidden_states[-1]) and take the CLS token representation
            hidden_states = outputs.hidden_states[-1]
            embeddings.append(hidden_states[:, 0, :].squeeze().cpu().numpy())  # CLS token representation
    
    return np.array(embeddings)


model_pt = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states = True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model_pt.to(device)

embeddings_pt = generate_embeddings(model_pt, dataset_dic, tokenizer)
print(f"Pre-trained model embeddings: {len(embeddings_pt)}")

model_with_adapters = PeftModel.from_pretrained(model_pt, lora_path)
#model_with_adapters = AutoModelForMaskedLM.from_pretrained('full-lora-bert-finetuned', output_hidden_states = True) #full fine tuned
model_with_adapters.to(device)

embeddings_peft = generate_embeddings(model_with_adapters, dataset_dic, tokenizer)
print(f"PEFT model embeddings: {len(embeddings_peft)}")


np.savez("embeddings.npz", embeddings_pt=embeddings_pt, embeddings_peft=embeddings_peft,
         labels = pd.read_csv(dataset_path, nrows=nrows)['Label'].values)

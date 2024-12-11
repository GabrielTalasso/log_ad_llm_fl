import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel 
import numpy as np
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Step 1: Load Dataset
dataset_path = '../.dataset/hdfs/test.csv'

lora_path = "lora-smol/checkpoint-4000"

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
    #with torch.no_grad():
    #    for text in dataset['text']:
    #        # Tokenize input
    #        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    #        outputs = model(**inputs, output_hidden_states=True)
    #        hidden_states = outputs.hidden_states   
    #        embeddings.append(hidden_states[-1]).cpu().numpy()
    
    pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device)
    with torch.no_grad():
        text = dataset['text']
        outputs = pipe(text, return_tensors='pt')
    
    for i in range(len(outputs)):
        #(1, 560, 49152)
        emb = outputs[i].cpu().numpy()[0].mean(axis=0) 
        embeddings.append(emb)
    return embeddings #np.array(embeddings)


model_pt = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M").to(device)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
tokenizer.pad_token = tokenizer.eos_token
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

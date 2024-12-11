import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM ,TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_in_top_k(top_k, target):
    #implement in efficient way
    return target in top_k

def next_token_top_k(data, model, tokenizer, k):

    """
    Predict the next token for a given text using a language model.
    Calculate with the correct token is in the top k predictions.
    """
    accuracies = []
    model.to(device)
    for text in data["text"]:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: inputs[key].to(device) for key in inputs}

        outputs = model(**inputs)
        logits = outputs.logits

        top_k = torch.topk(logits, k, dim=-1).indices
        top_k = top_k.cpu().numpy() 
        
        #calculate the number of correct tokens in the top k predictions
        tokens = inputs['input_ids'][0].cpu().numpy()
        correct = sum(is_in_top_k(top_k[0], token) for token in tokens)
    
        #print(f"Correct tokens in this example: {correct}/{len(inputs['input_ids'][0])}. Accuracy: {correct/len(inputs['input_ids'][0])}")
        accuracies.append(correct/len(inputs['input_ids'][0]))
    
    return accuracies

def prepare_dataset(dataset_path, nrows = None):
    data = pd.read_csv(dataset_path, nrows=nrows)
    labels = data["Label"].tolist()
    data = data["Content"].tolist()
    dataset_dic = {"text": data}
    return dataset_dic, labels

def main(model_name, dataset_dic, labels, top_k = 5, lora_path = None):
    
    model_pretrained = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if lora_path:
        model = PeftModel.from_pretrained(model_pretrained, lora_path)
    else:
        model = model_pretrained

    model.to(device)
    accuracies = next_token_top_k(dataset_dic, model, tokenizer, top_k)

    return pd.DataFrame({"accuracy": accuracies, "label": labels})

if __name__ == "__main__":

    dataset_path = '../.dataset/hdfs/test.csv'
    dataset_dic, labels = prepare_dataset(dataset_path, nrows = 10000)

    model_name = "HuggingFaceTB/SmolLM-360M"
    lora_path = "models/checkpoint-4000_lora8"
    top_k = 5
    results = main(model_name, dataset_dic, labels, top_k, lora_path)

    results.to_csv(f"results/results_{lora_path.split('/')[-1]}.csv", index=False)

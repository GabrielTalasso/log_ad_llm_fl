import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
import json
import random
from typing import List, Tuple
from datasets import Dataset

def initialize_model(model_name, lora_rank = 8, lora = True):

    """
    Initialize model and tokenizer from Hugging Face model name
    If lora is True, apply LoRA to the model, else will fine-tune full model
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if lora:
        lora_config = LoraConfig(
                    r= lora_rank,
                    lora_alpha= lora_rank*2,
                    lora_dropout=0.1
                )
        
        model = get_peft_model(model, lora_config)

    return model, tokenizer

def load_dataset(path, nrows = None):
    
    """
    Load huggingface dataset in path
    """
    dataset = DatasetDict.load_from_disk(path, keep_in_memory=True)

    if nrows:
        dataset["train"] = dataset["train"].select(list(range(nrows)))
    
    #create labels
    dataset["train"] = dataset["train"].map(lambda x: {"labels": x["input_ids"]}, batched=True)
    
    return dataset

def split_data(dataset, num_clients, eval_split = 0.1):
    """
    Split dataset into num_clients for federated learning.
    Returns a tuple of lists containing the training and evaluation datasets for each client.
    """
    indices = list(range(len(dataset['train']['text'])))

    clients_indices = []
    for i in range(num_clients):
        client_indices = indices[i::num_clients]
        clients_indices.append(client_indices)

    clients_datasets_train = []
    clients_datasets_eval = []

    for i in range(num_clients):
        client_data_indices = clients_indices[i]

        eval_size = int(len(client_data_indices) * eval_split)
        train_indices = client_data_indices[eval_size:]
        eval_indices = client_data_indices[:eval_size]

        client_dataset_train = dataset['train'].select(train_indices)
        client_dataset_eval = dataset['train'].select(eval_indices)

        clients_datasets_train.append(client_dataset_train)
        clients_datasets_eval.append(client_dataset_eval)

    return clients_datasets_train, clients_datasets_eval


def train_client(client, client_dataset, model, round, sim_name, tokenizer, 
                 epochs=1, batch_size=32, max_steps=100):

    """
    Train model on client dataset
    """

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./fl-results",
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=2e-5,
        max_steps=max_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=max_steps+1, 
    )

    class CustomTrainer(Trainer):
        def __init__(self, client, **kwargs):
            super().__init__(**kwargs)
            self.train_losses = {}
            self.validation_losses = {}
            self.client = client
    
        def log(self, logs):
            # Save client losses
            super().log(logs)
            if "loss" in logs:
                self.train_losses[client] = float(logs["loss"])
            if "eval_loss" in logs:
                self.validation_losses[client] = float(logs["eval_loss"])
    
    trainer = CustomTrainer(client=client,
                            model=model,
                            args=training_args,
                            train_dataset=client_dataset,
                            tokenizer=tokenizer,
                            eval_dataset=client_dataset)
    
    trainer.train()

    # Save model
    output_dir = f"./fl-results/{sim_name}/round_{round}/client_{client}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # Save losses
    with open(f"{output_dir}/training_losses.json", "w") as f:
        json.dump(trainer.train_losses, f)
    
    with open(f"{output_dir}/validation_losses.json", "w") as f:
        json.dump(trainer.validation_losses, f)
    
    return model

def get_adapters(model):
    adapter_weights = {}
    
    for name, param in model.named_parameters():
        # Check if the parameter belongs to an adapter layer (e.g., contains "lora")
        if "lora" in name:
            adapter_weights[name] = param.detach().cpu()  # Detach and move to CPU for saving
    
    return adapter_weights

def get_adapters(model):
    """
    Extract LoRA adapter weights from the model.
    Assumes that LoRA layers are identifiable by specific names or attributes.
    """
    adapters = {}
    for name, param in model.named_parameters():
        if "lora_" in name:  # Assume LoRA layers are named with "lora_"
            adapters[name] = param.data.clone()  # Clone to avoid modifying the original
    return adapters

def set_adapters(model, aggregated_adapters):
    """
    Update the model with aggregated LoRA adapter weights.
    """
    for name, param in model.named_parameters():
        if name in aggregated_adapters:
            param.data.copy_(aggregated_adapters[name])  # Update the parameter

def aggregate_models(models, lora=True):
    """
    Aggregate models by the mean (FedAvg).
    If LoRA, aggregate only the adapters.
    """
    if lora:
        all_adapters = [get_adapters(model) for model in models]
        
        aggregated_adapters = {}
        
        for key in all_adapters[0].keys():
            aggregated_adapters[key] = torch.mean(
                torch.stack([adapters[key] for adapters in all_adapters]), dim=0
            )
        
        return aggregated_adapters
    
    else:
        # Aggregate the entire model (not just adapters)
        global_model = models[0]
        for name, param in global_model.named_parameters():
            param.data.copy_(torch.mean(
                torch.stack([model.state_dict()[name].data for model in models]), dim=0
            ))
        return global_model

def save_global_model(global_model, round, sim_name):
    output_dir = f"./fl-results/{sim_name}/round_{round}/global_model"
    os.makedirs(output_dir, exist_ok=True)
    global_model.save_pretrained(output_dir)

    


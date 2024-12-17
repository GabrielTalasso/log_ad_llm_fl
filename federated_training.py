import os
import random
from utils import initialize_model, load_dataset, split_data, train_client, aggregate_models, set_adapters, save_global_model, get_adapters, cosine_learning_rate
import warnings
import torch
import hashlib
import numpy as np
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

SIM_NAME = 'experiment_smol135_lora32'
NUM_ROUNDS = 50
NUM_CLIENTS = 50
CLIENT_FRAC = 0.1
MODEL_NAME = 'HuggingFaceTB/SmolLM-135M'
path = '../.dataset/hdfs/tokenized'

global_model, tokenizer = initialize_model(MODEL_NAME, lora_rank=32, sim_name=SIM_NAME)
rs = random.SystemRandom()
tokenized_datasets = load_dataset(path, nrows=None)
clients_datasets, clients_datasets_eval = split_data(tokenized_datasets, NUM_CLIENTS)

for round in range(1, NUM_ROUNDS+1):

    #Select Clients
    clients = rs.sample(list(range(NUM_CLIENTS)), int(NUM_CLIENTS*CLIENT_FRAC))
    
    print(f"Round {round}: Clients Selected {clients}")

    # Train clients
    clients_models = []
    for client in clients:
        new_lr = cosine_learning_rate(current_round = round, total_rounds = NUM_ROUNDS, initial_lr=0.001)
        client_model = train_client(int(client), clients_datasets[client], round, SIM_NAME, tokenizer,
                                    max_steps=10, lr = new_lr, batch_size=32, model_name = MODEL_NAME)
        
        clients_models.append(client_model)
        print(f"Round {round}: Client {client} trained")

    # Aggregate model
    aggregated_adapters = aggregate_models(clients_models)
    set_adapters(global_model, aggregated_adapters)
    save_global_model(global_model, round, SIM_NAME)

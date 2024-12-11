import numpy as np
import os
from transformers import AutoTokenizer
from utils import initialize_model, load_dataset, split_data, train_client, aggregate_models, set_adapters, save_global_model
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SIM_NAME = 'first_simulation'
NUM_ROUNDS = 20
NUM_CLIENTS = 50
CLIENT_FRAC = 0.1
MODEL_NAME = 'HuggingFaceTB/SmolLM-360M'
path = '../.dataset/hdfs/tokenized'

global_model, tokenizer = initialize_model(MODEL_NAME)
tokenized_datasets = load_dataset(path, nrows=100000)
clients_datasets, clients_datasets_eval = split_data(tokenized_datasets, NUM_CLIENTS)

for round in range(1, NUM_ROUNDS+1):

    #Select Clients
    clients = np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS*CLIENT_FRAC), replace=False)
    
    print(f"Round {round}: Clients Selected {clients}")

    # Train clients
    clients_models = []
    for client in clients:
        client_model = train_client(int(client), clients_datasets[client], global_model, round, SIM_NAME, tokenizer)
        clients_models.append(client_model)
        print(f"Round {round}: Client {client} trained")

    # Aggregate model
    aggregated_adapters = aggregate_models(clients_models)
    set_adapters(global_model, aggregated_adapters)

    save_global_model(global_model, round, SIM_NAME)

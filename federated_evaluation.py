import os
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import json
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_in_top_k(top_k, target):
    return target in top_k

#tokenize
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def load_global_testset(nrows = 1000):

    dataset_path = '../.dataset/hdfs/test.csv'
    data = pd.read_csv(dataset_path, nrows=nrows)

    labels = data["Label"].tolist()
    data = data["Content"].tolist()
    dataset_dic = {"text": data}

    return Dataset.from_dict(dataset_dic), labels

def eval_global(calcule_topk = False, k = 5, nrows = 1000):
    results = []
    detection_results = []
    for round in range(1, len(os.listdir(experiment_path))):
        result = {}

        global_model_path = os.path.join(experiment_path, 'round_'+str(round), 'global_model')
        global_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")   
        global_model = PeftModel.from_pretrained(global_model, global_model_path)
        global_model.to(device)

        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
        tokenizer.pad_token = tokenizer.eos_token
        test_dataset, labels_anom = load_global_testset(nrows = nrows)
        if calcule_topk and round == 1:
            print('Calculating topk')
            # count number of ones
            ones = sum(labels_anom)
            zeros = len(labels_anom) - ones
            print(f'Anomalies: {ones}, Non-Anomalies: {zeros}')

        total_loss = 0
        num_batches = 0

        accuracies = []
        with torch.no_grad():
            global_model.eval()
            for batch in test_dataset['text']:
                inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                labels = inputs.input_ids.clone()
                outputs = global_model(**inputs, labels=labels)

                if calcule_topk:
                    logits = outputs.logits
                    top_k = torch.topk(logits, k, dim=-1).indices
                    top_k = top_k.cpu().numpy() 
                    tokens = inputs['input_ids'][0].cpu().numpy()
                    correct = sum(is_in_top_k(top_k[0], token) for token in tokens)
                
                    accuracies.append(correct/len(tokens))

                total_loss += outputs.loss.item()
                num_batches += 1
        print(f"Round {round} global val loss: {total_loss / num_batches}")
        if calcule_topk:
            #acc to cpu
            accuracies = np.array(accuracies)
            detection_result = pd.DataFrame({"accuracy": accuracies, "label": labels_anom})
            detection_result['round'] = round

        if round == 1:
            detection_results = detection_result
    
        else:
            detection_results = pd.concat([detection_results, detection_result], axis=0)
            
        result['round'] = round
        result['eval_global_loss'] = total_loss / num_batches
        results.append(result)

        df_results = pd.DataFrame(results)
        sim_name = experiment_path.split('/')[-1]
        df_results.to_csv(f'results_{sim_name}.csv', index=False)

        df_detection = pd.DataFrame(detection_results)
        df_detection.to_csv(f'detection_{sim_name}.csv', index=False)

    return results, detection_results

def aggregate_client_train_losses():
    results = []
    print('Training losses ---------')

    for round in range(1, len(os.listdir(experiment_path))):
        result = {}
        client_losses = []
        for client in os.listdir(os.path.join(experiment_path, 'round_'+str(round))):
            if client != 'global_model':
                client_loss_path = os.path.join(experiment_path, 'round_'+str(round), client, 'training_losses.json')
                with open(client_loss_path, 'r') as f:
                    client_loss = json.load(f)
                    client_losses.append(client_loss[str(client.split('_')[-1])])
        result['round'] = round
        result['client_losses_train'] = np.mean(client_losses)
        results.append(result)

        print(f"Round {round} client train loss: {np.mean(client_losses)}")

    return results

if __name__ == '__main__':
    experiment_path = 'fl-results/lr_schedule_cosine'
    results_global, detect = eval_global(calcule_topk = True, nrows = 3000)
    results_local = aggregate_client_train_losses()

    df_local = pd.DataFrame(results_local)
    df_results = pd.DataFrame(results_global)

    df_results['client_losses_train'] = df_local['client_losses_train']
    
    sim_name = experiment_path.split('/')[-1]
    df_results.to_csv(f'results_{sim_name}.csv', index=False)

    if len(detect) > 0:
        df_detection = pd.DataFrame(detect)
        df_detection.to_csv(f'detection_{sim_name}.csv', index=False)
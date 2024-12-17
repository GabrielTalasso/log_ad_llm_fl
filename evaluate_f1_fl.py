import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_in_top_k(top_k, target):
    #implement in efficient way
    return target in top_k


def next_token_top_k(data, model, tokenizer):

    """
    Predict the next token for a given text using a BERT model.
    Calculate with the correct token is in the top k predictions.
    """
    accuracies = {'top1':[], 'top3':[], 'top5':[], 'top10':[]}

    model.to(device)
    for text in data["text"]:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(device) for key in inputs}

        outputs = model(**inputs)
        logits = outputs.logits

        for k in [1, 3, 5, 10]:
            top_k = torch.topk(logits, k, dim=-1).indices
            top_k = top_k.cpu().numpy() 
            
            tokens = inputs['input_ids'][0].cpu().numpy()
            correct = sum(is_in_top_k(top_k[0], token) for token in tokens)

            accuracies[f'top{k}'].append(correct/len(tokens))
        
    return accuracies

dataset_path = '../.dataset/hdfs/test.csv'

nrows = 10000
N_ROUNDS = 50
SIM_NAME  = "experiment_small"
model_name = "HuggingFaceTB/SmolLM-360M"

data = pd.read_csv(dataset_path, nrows=nrows)
print(len(data[data['Label'] == 1]))

labels = data["Label"].tolist()
data = data["Content"].tolist()
dataset_dic = {"text": data}
df_acc = pd.DataFrame()
df_acc.to_csv(f"results_accs_{SIM_NAME}.csv", index=False)

df_f1 = pd.DataFrame()
df_f1.to_csv(f"results_f1_{SIM_NAME}.csv", index=False)

for round in range(1, N_ROUNDS+1):

    lora_path = f"fl-results/{SIM_NAME}/round_{round}/global_model"

    model_pretrained = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_ft = PeftModel.from_pretrained(model_pretrained, lora_path)
    model_ft.to(device)

    accuracies = next_token_top_k(dataset_dic, model_ft, tokenizer)
    for k in [1, 3, 5, 10]:

        df_results = pd.DataFrame(accuracies)
        df_results['label'] = labels
        df_results['round'] = round 
        df_results['k'] = k
    
        df_acc = pd.concat([df_acc, df_results])

        df_acc.to_csv(f"results_accs_{SIM_NAME}.csv", index=False)

        ths = np.linspace(0, 1, 1000)

        best_f1 = 0
        best_th = 0

        for th in ths:
            df_results['pred'] = df_results[f'top{k}'] < th
            df_results['pred'] = df_results['pred'].astype(int)

            f1 = f1_score(df_results['label'], df_results['pred'])
            precision = precision_score(df_results['label'], df_results['pred'])
            recall = recall_score(df_results['label'], df_results['pred'])

            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        df_results['pred'] = df_results[f'top{k}'] < best_th
        df_results['pred'] = df_results['pred'].astype(int)

        f1 = f1_score(df_results['label'], df_results['pred'])
        precision = precision_score(df_results['label'], df_results['pred'])
        recall = recall_score(df_results['label'], df_results['pred'])

        print(f'Round: {round}')
        print(f'K: {k}')
        print(f'Threshold: {best_th}')
        print(f'F1: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

        df_f1 = pd.concat([df_f1, pd.DataFrame({"round": [round], "k": [k], "threshold": [best_th], "f1": [f1], "precision": [precision], "recall": [recall]})])
        df_f1.to_csv(f"results_f1_{SIM_NAME}.csv", index=False)




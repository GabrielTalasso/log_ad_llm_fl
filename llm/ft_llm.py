import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#load nrows
nrows = None

tokenized_datasets = pd.read_csv('../.dataset/hdfs/train.csv')
tokenized_datasets = tokenized_datasets[tokenized_datasets['Label'] == 0]
# Prepare dataset for Hugging Face
if nrows:
    data = tokenized_datasets["Content"].head(nrows).tolist()
else:
    data = tokenized_datasets["Content"].tolist()
dataset_dic = {"text": data}


# Split into train, validation, and test datasets
split = int(len(data) * 0.8)
validation_split = int(len(data) * 0.9)

tokenized_datasets = DatasetDict({
    "train": Dataset.from_dict({"text": data[:split]}),
    "validation": Dataset.from_dict({"text": data[split:validation_split]}),
    "test": Dataset.from_dict({"text": data[validation_split:]})
})

# Tokenize datasets
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function_with_labels(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    # Add labels by duplicating input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize datasets
tokenized_datasets = tokenized_datasets.map(tokenize_function_with_labels, batched=True)

# LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.1
)

# Load model and apply LoRA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
model = get_peft_model(model, lora_config)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./lora-smol",
    logging_dir="./logs",
    logging_steps=100,  # Adjusted for frequent updates
    learning_rate=2e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    per_device_train_batch_size=32,  # Reduced for memory constraints
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=10000,  # Adjusted for smaller datasets
    fp16=torch.cuda.is_available(),
)

# Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_losses = []
        self.validation_losses = []

    def log(self, logs):
        super().log(logs)
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.validation_losses.append(logs["eval_loss"])

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model and training losses
output_dir = "./lora-smol-finetuned"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

with open(f"{output_dir}/training_losses.json", "w") as f:
    json.dump(trainer.train_losses, f)
with open(f"{output_dir}/validation_losses.json", "w") as f:
    json.dump(trainer.validation_losses, f)

print("Training completed and model saved!")

from torch.utils.data import DataLoader

# Function to evaluate the model on the test set
def evaluate_model(model, tokenizer, dataset, device, batch_size=8):
    model.eval()
    model.to(device)
    total_loss = 0
    num_batches = 0

    # Prepare the test data for evaluation with padding and truncation
    def collate_fn(batch):
        texts = [example["text"] for example in batch]
        tokenized = tokenizer(
            texts,
            padding="longest",  # Pad to the longest sequence in the batch
            truncation=True,    # Ensure inputs are truncated to max_length
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    test_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Average Loss on Test Set: {avg_loss}")
    return avg_loss

model_pt = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")

# Evaluate the pre-trained model
print("Evaluating the pre-trained model...")
pretrained_loss = evaluate_model(model_pt, tokenizer, tokenized_datasets["test"], device)

lora_path = "lora-smol-finetuned"  
model_ft = PeftModel.from_pretrained(model_pt, lora_path)

# Evaluate the fine-tuned model
print("Evaluating the fine-tuned model...")
fine_tuned_loss = evaluate_model(model, tokenizer, tokenized_datasets["test"], device)

print(f"Pre-trained Model Loss: {pretrained_loss}")
print(f"Fine-tuned Model Loss: {fine_tuned_loss}")
